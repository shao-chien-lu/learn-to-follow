// cppimport
#include "follower.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

// number of past steps the memory channels cover, as MEMORY_STEPS in follower/preprocessing.py
static const int MEMORY_STEPS = 8;
// pogema's action order: stay, up, down, left, right as (di, dj)
static const int MOVES[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
// LTF_FIX_SOFT_POSITIONS=1: set_state clears every agent's old cell before marking the new ones, so no agent is erased,
// matching follower/pogema_fix.py on the training side
static const bool FIX_SOFT_POSITIONS = [] {
    const char* value = std::getenv("LTF_FIX_SOFT_POSITIONS");
    return value != nullptr && std::string(value) == "1";
}();

// Read from agents_pos rather than the agent list, so that the occupation costs see the same agents as training does
// (the pogema agents window, see set_state).
std::list<std::pair<int, int>> Follower::get_occupied_cells(int agent_idx)
{
    std::pair<int, int> cur_pos = agents[agent_idx].cur_position;
    std::list<std::pair<int, int>> result;
    for(int i = cur_pos.first - cfg.obs_radius; i <= cur_pos.first + cfg.obs_radius; i++)
        for(int j = cur_pos.second - cfg.obs_radius; j <= cur_pos.second + cfg.obs_radius; j++)
            if(agents_pos[i][j] && std::make_pair(i, j) != cur_pos)
                result.push_back({i, j});
    return result;
}

// Whether agent a_id saw another agent on the absolute cell (i, j) `age` steps ago (1: the last step): the cell was in
// its observation window then, and not its own cell.
bool Follower::seen_occupied(size_t age, int a_id, int i, int j) const
{
    const auto& pos = position_history[age - 1][a_id];
    return std::abs(i - pos.first) <= cfg.obs_radius && std::abs(j - pos.second) <= cfg.obs_radius &&
           std::make_pair(i, j) != pos && agents_pos_history[age - 1][i][j];
}

std::vector<float> Follower::generate_input(int a_id, int radius, std::list<std::pair<int, int>> path)
{
    int obs_size = radius * 2 + 1;
    int num_neighbour_channels = cfg.neighbour_memory == "trail" ? 2 : 0;
    std::vector<float> input(obs_size*obs_size*((cfg.use_memory ? 4 : 2) + num_neighbour_channels), 0);
    for(int i = -radius; i <= radius; i++)
        for(int j = -radius; j <= radius; j++)
        {
            if(grid[agents[a_id].cur_position.first + i][agents[a_id].cur_position.second + j])
                input[(i + radius) * obs_size + j + radius] = -1;
            if(agents_pos[agents[a_id].cur_position.first + i][agents[a_id].cur_position.second + j])
                input[(i + radius) * obs_size + j + radius + obs_size * obs_size] = 1;
        }
    // as in training: the own cell is marked even when set_state erased it for this step
    if(agents[a_id].cur_position != agents[a_id].goal)
        input[radius * obs_size + radius + obs_size * obs_size] = 1;
    // As in training, the path is followed until it leaves the full observation window (cfg.obs_radius) and only then
    // cropped to the network window, so a path that leaves the network window and comes back is drawn in both parts.
    for(auto& node : path)
    {
        int di = node.first - agents[a_id].cur_position.first, dj = node.second - agents[a_id].cur_position.second;
        if(std::abs(di) > cfg.obs_radius || std::abs(dj) > cfg.obs_radius)
            break;
        if(std::abs(di) <= radius && std::abs(dj) <= radius)
            input[obs_size * (radius + di) + radius + dj] = 1;
    }
    // memory channels, as MemoryWrapper: 'blocked' (the cell the last action aimed at, while not moving), 'trail'
    if(cfg.use_memory)
    {
        float* blocked = input.data() + 2 * obs_size * obs_size;
        float* trail = blocked + obs_size * obs_size;
        if(stuck[a_id] > 0)
        {
            const int* move = MOVES[last_actions[a_id]];
            blocked[(radius + move[0]) * obs_size + radius + move[1]] =
                static_cast<float>(std::min(stuck[a_id], MEMORY_STEPS)) / MEMORY_STEPS;
        }
        int age = trails[a_id].size();
        for(auto it = trails[a_id].rbegin(); it != trails[a_id].rend(); ++it, --age)  // oldest first: the most recent wins
        {
            int di = it->first - agents[a_id].cur_position.first, dj = it->second - agents[a_id].cur_position.second;
            if(std::abs(di) <= radius && std::abs(dj) <= radius)
                trail[(radius + di) * obs_size + radius + dj] = static_cast<float>(MEMORY_STEPS + 1 - age) / MEMORY_STEPS;
        }
    }
    // memory of the other agents, as NeighbourMemoryWrapper, after the own memory: nb_trail (the most recent step the
    // cell was seen occupied), nb_wait (the steps directly before this one it was, if it is occupied now)
    if(num_neighbour_channels)
    {
        float* nb_trail = input.data() + (cfg.use_memory ? 4 : 2) * obs_size * obs_size;
        float* nb_wait = nb_trail + obs_size * obs_size;
        const auto& cur = agents[a_id].cur_position;
        for(int di = -radius; di <= radius; di++)
            for(int dj = -radius; dj <= radius; dj++)
            {
                int i = cur.first + di, j = cur.second + dj, cell = (radius + di) * obs_size + radius + dj;
                for(size_t age = 1; age <= agents_pos_history.size(); age++)
                    if(seen_occupied(age, a_id, i, j))
                    {
                        nb_trail[cell] = static_cast<float>(MEMORY_STEPS + 1 - age) / MEMORY_STEPS;
                        break;
                    }
                if(agents_pos[i][j] && std::make_pair(i, j) != cur)
                {
                    size_t steps = 0;
                    while(steps < agents_pos_history.size() && seen_occupied(steps + 1, a_id, i, j))
                        steps++;
                    nb_wait[cell] = static_cast<float>(steps) / MEMORY_STEPS;
                }
            }
    }
    // channel ablation, as AblateChannelsWrapper
    for(int channel : ablated_channels)
        std::fill(input.begin() + channel * obs_size * obs_size, input.begin() + (channel + 1) * obs_size * obs_size, 0.f);
    return input;
}

// Density gate of one agent from its own observation only: rho = the other agents in its cfg.obs_radius window (the
// cells get_occupied_cells reads, as the agents channel / the dynamic cost see them) / the free cells of the window
// other than its own; smoothed by an EMA (gate_alpha, first step: rho itself); g = clip((rho - rho0) / (rho1 - rho0)).
float Follower::update_gate(size_t agent_idx, size_t num_others)
{
    const auto& pos = agents[agent_idx].cur_position;
    int free_cells = 0;
    for(int i = pos.first - cfg.obs_radius; i <= pos.first + cfg.obs_radius; i++)
        for(int j = pos.second - cfg.obs_radius; j <= pos.second + cfg.obs_radius; j++)
            if(grid[i][j] == 0 && std::make_pair(i, j) != pos)
                free_cells++;
    float rho = free_cells > 0 ? static_cast<float>(num_others) / free_cells : 0.f;
    float& ema = rho_ema[agent_idx];
    ema = ema < 0.f ? rho : ema + gate_alpha * (rho - ema);
    float g = (ema - cfg.gate_rho0) / (cfg.gate_rho1 - cfg.gate_rho0);
    gates[agent_idx] = std::min(1.f, std::max(0.f, g));
    return gates[agent_idx];
}

// Network input of one agent, 2 (4 with cfg.use_memory) x (2r+1) x (2r+1): obstacles (-1) and the re-planned A* path
// (+1), other agents, and the memory channels.
// cfg.dflow: the moves of other agents that agent_idx can infer from its own last two views (agents carry no identity),
// added to its planner's flow after update_occupied_cells, i.e. after a goal-change reset, like the occupations. A move
// u -> v is credited when, with r = cfg.obs_radius:
//   - v (in the current window, not the own cell) is occupied now and was in the previous window but not occupied by
//     another agent then (an agent seen on v on both steps may have stayed: no flow);
//   - every free 4-neighbour of v was in the previous window (else the source may have been unseen);
//   - exactly one of them, u, was occupied by another agent then (the agent's own previous cell does not count).
// Everything else (several candidate sources, unseen neighbours) is skipped as ambiguous.
void Follower::add_observed_flow(size_t agent_idx)
{
    const int r = cfg.obs_radius;
    const int H = grid.size(), W = grid.front().size();
    const auto cur = agents[agent_idx].cur_position, prev = prev_positions[agent_idx];
    auto seen_prev = [&](int i, int j) { return std::abs(i - prev.first) <= r && std::abs(j - prev.second) <= r; };
    auto other_prev = [&](int i, int j) {
        return seen_prev(i, j) && std::make_pair(i, j) != prev && prev_agents_pos[i][j];
    };
    static const int DIRS[4][2] = {{0, 1}, {1, 0}, {-1, 0}, {0, -1}};  // planner.h d4: east, south, north, west
    for(int i = cur.first - r; i <= cur.first + r; i++)
        for(int j = cur.second - r; j <= cur.second + r; j++)
        {
            if(i < 0 || j < 0 || i >= H || j >= W || !agents_pos[i][j] || std::make_pair(i, j) == cur)
                continue;
            if(!seen_prev(i, j) || other_prev(i, j))
                continue;
            int num_sources = 0, src_i = -1, src_j = -1;
            bool unseen = false;
            for(const auto& d : DIRS)
            {
                int ui = i - d[0], uj = j - d[1];
                if(ui < 0 || uj < 0 || ui >= H || uj >= W || grid[ui][uj])
                    continue;
                if(!seen_prev(ui, uj)) { unseen = true; break; }
                if(other_prev(ui, uj)) { num_sources++; src_i = ui; src_j = uj; }
            }
            if(!unseen && num_sources == 1)
                planners[agent_idx].add_observed_move({src_i, src_j}, {i, j});
        }
}

// With both the density gate and dflow on, the gate is set first (it only reads the occupied cells), then the
// occupations and the flow are updated, then the path is planned with both.
std::vector<float> Follower::observe(size_t agent_idx, int radius)
{
    std::list<std::pair<int, int>> occupied_cells = get_occupied_cells(agent_idx);
    if(cfg.guidance_gate)
        planners[agent_idx].set_gate(update_gate(agent_idx, occupied_cells.size()));
    planners[agent_idx].update_occupied_cells(occupied_cells, agents[agent_idx].goal);
    if(dflow_on && has_prev_state)
        add_observed_flow(agent_idx);
    planners[agent_idx].update_path(agents[agent_idx].cur_position, agents[agent_idx].goal);
    return generate_input(agent_idx, radius, planners[agent_idx].get_path());
}

int Follower::get_action(size_t agent_idx, std::pair<int, int> cur_position, std::pair<int, int> cur_goal)
{
    auto input = observe(agent_idx, actor.obs_radius);
    auto result = actor.get_output({input, {0}});
    std::vector<int> score;
    score.reserve(result.first.size());
    for(auto v: result.first)
        score.push_back(static_cast<int>(v * 1e6));
    std::discrete_distribution<int> distr(score.begin(), score.end());
    int action = distr(generators[agent_idx]);
    last_actions[agent_idx] = action;
    return action;
}

// The actions taken since the last set_state, for the memory channels (act() records its own; tests call this).
void Follower::set_actions(std::vector<int> actions)
{
    last_actions = actions;
}

void Follower::set_state(std::vector<std::pair<int, int>> cur_positions, std::vector<std::pair<int, int>> goals)
{
    if(cfg.neighbour_memory != "none" && has_state)  // the state of the previous step becomes history
    {
        std::vector<std::pair<int, int>> positions;
        for(const auto& agent : agents)
            positions.push_back(agent.cur_position);
        position_history.push_front(positions);
        agents_pos_history.push_front(agents_pos);
        if(position_history.size() > MEMORY_STEPS)
        {
            position_history.pop_back();
            agents_pos_history.pop_back();
        }
    }
    if(dflow_on && has_state)  // the view of the previous step, for add_observed_flow
    {
        prev_agents_pos = agents_pos;
        prev_positions.resize(agents.size());
        for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
            prev_positions[agent_idx] = agents[agent_idx].cur_position;
        has_prev_state = true;
    }
    has_state = true;
    // Cleared and set one agent at a time, in index order, like pogema's soft collision system: an agent that moved
    // into the cell a later agent just left is erased for one step, and training saw the same agents grid. With
    // FIX_SOFT_POSITIONS all old cells are cleared first instead.
    if(FIX_SOFT_POSITIONS)
        for(const auto& agent : agents)
            agents_pos[agent.cur_position.first][agent.cur_position.second] = 0;
    for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
    {
        auto prev_position = agents[agent_idx].cur_position;
        if(!FIX_SOFT_POSITIONS)
            agents_pos[agents[agent_idx].cur_position.first][agents[agent_idx].cur_position.second] = 0;
        agents[agent_idx].cur_position = {cur_positions[agent_idx].first + agents[agent_idx].offset.first,
                                          cur_positions[agent_idx].second + agents[agent_idx].offset.second};
        agents_pos[agents[agent_idx].cur_position.first][agents[agent_idx].cur_position.second] = 1;
        if(cfg.use_memory && last_actions[agent_idx] >= 0)  // no step was taken before the first action
        {
            stuck[agent_idx] = agents[agent_idx].cur_position == prev_position ? stuck[agent_idx] + 1 : 0;
            trails[agent_idx].push_front(prev_position);
            if(trails[agent_idx].size() > MEMORY_STEPS)
                trails[agent_idx].pop_back();
        }
        agents[agent_idx].goal = {goals[agent_idx].first + agents[agent_idx].offset.first,
                                  goals[agent_idx].second + agents[agent_idx].offset.second};
    }
}

std::vector<int> Follower::act(std::vector<std::pair<int, int>> cur_positions, std::vector<std::pair<int, int>> goals)
{
    std::vector<int> actions(agents.size(), 0);
    set_state(cur_positions, goals);
    if(cfg.num_threads == 1)
    {
        for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
            actions[agent_idx] = get_action(agent_idx, cur_positions[agent_idx], goals[agent_idx]);
    }
    else
    {
        BS::multi_future<int> future(agents.size());
        for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
            future[agent_idx] = pool.submit(&Follower::get_action, this, agent_idx, cur_positions[agent_idx], goals[agent_idx]);
        actions = future.get();
    }
    return actions;
}

void Follower::init(const Config& cfg_, py::array_t<double> array, std::vector<std::pair<int, int>> abs_starts)
{
    cfg = cfg_;
    pool.reset(cfg.num_threads);
    if(!cfg.path_to_weights.empty())  // empty: observation-only use (tests), act() is unavailable
        actor = NN_module(cfg.path_to_weights);

    py::buffer_info buf = array.request();
    double *ptr = (double *) buf.ptr;
    grid = std::vector<std::vector<int>>(buf.shape[0], std::vector<int>(buf.shape[1], 0));
    agents_pos = grid;
    for(size_t i = 0; i < static_cast<size_t>(buf.shape[0]); i++)
        for(size_t j = 0; j < static_cast<size_t>(buf.shape[1]); j++)
            grid[i][j] = ptr[i*buf.shape[1] + j];

    agents = std::vector<Agent>(abs_starts.size());
    generators = std::vector<std::mt19937>(agents.size());
    trails = std::vector<std::deque<std::pair<int, int>>>(agents.size());
    last_actions = std::vector<int>(agents.size(), -1);
    stuck = std::vector<int>(agents.size(), 0);
    agents_pos_history.clear();
    position_history.clear();
    has_state = false;
    has_prev_state = false;
    prev_agents_pos.clear();
    prev_positions.clear();
    int dflow_mode;
    if(cfg.dflow == "off")
        dflow_mode = 0;
    else if(cfg.dflow == "add")
        dflow_mode = 1;
    else if(cfg.dflow == "split")
        dflow_mode = 2;
    else if(cfg.dflow == "ggboost")
        dflow_mode = 3;
    else
        throw std::invalid_argument("dflow: '" + cfg.dflow + "' is not one of off, add, split, ggboost");
    dflow_on = dflow_mode != 0;
    // the input channels in network order (ConcatPositionalFeatures), for cfg.ablate_channels
    std::vector<std::string> channels = {"obstacles", "agents"};
    if(cfg.use_memory)
        channels.insert(channels.end(), {"blocked", "trail"});
    if(cfg.neighbour_memory == "trail")
        channels.insert(channels.end(), {"nb_trail", "nb_wait"});
    ablated_channels.clear();
    std::stringstream names(cfg.ablate_channels);
    for(std::string name; std::getline(names, name, ',');)
    {
        auto it = std::find(channels.begin(), channels.end(), name);
        if(it == channels.end())
            throw std::invalid_argument("ablate_channels: '" + name + "' is not an input channel of this configuration");
        ablated_channels.push_back(static_cast<int>(it - channels.begin()));
    }
    planners.clear();
    planner temp_planner(grid, cfg.use_static_cost, cfg.use_dynamic_cost, cfg.reset_dynamic_cost);
    auto pen_matrix = temp_planner.precompute_penalty_matrix(cfg.obs_radius);
    // one guidance table shared by all planners (validated once, by the first planner)
    std::shared_ptr<const std::vector<float>> guidance, guidance_log;
    if(!cfg.guidance_table.empty())
        guidance = std::make_shared<const std::vector<float>>(cfg.guidance_table);
    rho_ema.assign(agents.size(), -1.f);
    gates.assign(agents.size(), 1.f);
    if(cfg.guidance_gate)
    {
        if(!guidance)
            throw std::invalid_argument("guidance_gate needs a guidance_table");
        if(!(cfg.gate_rho1 > cfg.gate_rho0))
            throw std::invalid_argument("guidance_gate needs gate_rho1 > gate_rho0");
        guidance_log = planner::log_table(guidance);
        gate_alpha = cfg.gate_halflife > 0.f ? 1.f - std::pow(2.f, -1.f / cfg.gate_halflife) : 1.f;
    }
    for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
    {
        agents[agent_idx].offset = abs_starts[agent_idx];
        agents_pos[abs_starts[agent_idx].first][abs_starts[agent_idx].second] = 1;
        planners.emplace_back(grid, cfg.use_static_cost, cfg.use_dynamic_cost, cfg.reset_dynamic_cost);
        planners.back().set_penalties(pen_matrix);
        planners.back().set_crisscross(cfg.crisscross_ratio, cfg.crisscross_all);
        planners.back().set_guidance_shared(guidance);
        planners.back().set_guidance_gate_shared(guidance_log);
        if(dflow_on)
            planners.back().set_dflow(dflow_mode, cfg.dflow_with, cfg.dflow_opp);
        generators[agent_idx].seed(cfg.seed);
    }
}

PYBIND11_MODULE(follower, m) {
py::class_<Follower>(m, "Follower")
.def(py::init<>())
.def("act", &Follower::act)
.def("init", &Follower::init)
.def("set_state", &Follower::set_state)
.def("set_actions", &Follower::set_actions)
.def("observe", &Follower::observe)
.def("get_gates", &Follower::get_gates)
.def("get_flow", &Follower::get_flow)
.def("get_occupations", &Follower::get_occupations)
;
}

<%
cfg['libraries'] = ['onnxruntime']
cfg['extra_compile_args'] = ['-std=c++17']
cfg['dependencies'] = ['follower.h', 'planner.h', 'NN_module.h', 'config.h', 'agent.h']
setup_pybind11(cfg)
%>