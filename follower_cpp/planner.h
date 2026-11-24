#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <queue>
#include <cmath>
#include <list>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>
#define INF 1000000000
namespace py = pybind11;

// Same ordering semantics as the original Node (f, then g, with 1e-5 epsilon),
// but without the parent field so heap entries are smaller.
struct QNode {
    int idx;
    float g;
    float f;
    bool operator>(const QNode& o) const
    {
        return f > o.f or (std::abs(f - o.f) < 1e-5 and g > o.g);
    }
};

class planner
{
    int H, W;
    int start, goal;
    std::pair<int, int> abs_offset;
    std::vector<uint8_t> free_;            // 1 = traversable
    std::vector<float> num_occupations;
    std::vector<float> penalties;
    std::vector<float> h_values;
    std::vector<float> g_;
    std::vector<int> parent_;
    std::vector<uint32_t> stamp_;          // lazy reset: g_/parent_ valid only if stamp_ == search_id
    uint32_t search_id = 0;
    std::vector<QNode> heap_;              // reused storage for the open list
    bool use_static_cost;
    bool use_dynamic_cost;
    bool reset_dynamic_cost;
    float crisscross_ratio = 1.f;          // cost factor of moves against the crisscross highways (1: off)
    bool crisscross_all = false;           // the factor also scales the dynamic cost, not only the static one
    // Guidance table (opt-in): H*W*4 cost factors of the move from cell n in direction d4[k] at [n * 4 + k], shared by
    // all planners of a map. When set it replaces the crisscross factor (applied where that one is, with the same
    // crisscross_all semantics). Factors below 1 would make the heuristic inadmissible, so h is scaled by h_scale =
    // min(1, smallest factor) then (1 for factors >= 1, which keeps the original expansion order).
    std::shared_ptr<const std::vector<float>> guidance;
    float h_scale = 1.f;
    // Density gate (opt-in, set_guidance_gate): the move cost factor is blended in log space between the crisscross
    // factor (as without a table) and the table's, log f = log f_cc + gate * (log f_table - log f_cc). gate <= 0 runs
    // exactly the no-table search (crisscross), gate >= 1 exactly the table search; in between h is scaled by
    // h_scale^gate (a lower bound of the blended factors, as the crisscross factors are >= 1).
    std::shared_ptr<const std::vector<float>> guidance_log;  // log of the table's factors (null: gate off)
    float gate = 1.f;
    // Directional flow (opt-in, eval-only, see set_dflow): flow[n * 4 + k] counts the observed moves of other agents in
    // direction d4[k] that left or entered cell n. It lives exactly as long as num_occupations (cleared with it on a
    // new goal). The opposite of direction k is 3 - k. Empty while off. Independent of the density gate: the gate
    // changes the move cost factor, dflow the dynamic cost (ggboost: the factor of a penalised step, gated or not).
    enum DflowMode { DFLOW_OFF = 0, DFLOW_ADD = 1, DFLOW_SPLIT = 2, DFLOW_GGBOOST = 3 };
    int dflow_mode = DFLOW_OFF;
    float dflow_with = 1.f;  // DFLOW_SPLIT: weight of observations moving the same way as the planned step
    float dflow_opp = 1.f;   // DFLOW_SPLIT: weight of observations moving against it
    std::vector<float> flow;
    int d4[4];

    inline int id(int i, int j) const { return i * W + j; }

    // Crisscross highways: horizontal moves are encouraged eastwards on even rows and westwards on odd rows, vertical
    // moves southwards on even columns and northwards on odd columns (absolute grid coordinates). k indexes d4.
    inline bool against_crisscross(int from, int k) const
    {
        int i = from / W, j = from % W;
        switch(k) {
            case 0: return i % 2 == 1;   // east
            case 1: return j % 2 == 1;   // south
            case 2: return j % 2 == 0;   // north
            default: return i % 2 == 0;  // west
        }
    }
    inline float g_of(int n) const { return stamp_[n] == search_id ? g_[n] : (float)INF; }

    void push(std::vector<QNode>& h, QNode n) { h.push_back(n); std::push_heap(h.begin(), h.end(), std::greater<QNode>()); }
    QNode pop(std::vector<QNode>& h) { std::pop_heap(h.begin(), h.end(), std::greater<QNode>()); QNode n = h.back(); h.pop_back(); return n; }

    void compute_shortest_path()
    {
        ++search_id;
        heap_.clear();
        // NOTE: like the original, the start cell's g is left at INF, so it can be re-reached from a neighbour
        // and re-expanded. Kept on purpose: it changes heap tie-breaking, and we want bit-identical paths.
        stamp_[start] = search_id; g_[start] = (float)INF; parent_[start] = -1;
        const float* table = guidance ? guidance->data() : nullptr;
        const float* ltable = nullptr;  // set: blend crisscross and table factors with weight gate
        float hs = h_scale;
        const float log_ratio = std::log(crisscross_ratio);
        if(table and guidance_log) {
            if(gate <= 0.f) {
                table = nullptr;
                hs = 1.f;
            } else if(gate < 1.f) {
                ltable = guidance_log->data();
                hs = std::pow(h_scale, gate);
            }
        }
        push(heap_, {start, 0.f, 0.f + hs * h_values[start]});
        while(!heap_.empty())
        {
            QNode cur = pop(heap_);
            if(g_[cur.idx] < cur.g)
                continue;
            if(cur.idx == goal)
                break;  // expanding the goal cannot change any node on the goal's parent chain
            for(int k = 0; k < 4; k++) {
                int n = cur.idx + d4[k];
                if(!free_[n])
                    continue;
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n];
                // Crisscross only raises costs, so the heuristic (which ignores the highways) is no less admissible
                // than before. A guidance table replaces the crisscross factor (a factor of exactly 1 changes nothing,
                // so the crisscross pattern as a table gives the same costs); h_scale keeps h admissible.
                bool against;
                float factor;
                if(ltable) {
                    against = true;
                    float lcc = (crisscross_ratio != 1.f and against_crisscross(cur.idx, k)) ? log_ratio : 0.f;
                    factor = std::exp(lcc + gate * (ltable[cur.idx * 4 + k] - lcc));
                } else if(table) {
                    against = true;
                    factor = table[cur.idx * 4 + k];
                } else {
                    against = crisscross_ratio != 1.f and against_crisscross(cur.idx, k);
                    factor = crisscross_ratio;
                }
                // dflow ggboost: the contraflow penalty (factor - 1) of a penalised step is multiplied by
                // 1 + (observed flow at n against the step); factors <= 1 stay, so it only ever strengthens
                if(dflow_mode == DFLOW_GGBOOST and against and factor > 1.f)
                    factor = 1.f + (factor - 1.f) * (1.f + flow[n * 4 + 3 - k]);
                if(against and not crisscross_all)
                    cost *= factor;
                if(use_dynamic_cost) {
                    float dyn = num_occupations[n];
                    if(dflow_mode == DFLOW_ADD)
                        dyn += flow[n * 4 + 3 - k];  // each opposing move costs as much as one more occupation
                    else if(dflow_mode == DFLOW_SPLIT)
                        dyn = split_dynamic_cost(dyn, n, k);
                    cost += dyn;
                }
                if(against and crisscross_all)
                    cost *= factor;
                float ng = cur.g + cost;
                if(g_of(n) > ng)
                {
                    stamp_[n] = search_id; g_[n] = ng; parent_[n] = cur.idx;
                    push(heap_, {n, ng, ng + hs * h_values[n]});
                }
            }
        }
    }

    void update_h_values(int g)
    {
        std::fill(h_values.begin(), h_values.end(), (float)INF);
        std::vector<QNode> open;
        open.reserve(H * W / 4);
        h_values[g] = 0;
        push(open, {g, 0.f, 0.f});
        while(!open.empty())
        {
            QNode cur = pop(open);
            if(h_values[cur.idx] < cur.g)
                continue;
            for(int k = 0; k < 4; k++) {
                int n = cur.idx + d4[k];
                if(!free_[n])
                    continue;
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n];
                if(h_values[n] > cur.g + cost)
                {
                    h_values[n] = cur.g + cost;
                    push(open, {n, cur.g + cost, cur.g + cost});
                }
            }
        }
    }

    std::vector<std::vector<float>> to_2d(const std::vector<float>& v) const
    {
        std::vector<std::vector<float>> r(H, std::vector<float>(W));
        for(int i = 0; i < H; i++)
            for(int j = 0; j < W; j++)
                r[i][j] = v[id(i, j)];
        return r;
    }

    void reset_occupations_if_new_goal(int cur_goal)
    {
        if(reset_dynamic_cost and goal != cur_goal) {
            std::fill(num_occupations.begin(), num_occupations.end(), 0.f);
            std::fill(flow.begin(), flow.end(), 0.f);  // same lifetime as num_occupations (empty while dflow is off)
        }
    }

    // DFLOW_SPLIT: the occupation count occ of cell n made direction-aware for a step in direction k. Observations that
    // moved with the step weigh dflow_with, against it dflow_opp, perpendicular or stationary 1. A move credits both its
    // cells and one observation of a passing agent can be credited twice (entering and leaving), so the directional
    // counts are scaled to at most occ in total: occ' = occ + s * ((with - 1) * f_with + (opp - 1) * f_opp) with
    // s = min(1, occ / sum of flow at n). With with, opp >= 0 this keeps occ' >= 0.
    inline float split_dynamic_cost(float occ, int n, int k) const
    {
        const float* f = flow.data() + n * 4;
        float total = f[0] + f[1] + f[2] + f[3];
        if(total <= 0.f)
            return occ;
        float s = occ >= total ? 1.f : occ / total;
        return std::max(0.f, occ + s * ((dflow_with - 1.f) * f[k] + (dflow_opp - 1.f) * f[3 - k]));
    }

public:
    planner(std::vector<std::vector<int>> _grid={}, bool _use_static_cost=true, bool _use_dynamic_cost=true, bool _reset_dynamic_cost=true):
    use_static_cost(_use_static_cost), use_dynamic_cost(_use_dynamic_cost), reset_dynamic_cost(_reset_dynamic_cost)
    {
        H = _grid.size();
        W = _grid.front().size();
        d4[0] = 1; d4[1] = W; d4[2] = -W; d4[3] = -1;   // same neighbor order as the original
        free_.assign(H * W, 0);
        for(int i = 0; i < H; i++)
            for(int j = 0; j < W; j++)
                free_[id(i, j)] = _grid[i][j] == 0;
        abs_offset = {0, 0};
        goal = 0;
        start = 0;
        num_occupations.assign(H * W, 0.f);
        penalties.assign(H * W, 1.f);
        h_values.assign(H * W, (float)INF);
        g_.assign(H * W, (float)INF);
        parent_.assign(H * W, -1);
        stamp_.assign(H * W, 0);
    }

    std::vector<std::vector<float>> get_num_occupied_matrix()
    {
        return to_2d(num_occupations);
    }

    std::vector<std::vector<float>> precompute_penalty_matrix(int obs_radius)
    {
        std::vector<float> pen(H * W, 0.f);
        std::vector<int> dist(H * W, -1), fringe(H * W);
        float max_avg_dist(0);
        for(int i = obs_radius; i < H - obs_radius; i++)
            for(int j = obs_radius; j < W - obs_radius; j++)
            {
                int s = id(i, j);
                if(!free_[s])
                    continue;
                dist[s] = 0;
                int head = 0, tail = 0;
                fringe[tail++] = s;
                long long sum = 0;
                while(head < tail)
                {
                    int p = fringe[head++];
                    sum += dist[p];
                    for(int k = 0; k < 4; k++) {
                        int n = p + d4[k];
                        if(free_[n] && dist[n] < 0) {
                            dist[n] = dist[p] + 1;
                            fringe[tail++] = n;
                        }
                    }
                }
                for(int q = 0; q < tail; q++)   // reset only the cells we touched
                    dist[fringe[q]] = -1;
                // Integer sum is exact; equals the original float accumulation while the sum < 2^24.
                pen[s] = (float)sum / (float)(tail - 1);
                max_avg_dist = std::fmax(max_avg_dist, pen[s]);
            }
        for(int i = obs_radius; i < H - obs_radius; i++)
            for(int j = obs_radius; j < W - obs_radius; j++)
                if(free_[id(i, j)])
                    pen[id(i, j)] = max_avg_dist / pen[id(i, j)];
        penalties = pen;
        return to_2d(pen);
    }

    // Soft crisscross highways (as SILLM's Static Guidance): moves against them cost `ratio` times as much.
    void set_crisscross(float ratio, bool all)
    {
        if(!(ratio >= 1.f))
            throw std::invalid_argument("crisscross ratio must be >= 1");
        crisscross_ratio = ratio;
        crisscross_all = all;
    }

    // Guidance table (see `guidance`): flat H*W*4 factors, [n * 4 + k] for the move from cell n in direction d4[k]
    // (east, south, north, west in absolute grid coordinates); all finite and > 0. Empty: off (crisscross as set).
    void set_guidance_shared(std::shared_ptr<const std::vector<float>> table)
    {
        if(table and table->empty())
            table = nullptr;
        if(table) {
            if(table->size() != (size_t)H * W * 4)
                throw std::invalid_argument("guidance table must have H * W * 4 entries");
            float lo = 1.f;
            for(float v : *table) {
                if(!(v > 0.f) or !std::isfinite(v))
                    throw std::invalid_argument("guidance factors must be finite and > 0");
                lo = std::min(lo, v);
            }
            h_scale = lo;
        } else
            h_scale = 1.f;
        guidance = table;
        guidance_log = nullptr;  // a new table: the gate (if wanted) is set again after it
    }

    void set_guidance(std::vector<float> table)
    {
        set_guidance_shared(std::make_shared<const std::vector<float>>(std::move(table)));
    }

    // log of the factors of a guidance table, shared by all planners of a map (null table: null)
    static std::shared_ptr<const std::vector<float>> log_table(const std::shared_ptr<const std::vector<float>>& table)
    {
        if(!table)
            return nullptr;
        auto l = std::make_shared<std::vector<float>>(table->size());
        for(size_t e = 0; e < table->size(); e++)
            (*l)[e] = std::log((*table)[e]);
        return l;
    }

    // Density gate (see `gate`): on with the log table of the guidance table set, off with null (plain table)
    void set_guidance_gate_shared(std::shared_ptr<const std::vector<float>> log_tab)
    {
        if(log_tab and (!guidance or log_tab->size() != guidance->size()))
            throw std::invalid_argument("the gate needs the log table of the guidance table set");
        guidance_log = log_tab;
    }

    // python / tests: gate the current guidance table (on) or not (off)
    void set_guidance_gate(bool on)
    {
        if(on and !guidance)
            throw std::invalid_argument("the gate needs a guidance table");
        set_guidance_gate_shared(on ? log_table(guidance) : nullptr);
    }

    void set_gate(float g)
    {
        if(!std::isfinite(g))
            throw std::invalid_argument("gate must be finite");
        gate = std::min(1.f, std::max(0.f, g));
    }

    float get_gate() const { return gate; }

    // the guidance table set (null: none), for multi_planner::set_guidance_gate
    std::shared_ptr<const std::vector<float>> guidance_table() const { return guidance; }

    // Direction-aware dynamic cost from the observed flow of other agents (eval-only, opt-in; mode 0 = off keeps the
    // planner bit-identical). mode 1 add: a step into n in direction k costs flow[n][opposite k] more dynamic cost
    // (one occupation each); 2 split: the occupation count weighted by direction, see split_dynamic_cost (w_with,
    // w_opp >= 0); 3 ggboost: the guidance / crisscross penalty factor - 1 of the step is multiplied by
    // 1 + flow[n][opposite k]. Resets the flow.
    void set_dflow(int mode, float w_with, float w_opp)
    {
        if(mode < DFLOW_OFF or mode > DFLOW_GGBOOST)
            throw std::invalid_argument("dflow mode must be 0 (off), 1 (add), 2 (split) or 3 (ggboost)");
        if(!(w_with >= 0.f) or !(w_opp >= 0.f) or !std::isfinite(w_with) or !std::isfinite(w_opp))
            throw std::invalid_argument("dflow weights must be finite and >= 0");
        dflow_mode = mode;
        dflow_with = w_with;
        dflow_opp = w_opp;
        if(mode == DFLOW_OFF)
            flow.clear();
        else
            flow.assign((size_t)H * W * 4, 0.f);
    }

    bool dflow_on() const { return dflow_mode != DFLOW_OFF; }

    // One observed move of another agent from cell u to the adjacent cell v (coordinates relative to abs_offset, as
    // everywhere else), credited to both cells in its direction. No-op while dflow is off.
    void add_observed_move(std::pair<int, int> u, std::pair<int, int> v)
    {
        if(flow.empty())
            return;
        int a = id(u.first + abs_offset.first, u.second + abs_offset.second);
        int b = id(v.first + abs_offset.first, v.second + abs_offset.second);
        int k = 0;
        while(k < 4 and a + d4[k] != b)
            k++;
        if(k == 4 or std::abs(u.first - v.first) + std::abs(u.second - v.second) != 1)
            throw std::invalid_argument("add_observed_move: u and v must be adjacent");
        flow[(size_t)a * 4 + k] += 1.f;
        flow[(size_t)b * 4 + k] += 1.f;
    }

    // The flow as H x W x 4 (d4 order: east, south, north, west); all zeros while off.
    std::vector<std::vector<std::vector<float>>> get_flow_matrix() const
    {
        std::vector<std::vector<std::vector<float>>> r(H, std::vector<std::vector<float>>(W, std::vector<float>(4, 0.f)));
        if(!flow.empty())
            for(int i = 0; i < H; i++)
                for(int j = 0; j < W; j++)
                    for(int k = 0; k < 4; k++)
                        r[i][j][k] = flow[(size_t)id(i, j) * 4 + k];
        return r;
    }

    void set_penalties(std::vector<std::vector<float>> _penalties)
    {
        for(int i = 0; i < H; i++)
            for(int j = 0; j < W; j++)
                penalties[id(i, j)] = _penalties[i][j];
    }

    void update_occupied_cells(const std::list<std::pair<int, int>>& _occupied_cells, std::pair<int, int> cur_goal)
    {
        reset_occupations_if_new_goal(id(cur_goal.first, cur_goal.second));
        for(auto o:_occupied_cells)
            num_occupations[id(o.first, o.second)] += 1.0f;
    }

    // Row-major h x w window whose top-left corner is cur_pos (relative coords). With zero_center the middle cell is
    // ignored, which is what the python code did by temporarily zeroing the agent's own cell.
    template<typename T>
    void add_occupations(const T* a, py::ssize_t h, py::ssize_t w, std::pair<int, int> cur_pos, std::pair<int, int> cur_goal, bool zero_center)
    {
        reset_occupations_if_new_goal(id(cur_goal.first + abs_offset.first, cur_goal.second + abs_offset.second));
        int pi = cur_pos.first + abs_offset.first, pj = cur_pos.second + abs_offset.second;
        for(py::ssize_t i = 0; i < h; i++)
            for(py::ssize_t j = 0; j < w; j++)
                if(a[i * w + j] != 0 and not (zero_center and i == h / 2 and j == w / 2))
                    num_occupations[id(pi + i, pj + j)] += 1.0f;
    }

    void update_occupations(py::array_t<double, py::array::c_style | py::array::forcecast> array, std::pair<int, int> cur_pos, std::pair<int, int> cur_goal)
    {
        if(array.ndim() != 2)
            throw std::invalid_argument("occupations must be a 2D array");
        add_occupations(array.data(), array.shape(0), array.shape(1), cur_pos, cur_goal, false);
    }

    void set_penalties_flat(const float* p)
    {
        penalties.assign(p, p + (size_t)H * W);
    }

    // Node ids on the current path, start first and goal last; empty if there is no path (same cases as get_path).
    void path_ids(std::vector<int>& out) const
    {
        out.clear();
        if(g_of(goal) < INF and goal != start)
        {
            for(int n = goal; n != start; n = parent_[n])
                out.push_back(n);
            out.push_back(start);
            std::reverse(out.begin(), out.end());
        }
    }

    std::pair<int, int> to_rel(int n) const
    {
        return {n / W - abs_offset.first, n % W - abs_offset.second};
    }

    void update_path(std::pair<int, int> s, std::pair<int, int> g)
    {
        start = id(s.first + abs_offset.first, s.second + abs_offset.second);
        int ng = id(g.first + abs_offset.first, g.second + abs_offset.second);
        if(goal != ng)
            update_h_values(ng);
        goal = ng;
        compute_shortest_path();
    }

    std::list<std::pair<int, int>> get_path()
    {
        std::list<std::pair<int, int>> path;
        if(g_of(goal) < INF and goal != start)
        {
            int n = goal;
            for(; n != start; n = parent_[n])
                path.push_front({n / W - abs_offset.first, n % W - abs_offset.second});
            path.push_front({n / W - abs_offset.first, n % W - abs_offset.second});
        }
        return path;
    }

    std::pair<std::pair<int, int>, std::pair<int, int>> get_next_node()
    {
        if(g_of(goal) >= INF or goal == start)
            return {{INF, INF}, {INF, INF}};
        int n = goal;
        while(parent_[n] != start)
            n = parent_[n];
        return {{start / W - abs_offset.first, start % W - abs_offset.second},
                {n / W - abs_offset.first, n % W - abs_offset.second}};
    }

    void set_abs_start(std::pair<int, int> offset)
    {
        abs_offset = offset;
    }
};

// One planner per agent, driven for all agents with a single call per environment step. Does what
// FollowerWrapper.observation used to do in python for every agent: update occupations, re-plan, draw the path into
// the obstacle channel, and compute the intrinsic reward (+1 if the agent reached the sub-goal chosen last step).
class multi_planner
{
    std::vector<planner> ps;
    std::vector<std::pair<int, int>> prev_goals;
    bool has_prev = false;
    double intrinsic_target_reward;
    std::vector<int> path_buf;
    int H, W;
    // Eval-only planner options of the C++ Follower (follower.cpp), ported for the python path (Follower / training
    // wrappers); all off by default, which keeps step() bit-identical. Agent k's absolute cell is its relative xy +
    // offsets[k], as in planner::abs_offset.
    std::vector<std::pair<int, int>> offsets;
    std::vector<uint8_t> free_;            // 1 = traversable, absolute coordinates (the grid of the constructor)
    // density gate (set_guidance_gate), as Follower::update_gate
    bool gate_on = false;
    float gate_rho0 = 0.f, gate_rho1 = 0.1f, gate_alpha = 1.f;
    std::vector<float> rho_ema, gates;
    // directional flow (set_dflow), as Follower::add_observed_flow: every agent's agents window and absolute cell of
    // the previous step
    bool dflow_on = false;
    bool has_prev_view = false;
    std::vector<std::vector<float>> prev_views;
    std::vector<std::pair<int, int>> prev_cells;

    // Follower::update_gate from the agent's own windows: rho = other agents in the agents window / free cells of the
    // obstacles window (raw pogema obstacles, 1 = obstacle), the own (centre) cell excluded from both.
    float update_gate(size_t k, const float* a, const float* o, int R)
    {
        const int c = (R / 2) * R + R / 2;
        int num_others = 0, free_cells = 0;
        for(int e = 0; e < R * R; e++)
        {
            if(e == c)
                continue;
            if(a[e] != 0)
                num_others++;
            if(o[e] == 0)
                free_cells++;
        }
        float rho = free_cells > 0 ? static_cast<float>(num_others) / free_cells : 0.f;
        float& ema = rho_ema[k];
        ema = ema < 0.f ? rho : ema + gate_alpha * (rho - ema);
        float g = (ema - gate_rho0) / (gate_rho1 - gate_rho0);
        gates[k] = std::min(1.f, std::max(0.f, g));
        return gates[k];
    }

    // Follower::add_observed_flow from the agent's current and previous agents windows (same rules, see follower.cpp):
    // a move u -> v is credited when v (not the own cell) is occupied now, was seen but not occupied by another agent
    // on the previous step, all free 4-neighbours of v were seen then, and exactly one of them, u, was occupied by
    // another agent then. cur: the agent's absolute cell.
    void add_observed_flow(size_t k, const float* a, int R, std::pair<int, int> cur)
    {
        const int r = R / 2;
        const auto prev = prev_cells[k];
        const float* pa = prev_views[k].data();
        auto seen_prev = [&](int i, int j) { return std::abs(i - prev.first) <= r && std::abs(j - prev.second) <= r; };
        auto other_prev = [&](int i, int j) {
            return seen_prev(i, j) && std::make_pair(i, j) != prev &&
                   pa[(i - prev.first + r) * R + (j - prev.second + r)] != 0;
        };
        static const int DIRS[4][2] = {{0, 1}, {1, 0}, {-1, 0}, {0, -1}};  // d4: east, south, north, west
        const auto off = offsets[k];
        for(int di = -r; di <= r; di++)
            for(int dj = -r; dj <= r; dj++)
            {
                int i = cur.first + di, j = cur.second + dj;
                if(i < 0 || j < 0 || i >= H || j >= W || a[(di + r) * R + dj + r] == 0 || (di == 0 && dj == 0))
                    continue;
                if(!seen_prev(i, j) || other_prev(i, j))
                    continue;
                int num_sources = 0, src_i = -1, src_j = -1;
                bool unseen = false;
                for(const auto& d : DIRS)
                {
                    int ui = i - d[0], uj = j - d[1];
                    if(ui < 0 || uj < 0 || ui >= H || uj >= W || !free_[(size_t)ui * W + uj])
                        continue;
                    if(!seen_prev(ui, uj)) { unseen = true; break; }
                    if(other_prev(ui, uj)) { num_sources++; src_i = ui; src_j = uj; }
                }
                if(!unseen && num_sources == 1)
                    ps[k].add_observed_move({src_i - off.first, src_j - off.second}, {i - off.first, j - off.second});
            }
    }

public:
    multi_planner(std::vector<std::vector<int>> grid, bool use_static_cost, bool use_dynamic_cost, bool reset_dynamic_cost,
                  std::vector<std::pair<int, int>> starts, double _intrinsic_target_reward):
    intrinsic_target_reward(_intrinsic_target_reward)
    {
        H = grid.size();
        W = grid.front().size();
        free_.assign((size_t)H * W, 0);
        for(int i = 0; i < H; i++)
            for(int j = 0; j < W; j++)
                free_[(size_t)i * W + j] = grid[i][j] == 0;
        offsets = starts;
        ps.reserve(starts.size());
        for(auto& s: starts)
        {
            ps.emplace_back(grid, use_static_cost, use_dynamic_cost, reset_dynamic_cost);
            ps.back().set_abs_start(s);
        }
        prev_goals.assign(starts.size(), {0, 0});
        rho_ema.assign(starts.size(), -1.f);
        gates.assign(starts.size(), 1.f);
    }

    // One guidance table (flat H * W * 4, see planner::set_guidance_shared) shared by all planners; empty: off.
    // Turns the gate off (set it again after the table).
    void set_guidance(std::vector<float> table)
    {
        std::shared_ptr<const std::vector<float>> t;
        if(!table.empty())
            t = std::make_shared<const std::vector<float>>(std::move(table));
        for(auto& p: ps)
            p.set_guidance_shared(t);
        gate_on = false;
    }

    // Density gate of the guidance table, as Config.guidance_gate / gate_rho0 / gate_rho1 / gate_halflife of the C++
    // Follower (halflife <= 0: the raw density).
    void set_guidance_gate(float rho0, float rho1, float halflife)
    {
        if(!(rho1 > rho0))
            throw std::invalid_argument("guidance gate needs rho1 > rho0");
        if(ps.empty())
            return;
        auto log_tab = planner::log_table(ps.front().guidance_table());
        if(!log_tab)
            throw std::invalid_argument("guidance gate needs a guidance table");
        for(auto& p: ps)
            p.set_guidance_gate_shared(log_tab);
        gate_on = true;
        gate_rho0 = rho0;
        gate_rho1 = rho1;
        gate_alpha = halflife > 0.f ? 1.f - std::pow(2.f, -1.f / halflife) : 1.f;
        rho_ema.assign(ps.size(), -1.f);
        gates.assign(ps.size(), 1.f);
    }

    // Directional flow (planner::set_dflow) for all planners; mode 0 = off.
    void set_dflow(int mode, float w_with, float w_opp)
    {
        for(auto& p: ps)
            p.set_dflow(mode, w_with, w_opp);
        dflow_on = mode != 0;
        has_prev_view = false;
        prev_views.clear();
        prev_cells.clear();
    }

    // tests: the gates of the last step (1 without gate), one planner's flow (H x W x 4) and occupation counts
    std::vector<float> get_gates() const { return gates; }
    std::vector<std::vector<std::vector<float>>> get_flow(size_t k) const { return ps.at(k).get_flow_matrix(); }
    std::vector<std::vector<float>> get_occupations(size_t k) { return ps.at(k).get_num_occupied_matrix(); }

    void set_penalties(py::array_t<float, py::array::c_style | py::array::forcecast> pen)
    {
        if(pen.ndim() != 2 or pen.shape(0) != H or pen.shape(1) != W)
            throw std::invalid_argument("penalties shape must match the grid");
        for(auto& p: ps)
            p.set_penalties_flat(pen.data());
    }

    void set_crisscross(float ratio, bool all)
    {
        for(auto& p: ps)
            p.set_crisscross(ratio, all);
    }

    // agents / obstacles: float32 (N, R, R), modified in place (agents: own cell set to 1 for re-planned agents;
    // obstacles: obstacles negated, path drawn as +1). xy / target: (N, 2). Returns the intrinsic rewards (N,).
    py::array_t<double> step(py::array_t<float, py::array::c_style> agents, py::array_t<float, py::array::c_style> obstacles,
                             py::array_t<long long, py::array::c_style | py::array::forcecast> xy,
                             py::array_t<long long, py::array::c_style | py::array::forcecast> target)
    {
        py::ssize_t n = ps.size();
        if(agents.ndim() != 3 or agents.shape(0) != n or agents.shape(1) != agents.shape(2) or obstacles.ndim() != 3 or
           obstacles.shape(0) != n or obstacles.shape(1) != agents.shape(1) or obstacles.shape(2) != agents.shape(2) or
           xy.ndim() != 2 or xy.shape(0) != n or xy.shape(1) != 2 or target.ndim() != 2 or target.shape(0) != n or target.shape(1) != 2)
            throw std::invalid_argument("expected agents/obstacles (N, R, R) and xy/target (N, 2), with N == number of agents");
        const int R = agents.shape(1), r = R / 2;
        float* A = agents.mutable_data();
        float* O = obstacles.mutable_data();
        const long long* XY = xy.data();
        const long long* T = target.data();
        py::array_t<double> rewards(n);
        double* rew = rewards.mutable_data();
        std::vector<std::pair<int, int>> new_goals(n);

        if(dflow_on and prev_views.size() != (size_t)n)
        {
            prev_views.assign(n, std::vector<float>(R * R, 0.f));
            prev_cells.assign(n, {0, 0});
        }
        for(py::ssize_t k = 0; k < n; k++)
        {
            planner& p = ps[k];
            float* a = A + k * R * R;
            float* o = O + k * R * R;
            std::pair<int, int> pos(XY[2 * k], XY[2 * k + 1]), goal(T[2 * k], T[2 * k + 1]);
            std::pair<int, int> cell(pos.first + offsets[k].first, pos.second + offsets[k].second);
            // as Follower::observe: the gate (every agent, every step), then occupations, flow, and the path
            if(gate_on)
                p.set_gate(update_gate(k, a, o, R));
            if(pos != goal)
            {
                p.add_occupations(a, R, R, {pos.first - r, pos.second - r}, goal, true);
                if(dflow_on and has_prev_view)
                    add_observed_flow(k, a, R, cell);
                a[r * R + r] = 1.0f;
                p.update_path(pos, goal);
            }
            if(dflow_on)  // this step's view for the next one (the own cell is never read)
            {
                std::copy(a, a + R * R, prev_views[k].begin());
                prev_cells[k] = cell;
            }
            p.path_ids(path_buf);
            if(path_buf.size() < 2)
                throw std::out_of_range("no path with a next node for agent " + std::to_string(k));
            rew[k] = (has_prev and prev_goals[k] == pos) ? intrinsic_target_reward : 0.0;
            new_goals[k] = p.to_rel(path_buf[1]);

            for(int i = 0; i < R * R; i++)
                if(o[i] > 0)
                    o[i] = -o[i];
            for(int node: path_buf)
            {
                auto g = p.to_rel(node);
                int dx = pos.first - g.first, dy = pos.second - g.second;
                if(dx > r or dx < -r or dy > r or dy < -r)
                    break;
                o[(r - dx) * R + (r - dy)] = 1.0f;
            }
        }
        prev_goals = std::move(new_goals);
        has_prev = true;
        has_prev_view = dflow_on;
        return rewards;
    }
};
