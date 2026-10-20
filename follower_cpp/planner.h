#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cmath>
#include <list>
#include <cstdint>
#include <algorithm>
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
    int d4[4];

    inline int id(int i, int j) const { return i * W + j; }
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
        push(heap_, {start, 0.f, 0.f + h_values[start]});
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
                if(use_dynamic_cost)
                    cost += num_occupations[n];
                float ng = cur.g + cost;
                if(g_of(n) > ng)
                {
                    stamp_[n] = search_id; g_[n] = ng; parent_[n] = cur.idx;
                    push(heap_, {n, ng, ng + h_values[n]});
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
        if(reset_dynamic_cost and goal != cur_goal)
            std::fill(num_occupations.begin(), num_occupations.end(), 0.f);
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

    void update_occupations(py::array_t<double, py::array::c_style | py::array::forcecast> array, std::pair<int, int> cur_pos, std::pair<int, int> cur_goal)
    {
        reset_occupations_if_new_goal(id(cur_goal.first + abs_offset.first, cur_goal.second + abs_offset.second));
        auto a = array.unchecked<2>();
        int pi = cur_pos.first + abs_offset.first, pj = cur_pos.second + abs_offset.second;
        for(py::ssize_t i = 0; i < a.shape(0); i++)
            for(py::ssize_t j = 0; j < a.shape(1); j++)
                if(a(i, j) != 0)
                    num_occupations[id(pi + i, pj + j)] += 1.0f;
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
