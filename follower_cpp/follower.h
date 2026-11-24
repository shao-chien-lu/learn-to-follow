#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <deque>
#include <random>
#include "NN_module.h"
#include "planner.h"
#include "config.h"
#include "agent.h"
#include "BS_thread_pool.hpp"

namespace py = pybind11;

class Follower {
    std::vector<Agent> agents;
    std::vector<std::vector<int>> agents_pos;
    std::vector<std::vector<int>> grid;
    NN_module actor;
    Config cfg;
    std::vector<std::mt19937> generators;
    std::vector<planner> planners;
    BS::thread_pool pool;
    // short-term memory of the own motion (cfg.use_memory), as MemoryWrapper in follower/preprocessing.py
    std::vector<std::deque<std::pair<int, int>>> trails;  // previous positions, most recent first
    std::vector<int> last_actions;  // -1 before the first action
    std::vector<int> stuck;  // steps without moving
    // memory of the other agents (cfg.neighbour_memory), as NeighbourMemoryWrapper: agents_pos and every agent's
    // position on the previous steps, most recent first
    std::deque<std::vector<std::vector<int>>> agents_pos_history;
    std::deque<std::vector<std::pair<int, int>>> position_history;
    bool has_state = false;  // set_state was called since init
    std::vector<int> ablated_channels;  // indices of the input channels fed as zeros (cfg.ablate_channels)
    // density gate (cfg.guidance_gate): every agent's smoothed local density (-1: no step yet) and current gate
    std::vector<float> rho_ema;
    std::vector<float> gates;
    float gate_alpha = 1.f;  // EMA weight of the new density (1: raw)
    float update_gate(size_t agent_idx, size_t num_others);
    // cfg.dflow != "off": agents_pos and every agent's position at the previous set_state, to infer the moves of the
    // other agents each agent saw (see add_observed_flow)
    bool dflow_on = false;
    bool has_prev_state = false;
    std::vector<std::vector<int>> prev_agents_pos;
    std::vector<std::pair<int, int>> prev_positions;
    void add_observed_flow(size_t agent_idx);

    std::list<std::pair<int, int>> get_occupied_cells(int agent_idx);
    bool seen_occupied(size_t age, int a_id, int i, int j) const;
    std::vector<float> generate_input(int a_id, int radius, std::list<std::pair<int, int>> path);
    int get_action(size_t agent_idx, std::pair<int, int> cur_position, std::pair<int, int> cur_goal);

public:
    explicit Follower(): pool(BS::thread_pool(cfg.num_threads)){}
    void init(const Config& cfg_, py::array_t<double> array, std::vector<std::pair<int, int>> abs_starts);
    void set_state(std::vector<std::pair<int, int>> cur_positions, std::vector<std::pair<int, int>> goals);
    void set_actions(std::vector<int> actions);
    std::vector<float> observe(size_t agent_idx, int radius);
    std::vector<float> get_gates() const { return gates; }  // the agents' gates of their last observe (1 without gate)
    // tests: the planner flow of one agent (H x W x 4, see planner::get_flow_matrix) and its occupation counts
    std::vector<std::vector<std::vector<float>>> get_flow(size_t agent_idx) const { return planners[agent_idx].get_flow_matrix(); }
    std::vector<std::vector<float>> get_occupations(size_t agent_idx) { return planners[agent_idx].get_num_occupied_matrix(); }
    std::vector<int> act(std::vector<std::pair<int, int>> cur_positions, std::vector<std::pair<int, int>> goals);
};

