#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <vector>
namespace py = pybind11;

class Config
{
public:
    bool use_static_cost = true;
    bool use_dynamic_cost = true;
    bool reset_dynamic_cost = true;
    float crisscross_ratio = 1.f;  // soft crisscross highways, see planner::set_crisscross
    bool crisscross_all = false;
    // per-map guidance table, H*W*4 move cost factors replacing the crisscross factor (empty: off), see
    // planner::set_guidance; H x W is the grid passed to Follower::init (obstacle border included)
    std::vector<float> guidance_table;
    // density gate of the guidance table (off by default), see Follower::observe and planner::gate: every agent blends
    // crisscross and table factors with g = clip((rho - rho0) / (rho1 - rho0), 0, 1), rho = the other agents in its own
    // obs_radius window / the free cells there, smoothed by an EMA of half-life gate_halflife steps (<= 0: raw rho)
    bool guidance_gate = false;
    float gate_rho0 = 0.f;
    float gate_rho1 = 0.1f;
    float gate_halflife = 4.f;
    // direction-aware dynamic cost from the observed flow of other agents (eval-only), see planner::set_dflow:
    // "off" (default, bit-identical), "add", "split" (weights dflow_with / dflow_opp) or "ggboost"
    std::string dflow = "off";
    float dflow_with = 1.f;
    float dflow_opp = 1.f;
    int obs_radius = 5;
    int num_threads = 8;
    int seed = 42;
    bool use_memory = false;  // two short-term memory input channels, see follower.cpp
    std::string neighbour_memory = "none";  // memory of the other agents: "none" or "trail", see follower.cpp
    std::string ablate_channels = "";  // comma-separated input channels fed as zeros, see Follower::init
    std::string path_to_weights = "";
};
