#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
namespace py = pybind11;

class Config
{
public:
    bool use_static_cost = true;
    bool use_dynamic_cost = true;
    bool reset_dynamic_cost = true;
    float crisscross_ratio = 1.f;  // soft crisscross highways, see planner::set_crisscross
    bool crisscross_all = false;
    int obs_radius = 5;
    int num_threads = 8;
    int seed = 42;
    bool use_memory = false;  // two short-term memory input channels, see follower.cpp
    std::string neighbour_memory = "none";  // memory of the other agents: "none" or "trail", see follower.cpp
    std::string ablate_channels = "";  // comma-separated input channels fed as zeros, see Follower::init
    std::string path_to_weights = "";
};
