// cppimport
#include "config.h"

PYBIND11_MODULE(config, m) {
py::class_<Config>(m, "Config")
    .def(py::init<>())
    .def_readwrite("use_static_cost", &Config::use_static_cost)
    .def_readwrite("use_dynamic_cost", &Config::use_dynamic_cost)
    .def_readwrite("reset_dynamic_cost", &Config::reset_dynamic_cost)
    .def_readwrite("crisscross_ratio", &Config::crisscross_ratio)
    .def_readwrite("crisscross_all", &Config::crisscross_all)
    .def_readwrite("guidance_table", &Config::guidance_table)
    .def_readwrite("guidance_gate", &Config::guidance_gate)
    .def_readwrite("gate_rho0", &Config::gate_rho0)
    .def_readwrite("gate_rho1", &Config::gate_rho1)
    .def_readwrite("gate_halflife", &Config::gate_halflife)
    .def_readwrite("dflow", &Config::dflow)
    .def_readwrite("dflow_with", &Config::dflow_with)
    .def_readwrite("dflow_opp", &Config::dflow_opp)
    .def_readwrite("obs_radius", &Config::obs_radius)
    .def_readwrite("num_threads", &Config::num_threads)
    .def_readwrite("seed", &Config::seed)
    .def_readwrite("use_memory", &Config::use_memory)
    .def_readwrite("neighbour_memory", &Config::neighbour_memory)
    .def_readwrite("ablate_channels", &Config::ablate_channels)
    .def_readwrite("path_to_weights", &Config::path_to_weights)
;
}
<%
cfg['extra_compile_args'] = ['-std=c++17']
cfg['dependencies'] = ['config.h']
setup_pybind11(cfg)
%>
