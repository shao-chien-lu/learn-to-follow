// cppimport
#include "planner.h"

PYBIND11_MODULE(planner, m) {
    py::class_<planner>(m, "planner")
            .def(py::init<std::vector<std::vector<int>>, bool, bool, bool>())
            .def("set_abs_start", &planner::set_abs_start)
            .def("update_path", &planner::update_path)
            .def("get_path", &planner::get_path)
            .def("get_next_node", &planner::get_next_node)
            .def("precompute_penalty_matrix", &planner::precompute_penalty_matrix)
            .def("set_penalties", &planner::set_penalties)
            .def("set_crisscross", &planner::set_crisscross)
            .def("set_guidance", &planner::set_guidance)
            .def("set_guidance_gate", &planner::set_guidance_gate)
            .def("set_gate", &planner::set_gate)
            .def("get_gate", &planner::get_gate)
            .def("set_dflow", &planner::set_dflow)
            .def("add_observed_move", &planner::add_observed_move)
            .def("get_flow_matrix", &planner::get_flow_matrix)
            .def("get_num_occupied_matrix", &planner::get_num_occupied_matrix)
            .def("update_occupations", &planner::update_occupations);

    py::class_<multi_planner>(m, "multi_planner")
            .def(py::init<std::vector<std::vector<int>>, bool, bool, bool, std::vector<std::pair<int, int>>, double>())
            .def("set_penalties", &multi_planner::set_penalties)
            .def("set_crisscross", &multi_planner::set_crisscross)
            .def("set_guidance", &multi_planner::set_guidance)
            .def("set_guidance_gate", &multi_planner::set_guidance_gate)
            .def("set_dflow", &multi_planner::set_dflow)
            .def("get_gates", &multi_planner::get_gates)
            .def("get_flow", &multi_planner::get_flow)
            .def("get_occupations", &multi_planner::get_occupations)
            .def("step", &multi_planner::step);
}

<%
cfg['extra_compile_args'] = ['-std=c++17', '-O3']
cfg['dependencies'] = ['planner.h']
setup_pybind11(cfg)
%>