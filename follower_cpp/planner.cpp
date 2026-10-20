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
            .def("update_occupations", &planner::update_occupations);

    py::class_<multi_planner>(m, "multi_planner")
            .def(py::init<std::vector<std::vector<int>>, bool, bool, bool, std::vector<std::pair<int, int>>, double>())
            .def("set_penalties", &multi_planner::set_penalties)
            .def("step", &multi_planner::step);
}

<%
cfg['extra_compile_args'] = ['-std=c++17', '-O3']
cfg['dependencies'] = ['planner.h']
setup_pybind11(cfg)
%>