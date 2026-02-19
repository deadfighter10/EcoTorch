#include <nanobind/nanobind.h>

#include <utility>
#include "monitor_test.cpp"

namespace nb = nanobind;

NB_MODULE(apple, m) {
    nb::class_<Monitor>(std::move(m), "Monitor")
        .def(nb::init<>())
        .def("get_current_power", &Monitor::get_current_power)
        .def("get_gpu_utilization", &Monitor::get_gpu_utilization);
}