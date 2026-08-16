#include <pybind11/pybind11.h>

#include "odss/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
  module.doc() = "Compiled odss core";
  module.def("version", &odss::version, "Return the library version.");
}
