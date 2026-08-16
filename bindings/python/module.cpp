#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>

#include "odss/data_model.hpp"
#include "odss/version.hpp"

namespace py = pybind11;

namespace {

template <typename Range>
py::tuple as_tuple(const Range& values) {
  py::tuple result(values.size());
  std::size_t index = 0;
  for (const auto& value : values) {
    result[index] = py::cast(value);
    ++index;
  }
  return result;
}

template <typename Type>
void bind_equality(py::class_<Type>& type) {
  type.def(
      "__eq__", [](const Type& lhs, const Type& rhs) { return lhs == rhs; }, py::is_operator());
}

} // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "Compiled odss core";
  module.def("version", &odss::version, "Return the library version.");

  auto epoch = py::class_<odss::Epoch>(
                   module, "Epoch",
                   "An instant as SI seconds from an explicit reference epoch and time scale.")
                   .def(py::init<double, std::string, std::string>(), py::arg("offset_s"),
                        py::arg("reference_epoch"), py::arg("time_scale"))
                   .def_property_readonly("offset_s", &odss::Epoch::offset_s)
                   .def_property_readonly("reference_epoch", &odss::Epoch::reference_epoch)
                   .def_property_readonly("time_scale", &odss::Epoch::time_scale);
  bind_equality(epoch);

  auto reference_frame =
      py::class_<odss::ReferenceFrame>(module, "ReferenceFrame",
                                       "An opaque identifier for a Cartesian coordinate frame.")
          .def(py::init<std::string>(), py::arg("identifier"))
          .def_property_readonly("identifier", &odss::ReferenceFrame::identifier);
  bind_equality(reference_frame);

  auto cartesian_state =
      py::class_<odss::CartesianState>(
          module, "CartesianState",
          "Position and velocity in SI units with explicit epoch and frame metadata.")
          .def(py::init<std::array<double, 3>, std::array<double, 3>, odss::Epoch,
                        odss::ReferenceFrame>(),
               py::arg("position_m"), py::arg("velocity_m_s"), py::arg("epoch"), py::arg("frame"))
          .def_property_readonly(
              "position_m",
              [](const odss::CartesianState& self) { return as_tuple(self.position_m()); })
          .def_property_readonly(
              "velocity_m_s",
              [](const odss::CartesianState& self) { return as_tuple(self.velocity_m_s()); })
          .def_property_readonly("epoch", &odss::CartesianState::epoch,
                                 py::return_value_policy::reference_internal)
          .def_property_readonly("frame", &odss::CartesianState::frame,
                                 py::return_value_policy::reference_internal);
  bind_equality(cartesian_state);

  auto physical_properties =
      py::class_<odss::PhysicalProperties>(module, "PhysicalProperties",
                                           "An object's positive mass and area in SI units.")
          .def(py::init<double, double>(), py::arg("mass_kg"), py::arg("area_m2"))
          .def_property_readonly("mass_kg", &odss::PhysicalProperties::mass_kg)
          .def_property_readonly("area_m2", &odss::PhysicalProperties::area_m2);
  bind_equality(physical_properties);

  auto particle_population =
      py::class_<odss::ParticlePopulation>(
          module, "ParticlePopulation",
          "Immutable particle fields stored internally as contiguous structure-of-arrays data.")
          .def(py::init<odss::Epoch, odss::ReferenceFrame, std::vector<double>, std::vector<double>,
                        std::vector<double>, std::vector<double>, std::vector<double>,
                        std::vector<double>, std::vector<double>, std::vector<double>>(),
               py::arg("epoch"), py::arg("frame"), py::arg("position_x_m"), py::arg("position_y_m"),
               py::arg("position_z_m"), py::arg("velocity_x_m_s"), py::arg("velocity_y_m_s"),
               py::arg("velocity_z_m_s"), py::arg("mass_kg"), py::arg("area_m2"))
          .def("__len__", &odss::ParticlePopulation::size)
          .def_property_readonly("size", &odss::ParticlePopulation::size)
          .def_property_readonly("empty", &odss::ParticlePopulation::empty)
          .def_property_readonly("epoch", &odss::ParticlePopulation::epoch,
                                 py::return_value_policy::reference_internal)
          .def_property_readonly("frame", &odss::ParticlePopulation::frame,
                                 py::return_value_policy::reference_internal)
          .def_property_readonly(
              "position_x_m",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.position_x_m()); })
          .def_property_readonly(
              "position_y_m",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.position_y_m()); })
          .def_property_readonly(
              "position_z_m",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.position_z_m()); })
          .def_property_readonly(
              "velocity_x_m_s",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.velocity_x_m_s()); })
          .def_property_readonly(
              "velocity_y_m_s",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.velocity_y_m_s()); })
          .def_property_readonly(
              "velocity_z_m_s",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.velocity_z_m_s()); })
          .def_property_readonly(
              "mass_kg",
              [](const odss::ParticlePopulation& self) { return as_tuple(self.mass_kg()); })
          .def_property_readonly("area_m2", [](const odss::ParticlePopulation& self) {
            return as_tuple(self.area_m2());
          });
  bind_equality(particle_population);

  auto backend_spec =
      py::class_<odss::BackendSpec>(module, "BackendSpec", "A validated backend-kind identifier.")
          .def(py::init<std::string>(), py::arg("kind"))
          .def_property_readonly("kind", &odss::BackendSpec::kind);
  bind_equality(backend_spec);

  auto observable_spec = py::class_<odss::ObservableSpec>(module, "ObservableSpec",
                                                          "A validated observable-kind identifier.")
                             .def(py::init<std::string>(), py::arg("kind"))
                             .def_property_readonly("kind", &odss::ObservableSpec::kind);
  bind_equality(observable_spec);

  auto experiment_spec =
      py::class_<odss::ExperimentSpec>(
          module, "ExperimentSpec",
          "An immutable initial population, backend choice, and requested observables.")
          .def(py::init<odss::ParticlePopulation, odss::BackendSpec,
                        std::vector<odss::ObservableSpec>>(),
               py::arg("population"), py::arg("backend"), py::arg("observables"))
          .def_property_readonly("population", &odss::ExperimentSpec::population,
                                 py::return_value_policy::reference_internal)
          .def_property_readonly("backend", &odss::ExperimentSpec::backend,
                                 py::return_value_policy::reference_internal)
          .def_property_readonly("observables", [](const odss::ExperimentSpec& self) {
            return as_tuple(self.observables());
          });
  bind_equality(experiment_spec);
}
