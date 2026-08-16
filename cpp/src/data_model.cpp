#include "odss/data_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace odss {
namespace {

void require_finite(double value, std::string_view field_name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string{field_name} + " must be finite");
  }
}

void require_positive_finite(double value, std::string_view field_name) {
  require_finite(value, field_name);
  if (value <= 0.0) {
    throw std::invalid_argument(std::string{field_name} + " must be positive");
  }
}

void require_identifier(std::string_view value, std::string_view field_name) {
  constexpr std::string_view whitespace{" \t\n\r"};
  if (value.empty() || value.find_first_not_of(whitespace) == std::string_view::npos) {
    throw std::invalid_argument(std::string{field_name} + " must not be empty");
  }
}

template <typename Range>
void require_all_finite(const Range& values, std::string_view field_name) {
  if (!std::ranges::all_of(values, [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(std::string{field_name} + " values must be finite");
  }
}

void require_same_size(std::size_t expected,
                       std::initializer_list<std::pair<std::size_t, std::string_view>> sizes) {
  for (const auto& [size, field_name] : sizes) {
    if (size != expected) {
      throw std::invalid_argument(std::string{field_name} + " must match population size");
    }
  }
}

} // namespace

Epoch::Epoch(double offset_s, std::string reference_epoch, std::string time_scale)
    : offset_s_(offset_s), reference_epoch_(std::move(reference_epoch)),
      time_scale_(std::move(time_scale)) {
  require_finite(offset_s_, "offset_s");
  require_identifier(reference_epoch_, "reference_epoch");
  require_identifier(time_scale_, "time_scale");
}

ReferenceFrame::ReferenceFrame(std::string identifier) : identifier_(std::move(identifier)) {
  require_identifier(identifier_, "frame identifier");
}

CartesianState::CartesianState(std::array<double, 3> position_m, std::array<double, 3> velocity_m_s,
                               Epoch epoch, ReferenceFrame frame)
    : position_m_(position_m), velocity_m_s_(velocity_m_s), epoch_(std::move(epoch)),
      frame_(std::move(frame)) {
  require_all_finite(position_m_, "position_m");
  require_all_finite(velocity_m_s_, "velocity_m_s");
}

PhysicalProperties::PhysicalProperties(double mass_kg, double area_m2)
    : mass_kg_(mass_kg), area_m2_(area_m2) {
  require_positive_finite(mass_kg_, "mass_kg");
  require_positive_finite(area_m2_, "area_m2");
}

ParticlePopulation::ParticlePopulation(
    Epoch epoch, ReferenceFrame frame, std::vector<double> position_x_m,
    std::vector<double> position_y_m, std::vector<double> position_z_m,
    std::vector<double> velocity_x_m_s, std::vector<double> velocity_y_m_s,
    std::vector<double> velocity_z_m_s, std::vector<double> mass_kg, std::vector<double> area_m2)
    : epoch_(std::move(epoch)), frame_(std::move(frame)), position_x_m_(std::move(position_x_m)),
      position_y_m_(std::move(position_y_m)), position_z_m_(std::move(position_z_m)),
      velocity_x_m_s_(std::move(velocity_x_m_s)), velocity_y_m_s_(std::move(velocity_y_m_s)),
      velocity_z_m_s_(std::move(velocity_z_m_s)), mass_kg_(std::move(mass_kg)),
      area_m2_(std::move(area_m2)) {
  const auto expected = position_x_m_.size();
  require_same_size(expected, {{position_y_m_.size(), "position_y_m"},
                               {position_z_m_.size(), "position_z_m"},
                               {velocity_x_m_s_.size(), "velocity_x_m_s"},
                               {velocity_y_m_s_.size(), "velocity_y_m_s"},
                               {velocity_z_m_s_.size(), "velocity_z_m_s"},
                               {mass_kg_.size(), "mass_kg"},
                               {area_m2_.size(), "area_m2"}});

  require_all_finite(position_x_m_, "position_x_m");
  require_all_finite(position_y_m_, "position_y_m");
  require_all_finite(position_z_m_, "position_z_m");
  require_all_finite(velocity_x_m_s_, "velocity_x_m_s");
  require_all_finite(velocity_y_m_s_, "velocity_y_m_s");
  require_all_finite(velocity_z_m_s_, "velocity_z_m_s");
  for (const auto mass : mass_kg_) {
    require_positive_finite(mass, "mass_kg");
  }
  for (const auto area : area_m2_) {
    require_positive_finite(area, "area_m2");
  }
}

BackendSpec::BackendSpec(std::string kind) : kind_(std::move(kind)) {
  require_identifier(kind_, "backend kind");
}

ObservableSpec::ObservableSpec(std::string kind) : kind_(std::move(kind)) {
  require_identifier(kind_, "observable kind");
}

ExperimentSpec::ExperimentSpec(ParticlePopulation population, BackendSpec backend,
                               std::vector<ObservableSpec> observables)
    : population_(std::move(population)), backend_(std::move(backend)),
      observables_(std::move(observables)) {}

} // namespace odss
