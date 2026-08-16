#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace odss {

/// An instant represented as SI seconds from an explicitly named reference epoch.
///
/// This value stores time metadata only. It does not interpret time scales or
/// convert between epoch representations.
class Epoch {
public:
  Epoch(double offset_s, std::string reference_epoch, std::string time_scale);

  double offset_s() const noexcept { return offset_s_; }

  const std::string& reference_epoch() const noexcept { return reference_epoch_; }

  const std::string& time_scale() const noexcept { return time_scale_; }

  bool operator==(const Epoch&) const = default;

private:
  double offset_s_;
  std::string reference_epoch_;
  std::string time_scale_;
};

/// Opaque identifier for the coordinate frame of Cartesian state data.
class ReferenceFrame {
public:
  explicit ReferenceFrame(std::string identifier);

  const std::string& identifier() const noexcept { return identifier_; }

  bool operator==(const ReferenceFrame&) const = default;

private:
  std::string identifier_;
};

/// Position and velocity in one explicitly identified frame and epoch.
class CartesianState {
public:
  CartesianState(std::array<double, 3> position_m, std::array<double, 3> velocity_m_s, Epoch epoch,
                 ReferenceFrame frame);

  const std::array<double, 3>& position_m() const noexcept { return position_m_; }

  const std::array<double, 3>& velocity_m_s() const noexcept { return velocity_m_s_; }

  const Epoch& epoch() const noexcept { return epoch_; }

  const ReferenceFrame& frame() const noexcept { return frame_; }

  bool operator==(const CartesianState&) const = default;

private:
  std::array<double, 3> position_m_;
  std::array<double, 3> velocity_m_s_;
  Epoch epoch_;
  ReferenceFrame frame_;
};

/// Minimal physical properties used to identify an object's mass and area.
class PhysicalProperties {
public:
  PhysicalProperties(double mass_kg, double area_m2);

  double mass_kg() const noexcept { return mass_kg_; }
  double area_m2() const noexcept { return area_m2_; }

  bool operator==(const PhysicalProperties&) const = default;

private:
  double mass_kg_;
  double area_m2_;
};

/// Immutable particle data stored as contiguous structure-of-arrays fields.
class ParticlePopulation {
public:
  ParticlePopulation(Epoch epoch, ReferenceFrame frame, std::vector<double> position_x_m,
                     std::vector<double> position_y_m, std::vector<double> position_z_m,
                     std::vector<double> velocity_x_m_s, std::vector<double> velocity_y_m_s,
                     std::vector<double> velocity_z_m_s, std::vector<double> mass_kg,
                     std::vector<double> area_m2);

  std::size_t size() const noexcept { return position_x_m_.size(); }
  bool empty() const noexcept { return position_x_m_.empty(); }

  const Epoch& epoch() const noexcept { return epoch_; }

  const ReferenceFrame& frame() const noexcept { return frame_; }

  const std::vector<double>& position_x_m() const noexcept { return position_x_m_; }
  const std::vector<double>& position_y_m() const noexcept { return position_y_m_; }
  const std::vector<double>& position_z_m() const noexcept { return position_z_m_; }
  const std::vector<double>& velocity_x_m_s() const noexcept { return velocity_x_m_s_; }
  const std::vector<double>& velocity_y_m_s() const noexcept { return velocity_y_m_s_; }
  const std::vector<double>& velocity_z_m_s() const noexcept { return velocity_z_m_s_; }
  const std::vector<double>& mass_kg() const noexcept { return mass_kg_; }
  const std::vector<double>& area_m2() const noexcept { return area_m2_; }

  bool operator==(const ParticlePopulation&) const = default;

private:
  Epoch epoch_;
  ReferenceFrame frame_;
  std::vector<double> position_x_m_;
  std::vector<double> position_y_m_;
  std::vector<double> position_z_m_;
  std::vector<double> velocity_x_m_s_;
  std::vector<double> velocity_y_m_s_;
  std::vector<double> velocity_z_m_s_;
  std::vector<double> mass_kg_;
  std::vector<double> area_m2_;
};

/// Identifier for a future computational backend configuration.
class BackendSpec {
public:
  explicit BackendSpec(std::string kind);

  const std::string& kind() const noexcept { return kind_; }

  bool operator==(const BackendSpec&) const = default;

private:
  std::string kind_;
};

/// Identifier for a requested experiment output.
class ObservableSpec {
public:
  explicit ObservableSpec(std::string kind);

  const std::string& kind() const noexcept { return kind_; }

  bool operator==(const ObservableSpec&) const = default;

private:
  std::string kind_;
};

/// Minimal immutable experiment definition for an initial population.
class ExperimentSpec {
public:
  ExperimentSpec(ParticlePopulation population, BackendSpec backend,
                 std::vector<ObservableSpec> observables);

  const ParticlePopulation& population() const noexcept { return population_; }
  const BackendSpec& backend() const noexcept { return backend_; }
  const std::vector<ObservableSpec>& observables() const noexcept { return observables_; }

  bool operator==(const ExperimentSpec&) const = default;

private:
  ParticlePopulation population_;
  BackendSpec backend_;
  std::vector<ObservableSpec> observables_;
};

} // namespace odss
