#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "odss/data_model.hpp"

namespace {

odss::Epoch epoch() { return {123.5, "J2000", "TAI"}; }

odss::ReferenceFrame frame() { return odss::ReferenceFrame{"GCRF"}; }

odss::ParticlePopulation population() {
  return {epoch(),    frame(),     {1.0, 2.0},   {3.0, 4.0},     {5.0, 6.0},
          {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}, {100.0, 200.0}, {2.0, 4.0}};
}

TEST(Epoch, StoresExplicitMetadataAndComparesByValue) {
  const odss::Epoch value = epoch();

  EXPECT_DOUBLE_EQ(value.offset_s(), 123.5);
  EXPECT_EQ(value.reference_epoch(), "J2000");
  EXPECT_EQ(value.time_scale(), "TAI");
  EXPECT_EQ(value, epoch());
  EXPECT_NE(value, odss::Epoch(124.5, "J2000", "TAI"));
}

TEST(Epoch, RejectsInvalidValues) {
  EXPECT_THROW(odss::Epoch(std::nan(""), "J2000", "TAI"), std::invalid_argument);
  EXPECT_THROW(odss::Epoch(0.0, "", "TAI"), std::invalid_argument);
  EXPECT_THROW(odss::Epoch(0.0, "J2000", "  "), std::invalid_argument);
}

TEST(ReferenceFrame, RequiresAnIdentifier) {
  EXPECT_EQ(frame().identifier(), "GCRF");
  EXPECT_THROW(odss::ReferenceFrame("\t"), std::invalid_argument);
}

TEST(CartesianState, StoresSiStateWithEpochAndFrame) {
  const odss::CartesianState state({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, epoch(), frame());

  EXPECT_EQ(state.position_m(), (std::array<double, 3>{1.0, 2.0, 3.0}));
  EXPECT_EQ(state.velocity_m_s(), (std::array<double, 3>{4.0, 5.0, 6.0}));
  EXPECT_EQ(state.epoch(), epoch());
  EXPECT_EQ(state.frame(), frame());
  EXPECT_THROW(odss::CartesianState({std::nan(""), 0.0, 0.0}, {0.0, 0.0, 0.0}, epoch(), frame()),
               std::invalid_argument);
}

TEST(PhysicalProperties, RequiresPositiveFiniteSiValues) {
  const odss::PhysicalProperties properties(10.0, 2.5);

  EXPECT_DOUBLE_EQ(properties.mass_kg(), 10.0);
  EXPECT_DOUBLE_EQ(properties.area_m2(), 2.5);
  EXPECT_EQ(properties, odss::PhysicalProperties(10.0, 2.5));
  EXPECT_THROW(odss::PhysicalProperties(0.0, 1.0), std::invalid_argument);
  EXPECT_THROW(odss::PhysicalProperties(1.0, INFINITY), std::invalid_argument);
}

TEST(ParticlePopulation, StoresContiguousStructureOfArrays) {
  const auto particles = population();

  EXPECT_EQ(particles.size(), 2U);
  EXPECT_FALSE(particles.empty());
  EXPECT_EQ(particles.epoch(), epoch());
  EXPECT_EQ(particles.frame(), frame());
  EXPECT_EQ(particles.position_x_m(), (std::vector<double>{1.0, 2.0}));
  EXPECT_EQ(particles.velocity_z_m_s(), (std::vector<double>{11.0, 12.0}));
  EXPECT_EQ(particles.mass_kg(), (std::vector<double>{100.0, 200.0}));
  EXPECT_EQ(particles.position_x_m().data() + 1, &particles.position_x_m()[1]);
  EXPECT_EQ(particles, population());
}

TEST(ParticlePopulation, AllowsAConsistentEmptyPopulation) {
  const odss::ParticlePopulation particles(epoch(), frame(), {}, {}, {}, {}, {}, {}, {}, {});

  EXPECT_TRUE(particles.empty());
  EXPECT_EQ(particles.size(), 0U);
}

TEST(ParticlePopulation, RejectsMismatchedOrInvalidFields) {
  EXPECT_THROW(odss::ParticlePopulation(epoch(), frame(), {1.0}, {}, {1.0}, {1.0}, {1.0}, {1.0},
                                        {1.0}, {1.0}),
               std::invalid_argument);
  EXPECT_THROW(odss::ParticlePopulation(epoch(), frame(), {std::nan("")}, {1.0}, {1.0}, {1.0},
                                        {1.0}, {1.0}, {1.0}, {1.0}),
               std::invalid_argument);
  EXPECT_THROW(odss::ParticlePopulation(epoch(), frame(), {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0},
                                        {-1.0}, {1.0}),
               std::invalid_argument);
}

TEST(Specifications, AreMinimalValidatedValues) {
  const odss::BackendSpec backend("reference");
  const odss::ObservableSpec observable("state_count");
  const odss::ExperimentSpec experiment(population(), backend, {observable});

  EXPECT_EQ(backend.kind(), "reference");
  EXPECT_EQ(observable.kind(), "state_count");
  EXPECT_EQ(experiment.population(), population());
  EXPECT_EQ(experiment.backend(), backend);
  EXPECT_EQ(experiment.observables(), (std::vector<odss::ObservableSpec>{observable}));
  EXPECT_EQ(experiment, odss::ExperimentSpec(population(), backend, {observable}));
  EXPECT_THROW(odss::BackendSpec(""), std::invalid_argument);
  EXPECT_THROW(odss::ObservableSpec("\n"), std::invalid_argument);
}

} // namespace
