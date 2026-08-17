#pragma once

#include <cstdint>
#include <string_view>

namespace odss {

inline constexpr std::string_view rng_algorithm = "philox4x64-10-v1";

/// Immutable coordinates identifying one deterministic random stream.
class RandomKey {
public:
  RandomKey(std::uint64_t master_seed, std::uint64_t scenario_id, std::uint64_t run_id,
            std::uint64_t object_id, std::uint64_t stream_id);

  std::uint64_t master_seed() const noexcept { return master_seed_; }
  std::uint64_t scenario_id() const noexcept { return scenario_id_; }
  std::uint64_t run_id() const noexcept { return run_id_; }
  std::uint64_t object_id() const noexcept { return object_id_; }
  std::uint64_t stream_id() const noexcept { return stream_id_; }

  bool operator==(const RandomKey&) const = default;

private:
  std::uint64_t master_seed_;
  std::uint64_t scenario_id_;
  std::uint64_t run_id_;
  std::uint64_t object_id_;
  std::uint64_t stream_id_;
};

/// Return one schedule-independent random 64-bit value at draw_index.
std::uint64_t random_u64(const RandomKey& key, std::uint64_t draw_index) noexcept;

/// Return one schedule-independent uniform value in the half-open interval [0, 1).
double uniform_01(const RandomKey& key, std::uint64_t draw_index) noexcept;

} // namespace odss
