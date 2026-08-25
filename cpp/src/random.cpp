#include "odss/random.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace odss {
namespace {

using Counter = std::array<std::uint64_t, 4>;
using Key = std::array<std::uint64_t, 2>;

constexpr std::uint64_t multiplier_0 = 0xD2E7470EE14C6C93ULL;
constexpr std::uint64_t multiplier_1 = 0xCA5A826395121157ULL;
constexpr std::uint64_t key_increment_0 = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t key_increment_1 = 0xBB67AE8584CAA73BULL;
constexpr std::uint64_t low_word_mask = 0xFFFFFFFFULL;
constexpr std::uint64_t values_per_block = 4;

std::pair<std::uint64_t, std::uint64_t> multiply_high_low(std::uint64_t lhs,
                                                          std::uint64_t rhs) noexcept {
  const std::uint64_t lhs_low = lhs & low_word_mask;
  const std::uint64_t lhs_high = lhs >> 32U;
  const std::uint64_t rhs_low = rhs & low_word_mask;
  const std::uint64_t rhs_high = rhs >> 32U;

  const std::uint64_t product_00 = lhs_low * rhs_low;
  const std::uint64_t product_01 = lhs_low * rhs_high;
  const std::uint64_t product_10 = lhs_high * rhs_low;
  const std::uint64_t product_11 = lhs_high * rhs_high;
  const std::uint64_t middle =
      (product_00 >> 32U) + (product_01 & low_word_mask) + (product_10 & low_word_mask);

  const std::uint64_t low = (product_00 & low_word_mask) | (middle << 32U);
  const std::uint64_t high =
      product_11 + (product_01 >> 32U) + (product_10 >> 32U) + (middle >> 32U);
  return {high, low};
}

Counter philox_round(const Counter& counter, const Key& key) noexcept {
  const auto [high_0, low_0] = multiply_high_low(multiplier_0, counter[0]);
  const auto [high_1, low_1] = multiply_high_low(multiplier_1, counter[2]);
  return {high_1 ^ counter[1] ^ key[0], low_1, high_0 ^ counter[3] ^ key[1], low_0};
}

Counter philox4x64_10(Counter counter, Key key) noexcept {
  constexpr std::size_t rounds = 10;
  for (std::size_t round = 0; round < rounds; ++round) {
    counter = philox_round(counter, key);
    if (round + 1 < rounds) {
      key[0] += key_increment_0;
      key[1] += key_increment_1;
    }
  }
  return counter;
}

} // namespace

RandomKey::RandomKey(std::uint64_t master_seed, std::uint64_t scenario_id, std::uint64_t run_id,
                     std::uint64_t object_id, std::uint64_t stream_id)
    : master_seed_(master_seed), scenario_id_(scenario_id), run_id_(run_id), object_id_(object_id),
      stream_id_(stream_id) {}

std::uint64_t random_u64(const RandomKey& key, std::uint64_t draw_index) noexcept {
  const Counter counter = {key.scenario_id(), key.run_id(), key.object_id(),
                           draw_index / values_per_block};
  const Key philox_key = {key.master_seed(), key.stream_id()};
  const Counter values = philox4x64_10(counter, philox_key);
  return values[draw_index % values_per_block];
}

double uniform_01(const RandomKey& key, std::uint64_t draw_index) noexcept {
  constexpr double inverse_53_bit_range = 0x1.0p-53;
  return static_cast<double>(random_u64(key, draw_index) >> 11U) * inverse_53_bit_range;
}

} // namespace odss
