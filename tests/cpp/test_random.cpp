#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "odss/random.hpp"

namespace {

TEST(RandomKey, StoresTheCompleteRandomAddressPrefix) {
  const odss::RandomKey key(11, 22, 33, 44, 55);

  EXPECT_EQ(key.master_seed(), 11U);
  EXPECT_EQ(key.scenario_id(), 22U);
  EXPECT_EQ(key.run_id(), 33U);
  EXPECT_EQ(key.object_id(), 44U);
  EXPECT_EQ(key.stream_id(), 55U);
  EXPECT_EQ(key, odss::RandomKey(11, 22, 33, 44, 55));
}

TEST(RandomValues, MatchThePublishedPhilox4x64TenRoundVector) {
  const odss::RandomKey zero_key(0, 0, 0, 0, 0);
  constexpr std::array<std::uint64_t, 4> expected = {
      0x16554D9ECA36314CULL,
      0xDB20FE9D672D0FDCULL,
      0xD7E772CEE186176BULL,
      0x7E68B68AEC7BA23BULL,
  };

  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(odss::random_u64(zero_key, index), expected[index]);
  }
}

TEST(RandomValues, MatchThePublishedNonzeroPhiloxVector) {
  constexpr std::uint64_t block = 0x082EFA98EC4E6C89ULL;
  constexpr std::uint64_t first_draw = block * 4;
  const odss::RandomKey key(0x452821E638D01377ULL, 0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
                            0xA4093822299F31D0ULL, 0xBE5466CF34E90C6CULL);
  constexpr std::array<std::uint64_t, 4> expected = {
      0xA528F45403E61D95ULL,
      0x38C72DBD566E9788ULL,
      0xA5A1610E72FD18B5ULL,
      0x57BD43B5E52B7FE6ULL,
  };

  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(odss::random_u64(key, first_draw + index), expected[index]);
  }
}

TEST(RandomValues, SupportDeterministicRandomAccess) {
  const odss::RandomKey key(1234, 8, 13, 21, 34);
  const auto first = odss::random_u64(key, 1'000'000);

  EXPECT_EQ(odss::random_u64(key, 1'000'000), first);
  EXPECT_EQ(odss::random_u64(key, 7), odss::random_u64(key, 7));
  EXPECT_NE(odss::random_u64(key, 999'999), first);
  EXPECT_NE(odss::random_u64(key, 1'000'001), first);
}

TEST(RandomValues, SeparateEveryAddressDimension) {
  const odss::RandomKey baseline(1, 2, 3, 4, 5);
  const auto value = odss::random_u64(baseline, 6);

  EXPECT_NE(odss::random_u64(odss::RandomKey(7, 2, 3, 4, 5), 6), value);
  EXPECT_NE(odss::random_u64(odss::RandomKey(1, 7, 3, 4, 5), 6), value);
  EXPECT_NE(odss::random_u64(odss::RandomKey(1, 2, 7, 4, 5), 6), value);
  EXPECT_NE(odss::random_u64(odss::RandomKey(1, 2, 3, 7, 5), 6), value);
  EXPECT_NE(odss::random_u64(odss::RandomKey(1, 2, 3, 4, 7), 6), value);
}

TEST(RandomValues, ProduceUnitIntervalValues) {
  const odss::RandomKey key(9876, 5, 4, 3, 2);

  for (std::uint64_t draw_index = 0; draw_index < 1000; ++draw_index) {
    const double value = odss::uniform_01(key, draw_index);
    EXPECT_GE(value, 0.0);
    EXPECT_LT(value, 1.0);
  }
}

TEST(RandomValues, DoNotDependOnThreadScheduling) {
  constexpr std::size_t draw_count = 4096;
  const odss::RandomKey key(2026, 17, 3, 0, 9);
  std::vector<std::uint64_t> expected(draw_count);
  for (std::size_t index = 0; index < draw_count; ++index) {
    expected[index] = odss::random_u64(key, index);
  }

  for (const std::size_t thread_count : {1U, 2U, 4U, 7U}) {
    std::vector<std::uint64_t> actual(draw_count);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
      workers.emplace_back([&, worker, thread_count] {
        for (std::size_t index = worker; index < draw_count; index += thread_count) {
          actual[index] = odss::random_u64(key, index);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    EXPECT_EQ(actual, expected);
  }
}

} // namespace
