#include <gtest/gtest.h>

#include <string_view>

#include "odss/version.hpp"

TEST(Version, IsNonEmpty) {
  EXPECT_FALSE(odss::version().empty());
  EXPECT_EQ(odss::version(), std::string_view{"0.1.0"});
}
