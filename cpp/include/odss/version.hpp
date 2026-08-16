#pragma once

#include <string_view>

namespace odss {

/// Return the semantic version of the compiled core.
[[nodiscard]] std::string_view version() noexcept;

} // namespace odss
