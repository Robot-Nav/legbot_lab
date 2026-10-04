#pragma once

namespace w1w::device_guard {

// Returns false without touching any robot I/O when this executable is not
// running on the CPU selected at sealed-build time.
bool enforce(const char* component_name);

} // namespace w1w::device_guard
