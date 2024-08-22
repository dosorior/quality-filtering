#pragma once

#include <DelaunayIndex.hpp>

#include <cstdint>
#include <vector>
#include <string>

namespace utils
{
	std::vector<Fingerprint> loadFingerprintsFromJson(const std::string& filename);
}
