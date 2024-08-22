# pragma once

#include <Fingerprint.h>

#include <algorithm>


namespace MinutiaSelectionVariants
{
	using MinutiaSelection = std::vector<MinutiaPtr>(const std::vector<MinutiaPtr>&);

	std::vector<MinutiaPtr> all(const std::vector<MinutiaPtr>& minutiae)
	{
		return minutiae;
	}

	template<size_t numMinutiae>
	std::vector<MinutiaPtr> highestQuality(const std::vector<MinutiaPtr>& minutiae)
	{
		std::vector<MinutiaPtr> sorted = minutiae;
		const auto compareQuality = [](const MinutiaPtr& minu1, const MinutiaPtr& minu2) { return minu1->qual > minu2->qual; };
		std::sort(
			sorted.begin(),
			sorted.end(),
			compareQuality
		);
		sorted.resize(std::min(numMinutiae, sorted.size()));
		return sorted;
	}
}