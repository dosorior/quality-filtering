#pragma once

#include <Fingerprint.h>
#include <DelaunayIndex.hpp>

#include <vector>

class __declspec(dllexport) Application
{
public:
	enum HashingMode
	{
		HASH_QUAL,
		HASH_RFDD,
		HASH_GEOM,
		HASH_GEOM_RFDD,
		HASH_ALL
	};

	enum MinutiaSelectionMode
	{
		KEEP_ALL,
		QUALITY_BEST05,
		QUALITY_BEST10,
		QUALITY_BEST15,
		QUALITY_BEST20,
		QUALITY_BEST30,
		QUALITY_BEST40,
		QUALITY_BEST50,
		QUALITY_BEST60
	};

	Application(HashingMode hashtableVar, MinutiaSelectionMode selectionVar, std::vector<Fingerprint>&& fingerprints);

	void reportBinDistribution(const std::string& filename) const;

	std::pair<bool, size_t> search(const Fingerprint& fingerprint) const;

	std::pair<bool, size_t> Application::searchExhaustive(const Fingerprint& fingerprint) const;

	static Application create(HashingMode hashtableVar, MinutiaSelectionMode selectionVar, const std::vector<Fingerprint>& fingerprints);

private:
	std::unique_ptr<AbstractDelaunayIndex> delaunayIndex;
	HashingMode hashtableVar;
	MinutiaSelectionMode selectionVar;
};