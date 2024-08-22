#include <Application.h>

#include <HashtableVariants.hpp>
#include <MinutiaSelectionVariants.hpp>

namespace
{
	template <class HashtableType>
	std::unique_ptr<AbstractDelaunayIndex> makeDelaunayIndex(
		HashtableType&& hashtable,
		const AbstractDelaunayIndex::MinutiaSelection& selectionFun,
		std::vector<Fingerprint>&& fingerprints)
	{
		auto ptr = new DelaunayIndex<HashtableType>(
			std::move(hashtable),
			selectionFun,
			std::move(fingerprints));
		return std::unique_ptr<AbstractDelaunayIndex>(static_cast<AbstractDelaunayIndex*>(ptr));
	}
}

Application::Application(
	HashingMode hashingMode,
	MinutiaSelectionMode selectionMode,
	std::vector<Fingerprint>&& fingerprints) :
	hashtableVar(hashingMode), selectionVar(selectionMode)
{
	MinutiaSelectionVariants::MinutiaSelection* minutiaSelection = nullptr;
	if (selectionMode == MinutiaSelectionMode::KEEP_ALL)
	{
		minutiaSelection = MinutiaSelectionVariants::all;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST05)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<5>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST10)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<10>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST15)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<15>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST20)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<20>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST30)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<30>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST40)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<40>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST50)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<50>;
	}
	else if (selectionMode == MinutiaSelectionMode::QUALITY_BEST60)
	{
		minutiaSelection = MinutiaSelectionVariants::highestQuality<60>;
	}
	else
	{
		throw std::invalid_argument("Minutia selection mode not implemented!");
	}

	if (hashingMode == HashingMode::HASH_QUAL)
	{
		delaunayIndex = makeDelaunayIndex<HashtableVariants::Hashtable1D>(
			std::move(HashtableVariants::hashQuality(100)),
			minutiaSelection,
			std::move(fingerprints));
	}
	else if (hashingMode == HashingMode::HASH_RFDD)
	{
		delaunayIndex = makeDelaunayIndex<HashtableVariants::Hashtable2D>(
			std::move(HashtableVariants::hashRFDD(45)),
			minutiaSelection,
			std::move(fingerprints));
	}
	else if (hashingMode == HashingMode::HASH_GEOM)
	{
		delaunayIndex = makeDelaunayIndex<HashtableVariants::Hashtable3D>(
			std::move(HashtableVariants::hashGeom(40, 30, 16)),
			minutiaSelection,
			std::move(fingerprints));
	}
	else if (hashingMode == HashingMode::HASH_GEOM_RFDD)
	{
		delaunayIndex = makeDelaunayIndex<HashtableVariants::Hashtable5D>(
			std::move(HashtableVariants::hashGeomRFDD(12, 9, 8, 9)),
			minutiaSelection,
			std::move(fingerprints));
	}
	else if (hashingMode == HashingMode::HASH_ALL)
	{
		delaunayIndex = makeDelaunayIndex<HashtableVariants::Hashtable6D>(
			std::move(HashtableVariants::hashAll(12, 9, 8, 9, 10)),
			minutiaSelection,
			std::move(fingerprints));
	}
	else
	{
		throw std::invalid_argument("Hashing mode not implemented!");
	}
	return;
}

void Application::reportBinDistribution(const std::string& filename) const
{
	delaunayIndex->reportBinDistribution(filename);
}

std::pair<bool, size_t> Application::search(const Fingerprint& fingerprint) const
{
	const auto result = delaunayIndex->search(fingerprint);
	return result;
}

std::pair<bool, size_t> Application::searchExhaustive(const Fingerprint& fingerprint) const
{
	const auto result = delaunayIndex->searchExhaustive(fingerprint);
	return result;
}

Application Application::create(HashingMode hashtableVar, MinutiaSelectionMode selectionVar, const std::vector<Fingerprint>& fingerprints)
{
	std::vector<Fingerprint> fingerprintsCopy;
	fingerprintsCopy.reserve(fingerprints.size());
	std::copy(fingerprints.begin(), fingerprints.end(), std::back_inserter(fingerprintsCopy));
	return Application(
		std::move(hashtableVar),
		selectionVar,
		std::move(fingerprintsCopy));
}
