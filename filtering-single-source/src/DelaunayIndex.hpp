#pragma once

#include <Fingerprint.h>
#include <DebugDrawing.h>
#include <Triangulation.h>

#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <algorithm>
#include <unordered_map>

namespace
{
	size_t getMaxIdAllowed(const std::vector<Fingerprint>& fingerprints)
	{
		return std::max_element(
			fingerprints.begin(),
			fingerprints.end(),
			[](const auto& fp1, const auto& fp2) { return fp1.id < fp2.id; }
		)->id + 1;
	}

	std::vector<Id> getIdsSortedByCount(const std::unordered_map<Id, uint32_t>& idCounts)
	{
		std::vector<std::pair<size_t, uint32_t>> kvPairs(idCounts.size());
		std::copy(
			idCounts.begin(),
			idCounts.end(),
			kvPairs.begin()
		);
		std::sort(
			kvPairs.begin(),
			kvPairs.end(),
			[](const auto& a, const auto& b) { return b.second < a.second; }
		);

		std::vector<Id> fingerprintIds(idCounts.size());
		std::transform(
			kvPairs.begin(),
			kvPairs.end(),
			fingerprintIds.begin(),
			[](const auto& kv) { return kv.first; }
		);
		return fingerprintIds;
	}

	template <class ValueType>
	inline bool mapContainsKey(const std::unordered_map<Id, ValueType>& map, const Id& id)
	{
		return map.end() != map.find(id);
	}
}

class AbstractDelaunayIndex
{
public:
	using MinutiaSelection = std::function<std::vector<MinutiaPtr>(const std::vector<MinutiaPtr>&)>;

	virtual ~AbstractDelaunayIndex() {};

	virtual Fingerprint getFingerprint(const Id id) const = 0;

	virtual void reportBinDistribution(const std::string& filename) const = 0;

	virtual std::pair<bool, size_t> search(const Fingerprint& sample) const = 0;

	virtual std::pair<bool, size_t> searchExhaustive(const Fingerprint& sample) const = 0;
};


template<class HashtableType>
class DelaunayIndex : public AbstractDelaunayIndex
{

public:
	DelaunayIndex(
		HashtableType&& hashtableIn,
		const MinutiaSelection& minutiaSelection,
		std::vector<Fingerprint>&& fingerprintIn) :
		hashtable(std::move(hashtableIn)), minutiaSelection(minutiaSelection), fingerprints(std::move(fingerprintIn))
	{
		initializeLookupTable();
		for (const auto& fp : fingerprints)
		{
			hashtable.addValue(fp.id, getFeatures(fp));
			// debug::saveTriangulation(fp.name, extendedDelaunayTriangulation(minutiaSelection(fp.minutiae)), "all_");
		}

	}

	~DelaunayIndex() override = default;

	void reportBinDistribution(const std::string& filename) const override
	{
		hashtable.writeElementsPerBinToJson(filename);
	}

	Fingerprint getFingerprint(const Id id) const override
	{
		if (!idInIndex[id])
		{
			throw std::invalid_argument("Fingerprint with id " + std::to_string(id) + " is not in DelaunayIndex");
		}
		return fingerprints[fingerprintLookupTable[id]];
	}

	std::pair<bool, size_t> search(const Fingerprint& sample) const override
	{
		std::cout << "WARNING: DelaunayIndex.search() not implemented!" << std::endl;
		return { false, 0 };
	}

	std::pair<bool, size_t> searchExhaustive(const Fingerprint& sample) const override
	{
		auto query = HashtableType::Query(hashtable, getFeatures(sample));
		size_t numComparisons = 0;

		std::unordered_map<Id, bool> fingerprintCompared;
		while (!query.exhausted())
		{
			auto queryResults = query.yield();
			if (queryResults.empty())
			{
				continue;
			}

			std::unordered_map<Id, uint32_t> idCounts;
			for (const auto& id : queryResults)
			{
				if (mapContainsKey(fingerprintCompared, id))
				{
					continue;
				}
				if (!mapContainsKey(idCounts, id))
				{
					idCounts.insert({ id, 1 });
				}
				else
				{
					idCounts[id] += 1;
				}
			}

			const auto sortedIds = getIdsSortedByCount(idCounts);
			const auto searchResult = searchFingerprintId(sample, sortedIds);
			numComparisons += searchResult.second;
			if (searchResult.first)
			{
				return { true, numComparisons };
			}

			for (const auto& id : sortedIds)
			{
				fingerprintCompared.insert({ id, true });
			}
		}
		return { false, numComparisons };
	}

private:
	void initializeLookupTable() {
		fingerprintLookupTable.resize(getMaxIdAllowed(fingerprints));
		idInIndex.resize(fingerprintLookupTable.size(), false);
		for (size_t i = 0; i < fingerprints.size(); i++)
		{
			const auto& fp = fingerprints[i];
			fingerprintLookupTable[fp.id] = i;
			idInIndex[fp.id] = true;
		}
	}

	std::vector<TriangleFeatures> getFeatures(const Fingerprint& fingerprint) const
	{
		const auto s = minutiaSelection(fingerprint.minutiae);
		const auto triangles = extendedDelaunayTriangulation(s);
		if (triangles.empty())
		{
			std::cout << "WARNING: Input fingerprint " << fingerprint.name << " does not have enough minutiae ("
				<< fingerprint.minutiae.size() << ") to triangulate" << std::endl;
		}
		std::vector<TriangleFeatures> featureVectors;
		std::transform(
			triangles.begin(),
			triangles.end(),
			std::back_inserter(featureVectors),
			getTriangleFeatures
		);
		return featureVectors;
	}

	std::pair<bool, size_t> searchFingerprintId(const Fingerprint& fingerprint, const std::vector<Id>& bin) const
	{
		size_t numComparisons = 0;
		for (const auto& fingerprintId : bin)
		{
			const auto& fp = getFingerprint(fingerprintId);
			if (fingerprint.subjectId == getFingerprint(fingerprintId).subjectId)
			{
				return { true, numComparisons + 1 };
			}
			numComparisons++;
		}
		return { false, numComparisons };
	}

	HashtableType hashtable;
	const MinutiaSelection minutiaSelection;
	std::vector<Fingerprint> fingerprints;
	std::vector<size_t> fingerprintLookupTable;
	std::vector<bool> idInIndex;
};
