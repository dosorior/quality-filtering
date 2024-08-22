#pragma once

#include <json.hpp>

#include <functional>
#include <numeric>
#include <vector>
#include <array>
#include <cstdint>
#include <fstream>

namespace
{
	template<uint32_t numDims>
	std::array<size_t, numDims> getBlocksizeAtDim(const std::array<uint32_t, numDims>& binsPerDim)
	{
		std::array<size_t, numDims> blocksizeAtDim;
		blocksizeAtDim[numDims - 1] = 1;
		for (uint32_t dim = numDims - 1; dim > 0; dim--)
		{
			blocksizeAtDim[dim - 1] = blocksizeAtDim[dim] * binsPerDim[dim];
		}
		return blocksizeAtDim;
	}

	template <uint32_t numDims>
	inline std::array<uint32_t, numDims> changeHashEntry(
		const std::array<uint32_t, numDims>& hash,
		const uint32_t dim,
		const uint32_t newValue)
	{
		auto changedHash(hash);
		changedHash[dim] = newValue;
		return changedHash;
	}

	template <uint32_t numDims>
	void appendAdjacentHashes(
		std::vector<std::array<uint32_t, numDims>>& hashesOut,
		const std::array<uint32_t, numDims>& hash,
		const std::array<uint32_t, numDims>& sizeOfDim)
	{
		for (uint32_t dim = 0; dim < numDims; dim++)
		{
			if (hash[dim] > 0)
			{
				hashesOut.push_back(changeHashEntry(hash, dim, hash[dim] - 1));
			}
			if (hash[dim] < (sizeOfDim[dim] - 1))
			{
				hashesOut.push_back(changeHashEntry(hash, dim, hash[dim] + 1));
			}
		}
	}

	template <uint32_t numDims>
	std::vector<size_t> accumulateNumElements(
		const std::vector<size_t> elementsPerBin,
		const std::array<uint32_t, numDims>& sizeOfDim,
		const uint32_t dim)
	{
		const auto blockSize = getBlocksizeAtDim(sizeOfDim)[dim];

		std::vector<size_t> elementsAccumulated(sizeOfDim[dim], 0);
		for (size_t i = 0; i < elementsPerBin.size(); i++)
		{
			elementsAccumulated[i / blockSize % sizeOfDim[dim]] += elementsPerBin[i];
		}
		return elementsAccumulated;
	}
} // namespace

template <class ValueType, typename KeyType, uint32_t numDims>
class MultidimHashtable
{
public:
	using Hash = std::array<uint32_t, numDims>;
	using HashFunction = std::function<Hash(const KeyType&)>;
	using Bin = std::vector<size_t>;

	class Query
	{
	public:
		Query(const MultidimHashtable<ValueType, KeyType, numDims>& hashtable, const std::vector<KeyType>& keys) :
			hashtable(hashtable),
			firstQuery(true)
		{
			numVisited = 0;
			numYields = 0;
			hashAlreadyVisited.resize(hashtable.bins.size(), false);
			updateCurrentHashes(hashtable.getHashes(keys));
			for (const auto& hash : currentHashes) {
				hashAlreadyVisited[hashtable.getIndex(hash)] = true;
			}
		}

		std::vector<ValueType> yield()
		{
			if (firstQuery)
			{
				firstQuery = false;
			}
			else
			{
				updateCurrentHashes(getAdjacentHashes(currentHashes));
			}
			numYields++;
			return getValues(getBins(currentHashes));
		}

		bool exhausted() const
		{
			return currentHashes.empty();
		}

	private:
		void updateCurrentHashes(const std::vector<Hash>& newHashes)
		{
			currentHashes.clear();
			for (const auto& hash : newHashes)
			{
				const auto index = hashtable.getIndex(hash);
				if (!hashAlreadyVisited[index])
				{
					hashAlreadyVisited[index] = true;
					currentHashes.push_back(hash);
					numVisited++;
				}
			}
		}

		std::vector<Hash> getAdjacentHashes(const std::vector<Hash>& hashes)
		{
			std::vector<Hash> allAdjacent;
			for (const auto& hash : hashes)
			{
				appendAdjacentHashes(allAdjacent, hash, hashtable.sizeOfDim);
			}
			return allAdjacent;
		}

		std::vector<const Bin*> getBins(const std::vector<Hash>& hashes)
		{
			std::vector<const Bin*> bins;
			bins.reserve(hashes.size());
			std::transform(
				hashes.begin(),
				hashes.end(),
				std::back_inserter(bins),
				[this](const auto& hash) { return this->hashtable.getBin(this->hashtable.getIndex(hash)); }
			);
			return bins;
		}

		std::vector<ValueType> getValues(const std::vector<const Bin*> bins)
		{
			std::vector<ValueType> values;
			for (auto bin : bins)
			{
				std::transform(
					bin->begin(),
					bin->end(),
					std::back_inserter(values),
					[this](const auto& valueId) { return this->hashtable.getValue(valueId); }
				);
			}
			return values;
		}

		const MultidimHashtable<ValueType, KeyType, numDims>& hashtable;
		std::vector<Hash> currentHashes;
		bool firstQuery;
		std::vector<bool> hashAlreadyVisited;
		size_t numVisited;
		size_t numYields;
	};

	friend class Query;

	MultidimHashtable(
		HashFunction hashFunction,
		const std::array<uint32_t, numDims>& binsPerDim
	) :
		hashFunction(hashFunction),
		sizeOfDim(binsPerDim),
		blocksizeAtDim(getBlocksizeAtDim<numDims>(binsPerDim))
	{
		bins.resize(blocksizeAtDim[0] * sizeOfDim[0]);
	}

	MultidimHashtable(MultidimHashtable<ValueType, KeyType, numDims>&& other) :
		hashFunction(other.hashFunction),
		bins(std::move(other.bins)),
		sizeOfDim(std::move(other.sizeOfDim)),
		blocksizeAtDim(std::move(other.blocksizeAtDim))
	{
	}

	void addValue(const ValueType& value, const std::vector<KeyType>& keys)
	{
		values.push_back(value);
		const auto valueId = values.size() - 1;
		for (const auto& key : keys)
		{
			const auto binIndex = getIndex(getHash(key));
			bins[binIndex].push_back(valueId);
		}
	}

	void writeElementsPerBinToJson(const std::string& filename) const
	{
		std::vector<size_t> elementsPerBin(bins.size());
		std::transform(
			bins.begin(),
			bins.end(),
			elementsPerBin.begin(),
			[](const auto& bin) {return bin.size(); }
		);
		nlohmann::json jsonOutput;
		for (uint32_t dim = 0; dim < numDims; dim++)
		{
			const auto distributionOverDim = accumulateNumElements(elementsPerBin, sizeOfDim, dim);
			auto jsonArray = nlohmann::json::array();
			std::copy(
				distributionOverDim.begin(),
				distributionOverDim.end(),
				std::back_inserter(jsonArray)
			);
			jsonOutput["dimension_" + std::to_string(dim)] = jsonArray;
		}
		const auto numEmpty = static_cast<double>(
			std::count_if(elementsPerBin.begin(), elementsPerBin.end(), [](const auto& numElems) {return numElems == 0; }));
		const auto numBins = static_cast<double>(elementsPerBin.size());
		jsonOutput["total_entries"] = std::accumulate(
			elementsPerBin.begin(),
			elementsPerBin.end(),
			0,
			std::plus<size_t>()
		);
		jsonOutput["proportion_empty"] = numEmpty / numBins;
		jsonOutput["average_per_bin"] = static_cast<double>(
			std::accumulate(elementsPerBin.begin(), elementsPerBin.end(), 0, std::plus<double>())
			) / (numBins - numEmpty);

		std::ofstream file(filename, std::ios_base::trunc);
		file << jsonOutput.dump() << std::endl;
		file.close();
	}

private:
	inline Hash getHash(const KeyType& key) const
	{
		return hashFunction(key);
	}

	inline std::vector<Hash> getHashes(const std::vector<KeyType>& keys) const
	{
		std::vector<Hash> hashes;
		hashes.reserve(keys.size());
		std::transform(
			keys.begin(),
			keys.end(),
			std::back_inserter(hashes),
			[this](const auto& key) { return this->getHash(key); }
		);
		return hashes;
	}

	inline size_t getIndex(const Hash& multiDimIndex) const
	{
		size_t index = 0;
		for (uint32_t dim = 0; dim < numDims; dim++)
		{
			index += blocksizeAtDim[dim] * multiDimIndex[dim];
		}
		return index;
	}

	inline const Bin* getBin(const size_t& binIndex) const
	{
		return &bins[binIndex];
	}

	inline const ValueType& getValue(const size_t& valueId) const
	{
		return values[valueId];
	}

	std::vector<Hash> getGrid(const Hash& center, const uint32_t radius, const uint32_t currentDim) const
	{
		if (currentDim == numDims || radius == 0)
		{
			return { center };
		}
		const auto lowerDimGrid = getGrid(center, radius, currentDim + 1);
		const auto indicesCurrentDim = getIndicesAround(center[currentDim], radius, sizeOfDim[currentDim], isCyclicDim[currentDim]);
		std::vector<Hash> newHashes;
		addCombinations(newHashes, lowerDimGrid, indicesCurrentDim, currentDim);
		return newHashes;
	}

	std::vector<Hash> getGridSurface(const Hash& center, const uint32_t radius, const uint32_t currentDim) const
	{
		if (radius == 0)
		{
			return { center };
		}
		std::vector<Hash> newHashes;
		if (currentDim < numDims - 1)
		{
			const auto lowerDimSurface = getGridSurface(center, radius, currentDim + 1);
			const auto indicesCurrentDim = getIndicesAround(center[currentDim], radius - 1, sizeOfDim[currentDim], isCyclicDim[currentDim]);
			addCombinations(newHashes, lowerDimSurface, indicesCurrentDim, currentDim);
		}
		const auto lowerDimGrid = getGrid(center, radius, currentDim + 1);
		const auto indicesAtOffset = getIndicesAtOffset(center[currentDim], radius, sizeOfDim[currentDim], isCyclicDim[currentDim]);
		addCombinations(newHashes, lowerDimGrid, indicesAtOffset, currentDim);
		return newHashes;
	}

	HashFunction hashFunction;
	std::vector<ValueType> values;
	std::vector<Bin> bins;
	std::vector<bool> binReturned;
	const std::array<uint32_t, numDims> sizeOfDim;
	const std::array<size_t, numDims> blocksizeAtDim;
};
