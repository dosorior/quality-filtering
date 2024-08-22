#pragma once

#include <Fingerprint.h>
#include <MultidimHashtable.hpp>

#include <array>

namespace
{
	constexpr double pi = 3.142; // Purposefully too high s.t. the hashing function outputs (numBins - 1) if the input is exactly pi
	constexpr double pi_half = pi / 2;
	constexpr double pi_thirds = pi / 3;
	constexpr double pi_two_thirds = 2 * pi_thirds;

	inline uint32_t hashAlphaUniform(const double val, const double numBins) // Always in [pi / 3, pi]
	{
		return static_cast<uint32_t>((val - pi_thirds) / pi_two_thirds * numBins);
	}

	inline uint32_t hashBetaUniform(const double val, const double numBins) // Always in [0, pi / 2]
	{
		return static_cast<uint32_t>(val / pi_half * numBins);
	}

	inline uint32_t hashAngle90Uniform(const double val, const double numBins)
	{
		return static_cast<uint32_t>(val / pi_half * numBins);
	}

	inline uint32_t hashZeroToOneUniform(const double val, const double numBins)
	{
		return static_cast<uint32_t>(std::min(val * numBins, numBins - 1.001));
	}

}

namespace HashtableVariants
{
	using Hashtable1D = MultidimHashtable<Id, TriangleFeatures, 1>;
	using HashValue1D = std::array<uint32_t, 1>;
	using BinsPerDim1D = std::array<uint32_t, 1>;

	Hashtable1D hashQuality(const uint32_t binsQuality)
	{
		const auto binsQualityf = static_cast<double>(binsQuality);
		const auto hashFun = [binsQualityf](const TriangleFeatures& triangle) {
			return HashValue1D({ hashZeroToOneUniform(triangle.qual, binsQualityf) }); };
		return Hashtable1D(
			hashFun,
			BinsPerDim1D({ binsQuality }));
	}

	using Hashtable2D = MultidimHashtable<Id, TriangleFeatures, 2>;
	using HashValue2D = std::array<uint32_t, 2>;
	using BinsPerDim2D = std::array<uint32_t, 2>;

	Hashtable2D hashRFDD(const uint32_t binsRFDD)
	{
		const auto binsRFDDf = static_cast<double>(binsRFDD);
		const auto hashFun = [binsRFDDf](const TriangleFeatures& triangle) {
			return HashValue2D({
				hashAngle90Uniform(triangle.flow_diff1, binsRFDDf),
				hashAngle90Uniform(triangle.flow_diff2, binsRFDDf)
				}); };
		return Hashtable2D(
			hashFun,
			BinsPerDim2D({ binsRFDD, binsRFDD }));
	}

	using Hashtable3D = MultidimHashtable<Id, TriangleFeatures, 3>;
	using HashValue3D = std::array<uint32_t, 3>;
	using BinsPerDim3D = std::array<uint32_t, 3>;

	Hashtable3D hashGeom(const uint32_t binsAlpha, const uint32_t binsBeta, const uint32_t binsC)
	{
		const auto binsAlphaf = static_cast<double>(binsAlpha);
		const auto binsBetaf = static_cast<double>(binsBeta);
		const auto binsCf = static_cast<double>(binsC);
		const auto hashFun = [binsAlphaf, binsBetaf, binsCf](const TriangleFeatures& triangle) {
			return HashValue3D({
				hashAlphaUniform(triangle.angle_alpha, binsAlphaf),
				hashBetaUniform(triangle.angle_beta, binsBetaf),
				hashAngle90Uniform(triangle.flow_diff1, binsCf)
				}); };
		return Hashtable3D(
			hashFun,
			BinsPerDim3D({ binsAlpha, binsBeta, binsC }));
	}

	using Hashtable5D = MultidimHashtable<Id, TriangleFeatures, 5>;
	using HashValue5D = std::array<uint32_t, 5>;
	using BinsPerDim5D = std::array<uint32_t, 5>;

	Hashtable5D hashGeomRFDD(const uint32_t binsAlpha, const uint32_t binsBeta, const uint32_t binsC, const uint32_t binsRFDD)
	{
		const auto binsAlphaf = static_cast<double>(binsAlpha);
		const auto binsBetaf = static_cast<double>(binsBeta);
		const auto binsCf = static_cast<double>(binsC);
		const auto binsRFDDf = static_cast<double>(binsRFDD);
		const auto hashFun = [binsAlphaf, binsBetaf, binsRFDDf, binsCf](const TriangleFeatures& triangle) {
			return HashValue5D({
				hashAlphaUniform(triangle.angle_alpha, binsAlphaf),
				hashBetaUniform(triangle.angle_beta, binsBetaf),
				hashZeroToOneUniform(triangle.side_c, binsCf),
				hashAngle90Uniform(triangle.flow_diff1, binsRFDDf),
				hashAngle90Uniform(triangle.flow_diff2, binsRFDDf)
				}); };
		return Hashtable5D(
			hashFun,
			BinsPerDim5D({ binsAlpha, binsBeta, binsC, binsRFDD, binsRFDD }));
	}

	using Hashtable6D = MultidimHashtable<Id, TriangleFeatures, 6>;
	using HashValue6D = std::array<uint32_t, 6>;
	using BinsPerDim6D = std::array<uint32_t, 6>;

	Hashtable6D hashAll(const uint32_t binsAlpha, const uint32_t binsBeta, const uint32_t binsC, const uint32_t binsRFDD, const uint32_t binsQuality)
	{
		const auto binsAlphaf = static_cast<double>(binsAlpha);
		const auto binsBetaf = static_cast<double>(binsBeta);
		const auto binsCf = static_cast<double>(binsC);
		const auto binsRFDDf = static_cast<double>(binsRFDD);
		const auto binsQualityf = static_cast<double>(binsQuality);
		const auto hashFun = [binsAlphaf, binsBetaf, binsCf, binsRFDDf, binsQualityf](const TriangleFeatures& triangle) {
			return HashValue6D({
				hashAlphaUniform(triangle.angle_alpha, binsAlphaf),
				hashBetaUniform(triangle.angle_beta, binsBetaf),
				hashZeroToOneUniform(triangle.qual, binsCf),
				hashAngle90Uniform(triangle.flow_diff1, binsRFDDf),
				hashAngle90Uniform(triangle.flow_diff2, binsRFDDf),
				hashZeroToOneUniform(triangle.qual, binsQualityf)
				}); };
		return Hashtable6D(
			hashFun,
			BinsPerDim6D({ binsAlpha, binsBeta, binsC, binsRFDD, binsRFDD, binsQuality }));
	}
} // namespace HashtableVariants