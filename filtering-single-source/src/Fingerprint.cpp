#include <Fingerprint.h>

#include <cmath>

#include <iostream>


namespace
{

	constexpr double pi = 3.14159;
	constexpr double two_pi = 2 * pi;

	// MAXSIZE is the maximum distance between two points i.e. the diagonal of the image.
	// Here are the values for FVC2006 database:
	constexpr double DB1_MAXSIZE = 283.0; // Images are 200 x 200
										  // Originally were 96 x 96 but had to be resized for CoarseNet
	constexpr double DB2_MAXSIZE = 690.0; // Images are 400 x 560
	constexpr double DB3_MAXSIZE = 641.0; // Images are 400 x 500
	constexpr double DB4_MAXSIZE = 481.0; // Images are 288 x 384
	// Change this variable to match the database used:
	constexpr double MAXSIZE = DB2_MAXSIZE;

	using Vec = std::pair<double, double>;

	inline double length(const Vec& v)
	{
		return std::sqrt(v.first * v.first + v.second * v.second);
	}

	inline double angleBetweenVectors(const Vec& u, const Vec& v)
	{
		const auto dot = u.first * v.first + u.second * v.second;
		return std::acos(dot / (length(u) * length(v)));
	}

	inline double angleDifference(const double& a, const double& b)
	{
		const auto absdiff = std::abs(a - b);
		return std::min(std::min(absdiff, std::abs(absdiff - pi)), std::abs(absdiff - two_pi));
	}

	inline Vec minus(const Vec& v)
	{
		return { -v.first, -v.second };
	}

	inline double getLongestSide(const MinutiaTriangle& triangle, const double maxSize)
	{
		const auto vecAB = std::make_pair(triangle.Bx - triangle.Ax, triangle.By - triangle.Ay);
		const auto vecBC = std::make_pair(triangle.Cx - triangle.Bx, triangle.Cy - triangle.By);
		const auto vecCA = std::make_pair(triangle.Ax - triangle.Cx, triangle.Ay - triangle.Cy);
		return std::max(length(vecAB), std::max(length(vecBC), length(vecCA))) / maxSize;
	}

	inline double getTriangleQuality(const MinutiaTriangle& triangle)
	{
		const auto avgQual = (triangle.minuA->qual + triangle.minuB->qual + triangle.minuC->qual) / 3.0;
		return std::min(std::max(avgQual, 0.001), 0.999);
	}

	inline std::pair<double, double> getTwoLargestAngles(const MinutiaTriangle& triangle)
	{
		const auto vecAB = std::make_pair(triangle.Bx - triangle.Ax, triangle.By - triangle.Ay);
		const auto vecBC = std::make_pair(triangle.Cx - triangle.Bx, triangle.Cy - triangle.By);
		const auto vecCA = std::make_pair(triangle.Ax - triangle.Cx, triangle.Ay - triangle.Cy);

		const auto angleA = angleBetweenVectors(vecAB, minus(vecCA));
		const auto angleB = angleBetweenVectors(vecBC, minus(vecAB));
		const auto angleC = angleBetweenVectors(vecCA, minus(vecBC));

		if (angleA <= 0 || angleA >= pi || angleB <= 0 || angleB >= pi || angleC <= 0 || angleC >= pi)
		{
			return { pi, 0.0 };
		}

		const auto notSmallestAB = std::max(angleA, angleB);
		const auto notSmallestBC = std::max(angleB, angleC);
		const auto notSmallestCA = std::max(angleC, angleA);

		auto largest = std::max(notSmallestAB, std::max(notSmallestBC, notSmallestCA));
		auto secondLargest = std::min(notSmallestAB, std::min(notSmallestBC, notSmallestCA));

		return { largest, secondLargest };
	}

	inline std::pair<double, double> getRidgeFlowDifferences(const MinutiaTriangle& triangle)
	{
		const auto angleA = angleDifference(triangle.minuA->dir, triangle.minuB->dir);
		const auto angleB = angleDifference(triangle.minuB->dir, triangle.minuC->dir);
		const auto angleC = angleDifference(triangle.minuC->dir, triangle.minuA->dir);

		const auto notSmallestAB = std::max(angleA, angleB);
		const auto notSmallestBC = std::max(angleB, angleC);
		const auto notSmallestCA = std::max(angleC, angleA);

		auto largest = std::max(notSmallestAB, std::max(notSmallestBC, notSmallestCA));
		auto secondLargest = std::min(notSmallestAB, std::min(notSmallestBC, notSmallestCA));
		return { largest, secondLargest };
	}

}

TriangleFeatures getTriangleFeatures(const MinutiaTriangle& triangle)
{
	const auto angles = getTwoLargestAngles(triangle);
	const auto side_c = getLongestSide(triangle, MAXSIZE);
	const auto ridgeFlowDiffs = getRidgeFlowDifferences(triangle);
	const auto qual = getTriangleQuality(triangle);
	return TriangleFeatures{ angles.first, angles.second, side_c, 0.0, 0.0, ridgeFlowDiffs.first, ridgeFlowDiffs.second, qual };
}