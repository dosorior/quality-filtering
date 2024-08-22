#include <Triangulation.h>

#include <Delaunator/Delaunator.hpp>

#include <cmath>
#include <unordered_map>

namespace
{

	std::vector<double> getPoints(const std::vector<MinutiaPtr>& minutiae, const size_t skipIndex)
	{
		const auto numPoints = skipIndex < minutiae.size() ? (2 * minutiae.size() - 1) : (2 * minutiae.size());
		std::vector<double> points(numPoints);
		size_t pointIndex = 0;
		for (size_t i = 0; i < minutiae.size(); i++)
		{
			if (i != skipIndex)
			{
				points[2 * pointIndex] = minutiae[i]->x;
				points[2 * pointIndex + 1] = minutiae[i]->y;
				pointIndex++;
			}
		}
		return points;
	}

	std::vector<size_t> getActualMinutiaIndices(const std::vector<size_t>& minutiaIndices, const size_t skipIndex)
	{
		std::vector<size_t> actualIndices;
		actualIndices.reserve(minutiaIndices.size());
		std::transform(
			minutiaIndices.begin(),
			minutiaIndices.end(),
			std::back_inserter(actualIndices),
			[skipIndex](const auto& id) {return (id > skipIndex) ? id + 1 : id; }
		);
		return actualIndices;
	}

	std::vector<MinutiaTriangle> getTriangles(const std::vector<size_t>& minutiaIndices, const std::vector<MinutiaPtr>& minutiae)
	{
		std::vector<MinutiaTriangle> triangles;
		triangles.reserve(minutiaIndices.size() / 3);
		for (size_t i = 0; i < minutiaIndices.size(); i += 3)
		{
			const auto& minu1 = minutiae[minutiaIndices[i]];
			const auto& minu2 = minutiae[minutiaIndices[i + 1]];
			const auto& minu3 = minutiae[minutiaIndices[i + 2]];
			triangles.push_back({ minu1, minu2, minu3, minu1->x, minu1->y, minu2->x, minu2->y, minu3->x, minu3->y });
		}
		return triangles;
	}


	inline uint64_t makeMinutiaTriangleId(const MinutiaTriangle& triangle, const size_t numMinutiae)
	{
		const auto idA = triangle.minuA->id;
		const auto idB = triangle.minuB->id;
		const auto idC = triangle.minuC->id;

		const auto lowest = std::min(idA, std::min(idB, idC));
		const auto highest = std::max(idA, std::max(idB, idC));
		Id mid;
		if (idA == lowest)
		{
			mid = (idB == highest) ? idC : idB;
		}
		else if (idB == lowest)
		{
			mid = (idA == highest) ? idC : idA;
		}
		else
		{
			mid = (idA == highest) ? idB : idA;
		}

		auto id = lowest;
		id = (id * numMinutiae) + mid;
		id = (id * numMinutiae) + highest;
		return id;
	}

} // namespace



std::vector<MinutiaTriangle> extendedDelaunayTriangulation(const std::vector<MinutiaPtr>& minutiae)
{
	if (minutiae.size() < 3)
	{
		return {};
	}
	if (minutiae.size() == 3)
	{
		return getTriangles({ 0, 1, 2 }, minutiae);
	}
	std::vector<MinutiaTriangle> extendedTriangleSet;
	std::unordered_map<Id, bool> triangleAlreadyAdded;
	for (size_t skipIndex = 0; skipIndex <= minutiae.size(); skipIndex++)
	{
		delaunator::Delaunator triangulation(getPoints(minutiae, skipIndex));
		const auto actualIndices = getActualMinutiaIndices(triangulation.triangles, skipIndex);
		for (const auto& triangle : getTriangles(actualIndices, minutiae))
		{
			const auto id = makeMinutiaTriangleId(triangle, minutiae.size());
			const auto pos = triangleAlreadyAdded.find(id);
			if (pos == triangleAlreadyAdded.end())
			{
				extendedTriangleSet.push_back(triangle);
				triangleAlreadyAdded.insert({ id, true });
			}
		}
	}
	return extendedTriangleSet;
}