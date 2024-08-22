#pragma once

#include <vector>
#include <string>
#include <memory>

#define LOG(somestring) (std::cout << somestring << std::endl)

using Id = uint64_t;

struct __declspec(dllexport) Minutia {
	Id id;
	double x;
	double y;
	double dir; // as angle to x axis
	double qual;
};

using MinutiaPtr = std::shared_ptr<Minutia>;

struct __declspec(dllexport) Fingerprint
{
	Id id;
	Id subjectId;
	std::string name;
	std::vector<MinutiaPtr> minutiae;
};

struct MinutiaTriangle
{
	const MinutiaPtr minuA;
	const MinutiaPtr minuB;
	const MinutiaPtr minuC;
	double Ax;
	double Ay;
	double Bx;
	double By;
	double Cx;
	double Cy;
};

struct TriangleFeatures
{
	double angle_alpha;
	double angle_beta;
	double side_c;
	double center_x;
	double center_y;
	double flow_diff1;
	double flow_diff2;
	double qual;
};


TriangleFeatures getTriangleFeatures(const MinutiaTriangle& triangle);