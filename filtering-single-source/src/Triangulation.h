#pragma once

#include <Fingerprint.h>

#include <vector>

std::vector<MinutiaTriangle> extendedDelaunayTriangulation(const std::vector<MinutiaPtr>& minutiae);
