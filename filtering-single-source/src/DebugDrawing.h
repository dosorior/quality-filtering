#pragma once

#include <Fingerprint.h>

#include <string>

namespace debug
{

	void setFingerprintImageDir(const std::string& directory);

	void setDebugOutputDir(const std::string& directory);

	void showTriangulation(const std::string& fingerprintName, const std::vector<MinutiaTriangle>& triangles);

	void saveTriangulation(
		const std::string& fingerprintName,
		const std::vector<MinutiaTriangle>& triangles,
		const std::string& outnamePrefix = "_triangulation");

} // namespace debug