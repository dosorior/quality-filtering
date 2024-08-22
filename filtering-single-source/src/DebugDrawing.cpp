#include <DebugDrawing.h>

#include <opencv2/opencv.hpp>


namespace
{

	void drawTriangleSides(cv::Mat& image, const MinutiaTriangle& triangle)
	{
		constexpr unsigned int LINE_THICKNESS = 1;
		const auto LINE_COLOR = cv::Scalar(255, 0, 0);
		const auto A = cv::Point(triangle.Ax, triangle.Ay);
		const auto B = cv::Point(triangle.Bx, triangle.By);
		const auto C = cv::Point(triangle.Cx, triangle.Cy);
		cv::line(image, A, B, LINE_COLOR, LINE_THICKNESS);
		cv::line(image, B, C, LINE_COLOR, LINE_THICKNESS);
		cv::line(image, C, A, LINE_COLOR, LINE_THICKNESS);
	}

	void drawTriangleCorners(cv::Mat& image, const MinutiaTriangle& triangle)
	{
		constexpr unsigned int CIRCLE_RADIUS = 2;
		constexpr unsigned int CIRCLE_THICKNESS = 1;
		const auto CIRCLE_COLOR = cv::Scalar(0, 255, 0);
		cv::circle(image, cv::Point(triangle.Ax, triangle.Ay), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS);
		cv::circle(image, cv::Point(triangle.Bx, triangle.By), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS);
		cv::circle(image, cv::Point(triangle.Cx, triangle.Cy), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS);
	}

	void drawTriangulation(cv::Mat& image, const std::vector<MinutiaTriangle>& triangles)
	{
		for (const auto& triangle : triangles)
		{
			drawTriangleSides(image, triangle);
			drawTriangleCorners(image, triangle);
		}
	}

}

namespace debug
{

	std::string FINGERPRINT_IMAGE_DIR;
	std::string DEBUG_OUT_DIR;

	void setFingerprintImageDir(const std::string& directory)
	{
		FINGERPRINT_IMAGE_DIR = directory;
	}

	void setDebugOutputDir(const std::string& directory)
	{
		DEBUG_OUT_DIR = directory;
	}

	void showTriangulation(const std::string& fingerprintName, const std::vector<MinutiaTriangle>& triangles)
	{
		const auto filename = FINGERPRINT_IMAGE_DIR + fingerprintName + ".bmp";
		std::cout << "Showing triangulation of " << filename << std::endl;
		auto image = cv::imread(filename);
		drawTriangulation(image, triangles);
		const auto winname = "Delaunay triangulation of " + filename;
		cv::imshow(winname, image);
		cv::waitKey(0);
		cv::destroyWindow(winname);
	}

	void saveTriangulation(
		const std::string& fingerprintName,
		const std::vector<MinutiaTriangle>& triangles,
		const std::string& outnamePrefix)
	{
		const auto filename = FINGERPRINT_IMAGE_DIR + fingerprintName + ".bmp";
		const auto outname = DEBUG_OUT_DIR + outnamePrefix + fingerprintName + ".bmp";
		std::cout << "Saving triangulation of " << fingerprintName << " as " << outname << std::endl;
		auto image = cv::imread(filename);
		drawTriangulation(image, triangles);
		cv::imwrite(outname, image);
	}

} // namespace debug