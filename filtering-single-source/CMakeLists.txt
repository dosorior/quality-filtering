cmake_minimum_required(VERSION 3.4...3.18)
project(DelaunayIndex LANGUAGES CXX)

add_subdirectory(libs/pybind11)
pybind11_add_module(DelaunayIndex SHARED
	src/Module_DelaunayIndex.cpp
	src/Application.cpp
	src/Fingerprint.cpp
	src/JsonUtils.cpp
	src/Triangulation.cpp
	src/DebugDrawing.cpp)


target_include_directories(DelaunayIndex PRIVATE src src/nlohmann src/Delaunator)

# Change this to the location of your local opencv installations build/ directory
set(OpenCV_DIR "C:/Program Files (x86)/opencv/build/")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
target_link_libraries(DelaunayIndex PRIVATE ${OpenCV_LIBS})
