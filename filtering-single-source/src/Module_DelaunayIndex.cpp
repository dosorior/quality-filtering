#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Fingerprint.h>
#include <Application.h>
#include <DebugDrawing.h>
#include <JsonUtils.h>

namespace py = pybind11;

PYBIND11_MODULE(DelaunayIndex, m) {
	m.doc() = "Builds index from minutia information and allows to query for specific fingerprints"; // optional module docstring

	m.def("setFingerprintImageDir", &debug::setFingerprintImageDir);
	m.def("setDebugOutputDir", &debug::setDebugOutputDir);
	m.def("loadFingerprintsFromJson", &utils::loadFingerprintsFromJson);

	py::class_<Minutia>(m, "Minutia")
		.def(py::init<>())
		.def_readwrite("id", &Minutia::id)
		.def_readwrite("x", &Minutia::x)
		.def_readwrite("y", &Minutia::y)
		.def_readwrite("dir", &Minutia::dir)
		.def_readwrite("qual", &Minutia::qual)
		;

	py::class_<Fingerprint>(m, "Fingerprint")
		.def(py::init<>())
		.def_readwrite("id", &Fingerprint::id)
		.def_readwrite("subjectId", &Fingerprint::subjectId)
		.def_readwrite("name", &Fingerprint::name)
		.def_readwrite("minutiae", &Fingerprint::minutiae)
		;

	py::class_<Application> main_app(m, "DelaunayIndex");
	main_app.def(py::init(&Application::create));
	main_app.def("reportBinDistribution", &Application::reportBinDistribution);
	main_app.def("search", &Application::search);
	main_app.def("searchExhaustive", &Application::searchExhaustive);

	py::enum_<Application::HashingMode>(main_app, "Hashtable")
		.value("HASH_QUAL", Application::HashingMode::HASH_QUAL)
		.value("HASH_GEOM", Application::HashingMode::HASH_GEOM)
		.value("HASH_RFDD", Application::HashingMode::HASH_RFDD)
		.value("HASH_GEOM_RFDD", Application::HashingMode::HASH_GEOM_RFDD)
		.value("HASH_ALL", Application::HashingMode::HASH_ALL)
		.export_values();

	py::enum_<Application::MinutiaSelectionMode>(main_app, "MinutiaSelection")
		.value("KEEP_ALL", Application::MinutiaSelectionMode::KEEP_ALL)
		.value("QUALITY_BEST05", Application::MinutiaSelectionMode::QUALITY_BEST05)
		.value("QUALITY_BEST10", Application::MinutiaSelectionMode::QUALITY_BEST10)
		.value("QUALITY_BEST15", Application::MinutiaSelectionMode::QUALITY_BEST15)
		.value("QUALITY_BEST20", Application::MinutiaSelectionMode::QUALITY_BEST20)
		.value("QUALITY_BEST30", Application::MinutiaSelectionMode::QUALITY_BEST30)
		.value("QUALITY_BEST40", Application::MinutiaSelectionMode::QUALITY_BEST40)
		.value("QUALITY_BEST50", Application::MinutiaSelectionMode::QUALITY_BEST50)
		.value("QUALITY_BEST60", Application::MinutiaSelectionMode::QUALITY_BEST60)
		.export_values();
}

