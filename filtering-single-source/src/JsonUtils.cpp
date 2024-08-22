#include "JsonUtils.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>

using namespace nlohmann;

namespace
{

	class FingerprintFileException : std::exception
	{
	public:
		FingerprintFileException(
			const std::string& message
		) : message(message) {};

		const char* what()
		{
			return message.c_str();
		}

	private:
		const std::string message;
	};

	const json& getElementOrThrow(const json& parent, const std::string& key)
	{
		try
		{
			return parent.at(key);
		}
		catch (std::exception& e)
		{
			throw FingerprintFileException("Could not access element: " + key + "\n");
		}
	}

	Id parseId(const json& id)
	{
		try
		{
			return id.get<Id>();
		}
		catch (std::exception& exception)
		{
			throw FingerprintFileException(std::string("\"id\" must be non-negative integer: ") + exception.what());
		}
	}

	std::string parseName(const json& name)
	{
		try
		{
			return name.get<std::string>();
		}
		catch (std::exception& exception)
		{
			throw FingerprintFileException(std::string("\"name\" must be a string: ") + exception.what());
		}
	}

	double parseMinutiaFeature(const json& feature, const std::string& name)
	{
		try
		{
			return feature.get<double>();
		}
		catch (std::exception& exception)
		{
			throw FingerprintFileException("minutia feature \"" + name + "\" must be a real number: " + exception.what());
		}
	}

	MinutiaPtr parseMinutia(const json& minu)
	{
		if (!minu.is_object())
		{
			throw FingerprintFileException("Each minutia must be a json object");
		}
		const auto id = parseId(getElementOrThrow(minu, "id"));
		const auto x = parseMinutiaFeature(getElementOrThrow(minu, "x"), "x");
		const auto y = parseMinutiaFeature(getElementOrThrow(minu, "y"), "y");
		const auto dir = parseMinutiaFeature(getElementOrThrow(minu, "dir"), "dir");
		const auto qual = parseMinutiaFeature(getElementOrThrow(minu, "qual"), "qual");
		return MinutiaPtr(new Minutia{ id, x, y, dir, qual });
	}

	std::vector<MinutiaPtr> parseMinutiaArray(const json& minuarray)
	{
		if (!minuarray.is_array())
		{
			throw FingerprintFileException(std::string("\"minutiae\" must be a json array"));
		}
		std::vector<MinutiaPtr> minutiae;
		for (const auto& minu : minuarray)
		{
			minutiae.push_back(parseMinutia(minu));
		}
		return minutiae;
	}

	Id getSubjectId(const std::string& name)
	{
		return std::stoi(name.substr(0, name.find("_")));
	}

	Fingerprint parseFingerprint(const json& fp, const size_t maxIdAllowed)
	{
		if (!fp.is_object())
		{
			throw FingerprintFileException("Each fingerprint must be a json object");
		}
		const auto id = parseId(getElementOrThrow(fp, "id"));
		if (id > maxIdAllowed)
		{
			throw FingerprintFileException("Fingerprint id must be in range [0, ..., numFingerprints]");
		}
		const auto name = parseName(getElementOrThrow(fp, "name"));
		const auto minutiae = parseMinutiaArray(getElementOrThrow(fp, "minutiae"));
		return Fingerprint{ id, getSubjectId(name), name, minutiae };
	}

	std::vector<Fingerprint> parseFingerprintArray(const json& fparray)
	{
		if (!fparray.is_array())
		{
			throw FingerprintFileException("fingerprints must be stored in a json array");
		}
		std::vector<Fingerprint> fingerprints;
		size_t pos = 0;
		size_t maxIndexAllowed = fparray.size() - 1;
		for (const auto& fp : fparray)
		{
			try
			{
				fingerprints.push_back(parseFingerprint(fp, maxIndexAllowed));
				pos++;
			}
			catch (FingerprintFileException& exception)
			{
				throw FingerprintFileException("Fingerprint at position " + std::to_string(pos) + " - " + exception.what());
			}
		}
		return fingerprints;
	}

} // namespace

namespace utils
{
	std::vector<Fingerprint> loadFingerprintsFromJson(const std::string& filename)
	{
		try
		{
			std::ifstream ifs(filename);
			const auto root = json::parse(ifs);
			ifs.close();
			return parseFingerprintArray(getElementOrThrow(root, "fingerprints"));
		}
		catch (FingerprintFileException& exception)
		{
			const auto message = "Could not read fingerprint data from " + filename + ":\n    " + exception.what();
			std::cout << "\n\nERROR: " + message;
			throw FingerprintFileException(message);
		}
	}
}
