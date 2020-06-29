#ifndef FILE_PARSER_H
#define FILE_PARSER_H

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <typeinfo>
#include <boost/filesystem.hpp>

namespace knng
{

// Abstract class that models one instance of raw data extracted from a file.
class RawDataInstance
{
  public:
	virtual ~RawDataInstance() { }
};

// Class that is extended by other classes that should be capable of instantiating themselves based on RawDataInstance object and vice versa, to transform themselves into RawDataInstance object.
class RawDataInstanceConverter
{
  public:
	virtual RawDataInstance* to_raw_data_instance(const std::type_info& raw_instance_type) = 0;         // Returns RawDataInstance representation of the object. Return value is an instance of one of the RawDataInstance's sublcasses - parameter raw_instance_type suggests which RawDataInstance's subclass to instantiate.
	virtual void from_raw_data_instance(RawDataInstance* instance) = 0;                                 // Initializes object from the RawDataInstance.
};

// Parses the file and returns data structured in corresponding objects (objects of class derived from RawDataInstance).
class FileParser
{
  protected:
	std::string file_path;
	int label_index;

  public:
  	FileParser(std::string file_path, int label_index = -1);
	virtual RawDataInstance* get_instance() = 0;                          // Returns an data instance from the file. If there is no more data instances in the file, this method returns nullptr.
	virtual RawDataInstance* get_instance_at_pos(std::streampos pos) = 0; // Returns the data instance that starts from the given position. If the position is not valid, this method returns nullptr.
	virtual bool output(std::vector<RawDataInstance*> instances) = 0;     // Outputs RawDataInstances to the file.
	std::string get_file_path();
};

// Parses CSV files into CsvDataInstance<std::string>
class CsvFileParser : public FileParser
{
  protected:
	std::string delimiter = ",";
	std::fstream stream;

  public:
	CsvFileParser(std::string file_path, std::string delimiter, int label_index = -1);
	virtual RawDataInstance* get_instance() override;
	virtual RawDataInstance* get_instance_at_pos(std::streampos pos) override;
	bool output(std::vector<RawDataInstance*> instances) override;
};

// Parses CSV files into CsvDataInstance<double>
class NumericCsvFileParser : public CsvFileParser
{
  public:
  	NumericCsvFileParser(std::string file_path, std::string delimiter, int label_index = -1) : CsvFileParser(file_path, delimiter, label_index) { }
	RawDataInstance* get_instance_at_pos(std::streampos pos) override;
};

// Parses CSV files into CsvDataInstance<int>
class IntegerCsvFileParser : public CsvFileParser
{
  public:
  	IntegerCsvFileParser(std::string file_path, std::string delimiter, int label_index) : CsvFileParser(file_path, delimiter, label_index) { }
	RawDataInstance* get_instance_at_pos(std::streampos pos) override;
};

// Model class that stores values extracted from one line of CSV file.
template <typename T>
class CsvDataInstance : public RawDataInstance
{
  protected:
	std::vector<T> data;
	std::string label;
  
  public:
	CsvDataInstance(std::vector<T> data, std::string label = "");
	~CsvDataInstance() override = default;
	int size();
	T& operator[](const int index);
	std::string to_string(std::string delimiter, int label_index = -1);
	std::string get_label();
};

// FileParser class implementation.

FileParser::FileParser(std::string file_path, int label_index) : file_path(file_path), label_index(label_index) { }

std::string FileParser::get_file_path()
{
	return file_path;
}

// CsvFileParser class implementation.

CsvFileParser::CsvFileParser(std::string file_path, std::string delimiter, int label_index) : FileParser(file_path, label_index), delimiter(delimiter) { }

RawDataInstance* CsvFileParser::get_instance()
{
	return get_instance_at_pos(stream.is_open() ? stream.tellg() : (std::streampos)0);
}

RawDataInstance* CsvFileParser::get_instance_at_pos(std::streampos pos)
{
	if (!stream.is_open())
	{
		stream.open(file_path, std::ifstream::in|std::ifstream::binary);
		if (!stream.is_open()) 
			return nullptr;
	}

	std::vector<std::string> instance_data;
	std::string label = "";

	stream.clear();
	stream.seekg(pos);

	std::string line;
	if (std::getline(stream, line))
	{
		size_t pos = 0;
		int index = 0;
		while ((pos = line.find(delimiter)) != std::string::npos) {
			std::string str = line.substr(0, pos);
			if (index != label_index)
				instance_data.push_back(str);
			else
				label = str;			
			line.erase(0, pos + delimiter.length());
			index++;
		}
		if (index != label_index)
			instance_data.push_back(line);
		else
			label = line;
	}
	else
	{
		return nullptr;
	}
	return new CsvDataInstance<std::string>(instance_data, label);
}

bool CsvFileParser::output(std::vector<RawDataInstance*> instances)
{
	if (!stream.is_open())
	{
		boost::filesystem::path p(file_path);
		if (p.parent_path() != "")
			boost::filesystem::create_directories(p.parent_path());
		stream.open(file_path, std::fstream::out);
		if (!stream.is_open()) 
			return false;
	}
	std::vector<RawDataInstance*>::iterator it = instances.begin();
	while (it != instances.end())
	{
		CsvDataInstance<std::string>* instance_data_str = dynamic_cast<CsvDataInstance<std::string>*>(*it);
		CsvDataInstance<double>* instance_data_double = dynamic_cast<CsvDataInstance<double>*>(*it);
		CsvDataInstance<int>* instance_data_int = dynamic_cast<CsvDataInstance<int>*>(*it);
		if (instance_data_str != nullptr)
			stream << instance_data_str->to_string(delimiter, label_index) << std::endl;
		else if (instance_data_double != nullptr)
			stream << instance_data_double->to_string(delimiter, label_index) << std::endl;
		else if (instance_data_int != nullptr)
			stream << instance_data_int->to_string(delimiter, label_index) << std::endl;
		else
			break;		
		it++;
	}

	stream.close();

	return it == instances.end();
}

// NumericCsvFileParser class implementation.

RawDataInstance* NumericCsvFileParser::get_instance_at_pos(std::streampos pos)
{
	CsvDataInstance<std::string>* instance_data_str = (CsvDataInstance<std::string>*)CsvFileParser::get_instance_at_pos(pos);
	
	if (instance_data_str == nullptr) return nullptr;

	std::vector<double> instance_data;
	std::string label = instance_data_str->get_label();

	for (int i = 0; i < instance_data_str->size(); i++)
	{
		std::string s = (*instance_data_str)[i];
		instance_data.push_back(stod(s));
	}

	delete instance_data_str;

	return new CsvDataInstance<double>(instance_data, label);
}

// IntegerCsvFileParser class implementation.

RawDataInstance* IntegerCsvFileParser::get_instance_at_pos(std::streampos pos)
{
	CsvDataInstance<std::string>* instance_data_str = (CsvDataInstance<std::string>*)CsvFileParser::get_instance_at_pos(pos);
	
	if (instance_data_str == nullptr) return nullptr;

	std::vector<int> instance_data;
	std::string label = instance_data_str->get_label();

	for (int i = 0; i < instance_data_str->size(); i++)
	{
		std::string s = (*instance_data_str)[i];
		instance_data.push_back(stoi(s));
	}

	delete instance_data_str;

	return new CsvDataInstance<int>(instance_data, label);
}

// CsvDataInstance class implementation.

template <typename T>
CsvDataInstance<T>::CsvDataInstance(std::vector<T> data, std::string label)
{
	this->data = data;
	this->label = label;
}
	
template <typename T>
int CsvDataInstance<T>::size()
{
	return data.size();
}

template <typename T>
T& CsvDataInstance<T>::operator[](const int index)
{
	return data[index];
}

template <typename T>
std::string CsvDataInstance<T>::to_string(std::string delimiter, int label_index)
{
	if (data.size() == 0) return "";
	
	std::stringstream sout;

	int index = 0;
	if (label_index == index)
	{
		sout << label;
	}
	else
	{
		sout << data[index];
		index++;
	}

	for (int i = index; i < data.size(); i++)
	{
		if (label_index == i)
			sout << delimiter << label;
		sout << delimiter << data[i];
	}

	return sout.str();
}

template <typename T>
std::string CsvDataInstance<T>::get_label()
{
	return label;
}

} // namespace knng
#endif