#ifndef DATASET_H
#define DATASET_H

#include <fstream>
#include <sstream>
#include <type_traits>
#include <vector>
#include "../common/string_converter.h"
#include "../common/identifiable.h"
#include "../common/random.h"
#include "../datasets/file_parser.h"
#include "../exceptions/not_implemented_exception.h"

namespace knng
{

// Model class for dataset of objects of type InstanceType.
template <typename InstanceType>
class Dataset : public StringConverter
{
	static_assert(std::is_base_of<Identifiable, InstanceType>::value, "InstanceType must implement Identifiable class.");
	static_assert(std::is_base_of<StringConverter, InstanceType>::value, "InstanceType must implement StringConverter class.");
	static_assert(std::is_base_of<RawDataInstanceConverter, InstanceType>::value, "InstanceType must implement RawDataInstanceConverter class.");
  	static_assert(std::is_move_constructible<InstanceType>::value, "InstanceType has to have copy constructor.");
  
  protected:
	std::vector<InstanceType*> data;
	std::string file_path;

  public:
	Dataset();
	Dataset(FileParser& parser);
	Dataset(const std::vector<int> &ids);
	Dataset(const std::vector<InstanceType> &instances);
	virtual ~Dataset();
	static Dataset<InstanceType>* create_dataset(int size, RandomObjectGenerator<InstanceType> &rnd);
	virtual void add_instance(const InstanceType &instance);
	std::vector<InstanceType*>& get_instances(); // Returns all instances of the dataset.
	std::string get_file_path();

	// StringConverter override
	virtual std::string to_string(bool brief = true) override;
};

// Dataset<InstanceType> class implementation.

template <typename InstanceType>
Dataset<InstanceType>::Dataset() { }

template <typename InstanceType>
Dataset<InstanceType>::Dataset(FileParser& parser)
{
	file_path = parser.get_file_path();
	RawDataInstance* raw_instance = parser.get_instance();
	int id = 0;
	while (raw_instance != nullptr)
	{
		InstanceType* instance = new InstanceType;
		((Identifiable*)instance)->set_id(id);
		((RawDataInstanceConverter*)instance)->from_raw_data_instance(raw_instance);
		data.push_back(instance);
		id++;
		delete raw_instance;
		raw_instance = parser.get_instance();
	}
}

template <typename InstanceType>
Dataset<InstanceType>::Dataset(const std::vector<int> &ids)
{
	for (auto id: ids)
	{
		InstanceType* instance = new InstanceType;
		((Identifiable*)instance)->set_id(id);
		data.push_back(instance);
	}
}

template <typename InstanceType>
Dataset<InstanceType>::Dataset(const std::vector<InstanceType> &instances)
{
	typename std::vector<InstanceType>::const_iterator instances_it = instances.begin();
	while (instances_it != instances.end())
	{
		InstanceType* instance = new InstanceType(*instances_it);
		data.push_back(instance);
		instances_it++;
	}
}

template <typename InstanceType>
Dataset<InstanceType>::~Dataset()
{
	typename std::vector<InstanceType*>::iterator data_it = data.begin();
	while (data_it != data.end())
	{
		InstanceType* instance = *data_it;
		data_it = data.erase(data_it);
		delete instance;
	}
}

template <typename InstanceType>
Dataset<InstanceType>* Dataset<InstanceType>::create_dataset(int size, RandomObjectGenerator<InstanceType> &rnd)
{
	Dataset<InstanceType>* dataset = new Dataset<InstanceType>;
	for (int i = 0; i < size; i++)
	{
		InstanceType* rnd_instance = rnd.create_random_object();
		dataset->add_instance(*rnd_instance);
		delete rnd_instance;
	}
	return dataset;
}

template <typename InstanceType>
void Dataset<InstanceType>::add_instance(const InstanceType &instance)
{
	InstanceType* new_instance = new InstanceType(instance);
	((Identifiable*)new_instance)->set_id(data.size() + 1);
	data.push_back(new_instance);
}

template <typename InstanceType>
std::vector<InstanceType*>& Dataset<InstanceType>::get_instances()
{
	return data;
}

template <typename InstanceType>
std::string Dataset<InstanceType>::get_file_path()
{
	return file_path;
}

template <typename InstanceType>
std::string Dataset<InstanceType>::to_string(bool brief)
{
	std::stringstream sout;
	typename std::vector<InstanceType*>::iterator data_it = data.begin();
	while (data_it != data.end())
	{
		sout << ((StringConverter*)*data_it)->to_string(brief) << std::endl;
		data_it++;
	}
	return sout.str();
}

} // namespace knng
#endif