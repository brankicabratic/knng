#ifndef TIME_SERIES_BUFFER_H
#define TIME_SERIES_BUFFER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include "dataset.h"
#include "file_parser.h"
#include "../data_objects/time_series.h"

namespace knng
{

// Model class for buffered dataset of TimeSeries.
// This class is used when time series should be read in
// batches with the given sliding window.
class TimeSeriesBuffer : public Dataset<TimeSeries>
{
	std::unordered_map<TimeSeries*,RawDataInstance*> raw_instances; // Cached values of each time series' raw instance.
	int sliding_window;

  public:
	TimeSeriesBuffer(FileParser& parser, int sliding_window);
	virtual ~TimeSeriesBuffer() override;
	// Methods for loading new batches of time series. Map loads_cnts (if not null), 
	// is filled with the sizes of actual batches that are loaded from the file. If
	// there were enough new values in thefile, size of a batch will be equal to the cnt,
	// otherwise it will be lower, or even 0 if there were no new values in the file.
	void load(int cnt, std::unordered_map<TimeSeries*, int>* loads_cnts = nullptr); // Reads cnt values of all dataset's time series.
	void load(std::vector<TimeSeries*> points, int cnt, std::unordered_map<TimeSeries*, int>* loads_cnts = nullptr); // Reads cnt values only of time series given in vector points.
	void load(std::vector<TimeSeries*> points, std::vector<int> &cnts, std::unordered_map<TimeSeries*, int>* loads_cnts = nullptr); // Reads cnts[i] values of points[i] time series.
	void reset();
	int get_sliding_window();
};

TimeSeriesBuffer::TimeSeriesBuffer(FileParser& parser, int sliding_window) : sliding_window(sliding_window)
{
	this->file_path = parser.get_file_path();
	RawDataInstance* raw_instance = parser.get_instance();
	int id = 0;
	while (raw_instance != nullptr)
	{
		TimeSeries* instance = new TimeSeries(sliding_window);
		instance->set_id(id);
		instance->from_raw_data_instance(raw_instance);
		data.push_back(instance);
		raw_instances.insert({instance, raw_instance});
		id++;
		raw_instance = parser.get_instance();
	}
}

TimeSeriesBuffer::~TimeSeriesBuffer()
{
	for (std::pair<TimeSeries*, RawDataInstance*> pair: raw_instances)
		delete pair.second;
}

void TimeSeriesBuffer::load(int cnt, std::unordered_map<TimeSeries*, int>* loads_cnts)
{
	for (std::pair<TimeSeries*, RawDataInstance*> pair: raw_instances)
	{
		int loads = pair.first->read(pair.second, cnt);
		if (loads_cnts != nullptr)
			(*loads_cnts)[pair.first] = loads;
	}
}

void TimeSeriesBuffer::load(std::vector<TimeSeries*> points, int cnt, std::unordered_map<TimeSeries*, int>* loads_cnts)
{
	std::vector<int> cnts(points.size(), cnt);
	load(points, cnts, loads_cnts);
}

void TimeSeriesBuffer::load(std::vector<TimeSeries*> points, std::vector<int> &cnts, std::unordered_map<TimeSeries*, int>* loads_cnts)
{
	if (points.size() != cnts.size())
		throw std::invalid_argument("TimeSeriesBuffer::load - Vectors points and cnts must be of same size.");
	
	std::vector<TimeSeries*>::iterator points_it = points.begin();
	std::vector<int>::iterator cnts_it = cnts.begin();
	while (points_it != points.end())
	{
		std::unordered_map<TimeSeries*,RawDataInstance*>::iterator raw_instance_it = raw_instances.find(*points_it);
		int loads = raw_instance_it->first->read(raw_instance_it->second, *cnts_it);
		if (loads_cnts != nullptr)
			(*loads_cnts)[*points_it] = loads;
		points_it++;
		cnts_it++;
	}
}

void TimeSeriesBuffer::reset()
{
	for (const std::pair<TimeSeries*,RawDataInstance*> &pair: raw_instances)
		pair.first->from_raw_data_instance(pair.second);
}

int TimeSeriesBuffer::get_sliding_window()
{
	return sliding_window;
}

} // namespace knng
#endif