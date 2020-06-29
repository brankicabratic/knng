#ifndef TIME_SERIES_H
#define TIME_SERIES_H

#include <algorithm>
#include <vector>
#include "point.h"

namespace knng
{

// Model class for time series.
class TimeSeries : public Point
{
	// Sliding window of the time series. Maximal dimensionality of time series is equal to this value.
	// When new values arrive in time series, the oldest values are removed in order to preserve the
	// size of thesliding window. If this value is -1, then time series does not have dimensionality limit.
	int sliding_window;
	int next_index = 0;

  public:
	TimeSeries(int sliding_window = -1);
	void set_sliding_window(int sliding_window);       // Sets the sliding window size.
	int read(RawDataInstance* instance, int cnt);      // Reads cnt new values from raw instance starting from the value last_read+1, and adds them to the end of time series.

	// RawDataInstanceConverter override
	virtual void from_raw_data_instance(RawDataInstance* instance) override;

  protected:
	void push_back(const std::vector<double> &values); // Adds values to the end of time series.
};

// L2 (Euclidean) distance defined on class TimeSeries.
class TimeSeriesL2Dist : public Dist<TimeSeries>, public PointL2Dist
{
	static TimeSeriesL2Dist *instance;
	TimeSeriesL2Dist();

  public:
	static TimeSeriesL2Dist& get_inst(); // Returns an instance of TimeSeriesL2Dist.

	// Dist<TimeSeries> override
	double calc(const TimeSeries&, const TimeSeries&) override;
};

// DTW (Dynamic Time Warping) distance defined on class TimeSeries.
class TimeSeriesDTWDist : public Dist<TimeSeries>, public PointDTWDist
{
	static TimeSeriesDTWDist *instance;
	TimeSeriesDTWDist();

  public:
	static TimeSeriesDTWDist& get_inst(); // Returns an instance of TimeSeriesDTWDist.

	// Dist<TimeSeries> override
	double calc(const TimeSeries&, const TimeSeries&) override;
};

// TimeSeries class implementation.

TimeSeries::TimeSeries(int sliding_window) : sliding_window(sliding_window) { }

void TimeSeries::set_sliding_window(int sliding_window)
{
	this->sliding_window = sliding_window;
}

int TimeSeries::read(RawDataInstance* instance, int cnt)
{
	std::vector<double> values;
	CsvDataInstance<double>* casted_instance_dbl = dynamic_cast<CsvDataInstance<double>*>(instance);
	if (casted_instance_dbl != nullptr)
	{
		int limit = std::min(next_index + cnt, casted_instance_dbl->size());
		label = casted_instance_dbl->get_label();
		for (int i = next_index; i < limit; i++)
			values.push_back((*casted_instance_dbl)[i]);
		next_index = limit;
	}

	CsvDataInstance<int>* casted_instance_int = dynamic_cast<CsvDataInstance<int>*>(instance);
	if (casted_instance_int != nullptr)
	{
		int limit = std::min(next_index + cnt, casted_instance_dbl->size());
		label = casted_instance_dbl->get_label();
		for (int i = next_index; i < limit; i++)
			values.push_back((*casted_instance_int)[i]);
		next_index = limit;
	}

	push_back(values);
	
	return values.size();
}

void TimeSeries::from_raw_data_instance(RawDataInstance* instance)
{
	data.clear();
	next_index = 0;
	read(instance, sliding_window);
}

void TimeSeries::push_back(const std::vector<double> &values)
{
	data.insert(data.end(), values.begin(), values.end());
	if (sliding_window > 0 && data.size() > sliding_window)
		data.erase(data.begin(), data.begin() + data.size() - sliding_window);
}

// TimeSeriesL2Dist class implementation.

TimeSeriesL2Dist *TimeSeriesL2Dist::instance;

TimeSeriesL2Dist::TimeSeriesL2Dist()
{
}

TimeSeriesL2Dist& TimeSeriesL2Dist::get_inst()
{
	if (instance == nullptr)
		instance = new TimeSeriesL2Dist;
	return *instance;
}

double TimeSeriesL2Dist::calc(const TimeSeries &ts1, const TimeSeries &ts2)
{
	return PointL2Dist::calc(ts1, ts2);
}

// TimeSeriesDTWDist class implementation.

TimeSeriesDTWDist *TimeSeriesDTWDist::instance;

TimeSeriesDTWDist::TimeSeriesDTWDist()
{
}

TimeSeriesDTWDist& TimeSeriesDTWDist::get_inst()
{
	if (instance == nullptr)
		instance = new TimeSeriesDTWDist;
	return *instance;
}

double TimeSeriesDTWDist::calc(const TimeSeries &ts1, const TimeSeries &ts2)
{
	return PointDTWDist::calc(ts1, ts2);
}

} // namespace knng
#endif