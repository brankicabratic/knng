#ifndef DISTS_POOL_H
#define DISTS_POOL_H

#include <type_traits>
#include "dist.h"
#include "point.h"
#include "time_series.h"

namespace knng
{

const std::string zero_dist = "zero";
const std::string l2_dist = "l2";
const std::string dtw_dist = "dtw";

template <typename T>
Dist<T>& get_dist(std::string name)
{
	// Type independent distances.
	if (name == zero_dist)
		return ZeroDist<T>::get_inst();

	// Type dependent distances.
	if constexpr (std::is_same<T, Point>::value)
	{
		if (name == l2_dist)
			return PointL2Dist::get_inst();
		else if (name == dtw_dist)
			return PointDTWDist::get_inst();
	}
	if constexpr (std::is_same<T, TimeSeries>::value)
	{
		if (name == l2_dist)
			return TimeSeriesL2Dist::get_inst();
		if (name == dtw_dist)
			return TimeSeriesDTWDist::get_inst();
	}

	throw std::invalid_argument("get_dist - Unexisting distance measure.");
}

template <typename T>
std::string get_dist_name(Dist<T>& dist)
{
	if (dynamic_cast<const ZeroDist<T>*>(&dist) != nullptr)
		return zero_dist;
	else if (dynamic_cast<const PointL2Dist*>(&dist) != nullptr || dynamic_cast<const TimeSeriesL2Dist*>(&dist) != nullptr)
		return l2_dist;
	else if (dynamic_cast<const PointDTWDist*>(&dist) != nullptr || dynamic_cast<const TimeSeriesDTWDist*>(&dist) != nullptr)
		return dtw_dist;

	throw std::invalid_argument("get_dist - Unrecognized distance measure.");
}


} // namespace knng
#endif