#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <sstream>
#include <unordered_set>

#define BOOST_ALLOW_DEPRECATED_HEADERS 
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace knng
{

template <typename T>
class RandomObjectGenerator
{
  public:
	virtual T* create_random_object() = 0;
};

std::unordered_set<int> rand_range(int min_val, int max_val, int cnt)
{
	int vals_cnt = max_val - min_val + 1;
	if (vals_cnt < cnt)
	{
		std::stringstream s;
		s << "rand_range - Not enough values (needed " << cnt << ") between min_val (" << min_val << ") and max_val (" << max_val << ").";
		throw std::invalid_argument(s.str());
	}

	std::unordered_set<int> rnd_vals;
	while (rnd_vals.size() < cnt)
	{
		int rnd_val = rand() % vals_cnt;
		while (rnd_vals.find(min_val + rnd_val) != rnd_vals.end())
		{
			rnd_val = (rnd_val + 1) % vals_cnt;
		}
		rnd_vals.insert(min_val + rnd_val);
	}
	return rnd_vals;
}

std::unordered_set<int> rand_range(boost::random::mt19937 &gen, boost::random::uniform_int_distribution<> &rnd, int cnt)
{
	int vals_cnt = rnd.max() - rnd.min() + 1;
	if (vals_cnt < cnt)
	{
		std::stringstream s;
		s << "rand_range - Not enough values (needed " << cnt << ") between min_val (" << rnd.min() << ") and max_val (" << rnd.max() << ").";
		throw std::invalid_argument(s.str());
	}

	std::unordered_set<int> rnd_vals;
	while (rnd_vals.size() < cnt)
	{
		int rnd_val = rnd(gen);
		while (rnd_vals.find(rnd_val) != rnd_vals.end())
		{
			rnd_val = (rnd_val - rnd.min() + 1) % vals_cnt + rnd.min();
		}
		rnd_vals.insert(rnd_val);
	}
	return rnd_vals;
}

} // namespace knng
#endif