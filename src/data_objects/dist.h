#ifndef DIST_H
#define DIST_H

#include <exception>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include "../common/utils.h"
#include "../common/string_converter.h"

namespace knng
{

// Abstract class for distance function.
// Each concrete distance function should inherit this class.
template <class T> // T is the type of the objects for which distances will be calculated.
class Dist
{
  public:
	virtual double calc(const T&, const T&) = 0; // Calculates distance between two objects.
};

// Zero distance function.
// For each two objects this distance function returns 0.
template <class T> // T is the type of the objects for which distances will be calculated.
class ZeroDist : public Dist<T>
{
  static ZeroDist* instance;

  public:
	double calc(const T&, const T&) override;	
	static ZeroDist& get_inst();
};

// Exception that is returned in the case when two objects are not compatible
// and therefore the distance cannot be calculated.
struct IncompatibleException : public std::exception
{
   const char * what () const throw ()
   {
      return "Incompatible objects exception";
   }
};

// Cache mechanism for storing already calculated distance values.
// T is the type of the objects for wich distance is calculated.
template <class T>
class DistCache : StringConverter
{
	// Type of hash table that is used for storing calculated distances.
	// Key:   Pair of pointers to the objects for which distance is calculated and stored.
	// Value: Pair which contains distance value and number of dependencies on that value.
	//        When number of dependencies for the value becomes zero, the whole entry is expelled.
	typedef std::unordered_map<std::pair<T*, T*>, std::pair<double, int>, PointersPairHash<T>, PointersPairEqual<T> > cache_map;
	
	Dist<T>* dist_func; // Distance function.
	cache_map cache;    // Hash table that stores calculated distances.
	int dist_calcs = 0; // Total number of calls to dist_func.

  public:
	DistCache(Dist<T> &dist_func);                                               // Constructor that accepts distance function.
	void update_dist_func(Dist<T> &dist_func);                                   // Updates distance function.
	Dist<T>& get_dist_func();                                                    // Returns underlying distance function.
	double get_and_store_distance(T* node1, T* node2, int dependencies_cnt = 1); // Returns the distance between node1 and node2, and increases the number of dependencies to that distance.
	void store_distance(T* node1, T* node2, double d, int dependencies_cnt = 1); // Stores the distance between node1 and node2, and increases the number of dependencies to that distance.
	double get_distance(T* node1, T* node2);                                     // Returns the distance between node1 and node2. If the distance is not in cache, throws invalid_argument exception.
	bool has_distance(T* node1, T* node2);                                       // Returns true if chache has distance between given nodes.
	void remove_distance(T* node1, T* node2, int dependencies_cnt = 1);          // Decreases the number of dependencies for the distance.
	void nodes_changed(std::unordered_set<T*> &nodes);                           // Updates distance values for all pairs in which given nodes are present.
	int get_dist_calcs();                                                        // Returns the total number of calls to dist_func.
	void set_dist_calcs(int dist_calcs);                                         // Sets the dist_calcs value.
	int get_dists_cnt();                                                         // Returns the number of currently stored values.
	int get_dependencies_count(T* node1, T* node2);                              // Returns the dependencies count for the given pair.
	void clear();                                                                // Removes all entries from cache.
	void reset();                                                                // Removes all entries from cache and resets the dist calcs counter.

	// StringConverter overrides
	std::string to_string(bool brief = true) override;
};

// ZeroDist class implementation.

template <class T>
ZeroDist<T> *ZeroDist<T>::instance;

template <class T>
ZeroDist<T>& ZeroDist<T>::get_inst()
{
	if (instance == nullptr)
		instance = new ZeroDist;
	return *instance;
}

template <class T>
double ZeroDist<T>::calc(const T &p1, const T &p2)
{
	return 0;
}

// DistCache class implementation.

template <class T>
DistCache<T>::DistCache(Dist<T> &dist_func) : dist_func(&dist_func) { }

template <class T>
void DistCache<T>::update_dist_func(Dist<T> &dist_func)
{
	this->dist_func = &dist_func;
}

template <class T>
Dist<T>& DistCache<T>::get_dist_func()
{
	return *dist_func;
}

template <class T>
double DistCache<T>::get_and_store_distance(T* node1, T* node2, int dependencies_cnt)
{
	std::pair<T*, T*> cache_key(node1, node2);
	typename cache_map::iterator cache_it = cache.find(cache_key);
	double dist;
	if (cache_it == cache.end())
	{
		dist_calcs++;
		dist = dist_func->calc(*node1, *node2);
		// For supporting comparison with distances read from file.
		std::stringstream s;
		s << dist;
		dist = std::stod(s.str());
		cache[cache_key] = std::make_pair(dist, dependencies_cnt);
	}
	else
	{
		dist = cache_it->second.first;
		cache_it->second.second += dependencies_cnt;
	}
	return dist;
}

template <class T>
void DistCache<T>::store_distance(T* node1, T* node2, double d, int dependencies_cnt)
{
	std::pair<T*, T*> cache_key(node1, node2);
	typename cache_map::iterator cache_it = cache.find(cache_key);
	if (cache_it == cache.end())
		cache[cache_key] = std::make_pair(d, dependencies_cnt);
	else
		cache_it->second.second += dependencies_cnt;
}

template <class T>
double DistCache<T>::get_distance(T* node1, T* node2)
{
	typename cache_map::iterator cache_it = cache.find(std::make_pair(node1, node2));	
	if (cache_it == cache.end())
		throw std::invalid_argument("Distance for given nodes is not in the cache.");

	return cache_it->second.first;
}

template <class T>
bool DistCache<T>::has_distance(T* node1, T* node2)
{
	typename cache_map::iterator cache_it = cache.find(std::make_pair(node1, node2));	
	return cache_it != cache.end();
}

template <class T>
void DistCache<T>::remove_distance(T* node1, T* node2, int dependencies_cnt)
{
	if (dependencies_cnt <= 0) return;
	typename cache_map::iterator cache_it = cache.find(std::make_pair(node1, node2));
	if (cache_it == cache.end()) return;
	cache_it->second.second -= dependencies_cnt;
	if (cache_it->second.second <= 0)
		cache.erase(cache_it);
}

template <class T>
void DistCache<T>::nodes_changed(std::unordered_set<T*> &nodes)
{
	typename cache_map::iterator cache_it = cache.begin();
	while(cache_it != cache.end())
	{
		if (nodes.find(cache_it->first.first) != nodes.end() || nodes.find(cache_it->first.second) != nodes.end())
		{
			double dist = dist_func->calc(*(cache_it->first.first), *(cache_it->first.second));
			// For supporting comparison with distances read from file.
			std::stringstream s;
			s << dist;
			dist = std::stod(s.str());
			cache_it->second.first = dist;
			dist_calcs++;
		}
		cache_it++;
	}
}

template <typename T>
int DistCache<T>::get_dist_calcs()
{
	return dist_calcs;
}

template <typename T>
void DistCache<T>::set_dist_calcs(int dist_calcs)
{
	this->dist_calcs = dist_calcs;
}

template <typename T>
int DistCache<T>::get_dists_cnt()
{
	return cache.size();
}

template <typename T>
int DistCache<T>::get_dependencies_count(T* node1, T* node2)
{
	typename cache_map::iterator cache_it = cache.find(std::make_pair(node1, node2));
	if (cache_it == cache.end())
		throw std::invalid_argument("Distance for given nodes is not in the cache.");

	return cache_it->second.second;
}

template <typename T>
void DistCache<T>::clear()
{
	cache.clear();
}

template <typename T>
void DistCache<T>::reset()
{
	clear();
	dist_calcs = 0;
}

template <typename T>
std::string DistCache<T>::to_string(bool brief)
{
	std::stringstream sout;
	sout << "===== Dist cache =====" << std::endl;
	typename cache_map::iterator cache_it = cache.begin();
	while (cache_it != cache.end())
	{
		sout << "(";
		sout << ((StringConverter*)cache_it->first.first)->to_string();
		sout << ",";
		sout << ((StringConverter*)cache_it->first.second)->to_string();
		sout << ") -> d=" << cache_it->second.first << ", n=" << cache_it->second.second << std::endl;
		cache_it++;
	}
	sout << "======================" << std::endl;
	return sout.str();
}

} // namespace knng
#endif