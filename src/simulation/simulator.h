#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <string>
#include <ostream>
#include <sstream>
#include <chrono>
#include <typeinfo>
#include <random>
#include <boost/filesystem.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include "../algorithms/knng_approx.h"
#include "../datasets/dataset_ts_buffer.h"
#include "../datasets/file_parser.h"
#include "../exceptions/missing_file_exception.h"

namespace knng
{

class Simulator
{
	static std::string cache_path;
	TimeSeriesBuffer& dataset;
	int batch_size_min, batch_size_max;
	int points_min, points_max;
	boost::random::mt19937 gen;
	boost::random::uniform_int_distribution<> points_num_rand, batch_rand;

  public:
	Simulator(TimeSeriesBuffer& dataset, int batch_size);
	Simulator(TimeSeriesBuffer& dataset, int batch_size, int points);
	Simulator(TimeSeriesBuffer& dataset, int batch_size_min, int batch_size_max, int points_min, int points_max);
	void generate_reference_knngs(KNNGApproximation<TimeSeries>* algorithm, std::ostream &out, int run);
	void start_simulation(std::vector<KNNGApproximation<TimeSeries>*>& algorithms, KNNGApproximation<TimeSeries>* reference_algorithm, std::ostream &out, int run);

	static void clear_temp();
	static void clear_ds_cache(std::string dataset, bool temp);

  private:
  	void start_simulation(std::vector<KNNGApproximation<TimeSeries>*>& algorithms, std::ostream &out, int run, KNNGApproximation<TimeSeries>* reference_algorithm = nullptr);
	void update_random_seed(Dist<TimeSeries> &dist, int k, int run);
	void generate_or_load_knng(KNNGApproximation<TimeSeries>* algorithm, KNNGApproximation<TimeSeries>* reference_algorithm, int run, int sim_step, std::ostream &out, std::unordered_set<TimeSeries*> *changed_nodes = nullptr);
	bool load_new_batches(TimeSeriesBuffer &dataset, std::vector<TimeSeries*> &points, std::vector<TimeSeries*> &load_points);
	static std::string get_simulation_cache_path(bool temp);
	static std::string get_simulation_cache_ds_path(std::string dataset, bool temp);
	std::string get_simulation_cache_ds_path(bool temp);
	std::string get_simulation_cache_base_path(Dist<TimeSeries> &dist, int k, int run, bool temp);
	std::string get_algorithm_file_path(KNNGApproximation<TimeSeries>* algorithm, int run, int sim_step, bool temp);
};

// Simulator class implementation.

std::string Simulator::cache_path = "cache_sim";

Simulator::Simulator(TimeSeriesBuffer& dataset, int batch_size)
	: Simulator(dataset, batch_size, batch_size, -1, -1) { }

Simulator::Simulator(TimeSeriesBuffer& dataset, int batch_size, int points)
	: Simulator(dataset, batch_size, batch_size, points, points) { }

Simulator::Simulator(TimeSeriesBuffer& dataset, int batch_size_min, int batch_size_max, int points_min, int points_max)
	: dataset(dataset), batch_size_min(batch_size_min), batch_size_max(batch_size_max), points_min(points_min), points_max(points_max), batch_rand(batch_size_min, batch_size_max), points_num_rand(points_min, points_max) { }

void Simulator::generate_reference_knngs(KNNGApproximation<TimeSeries>* algorithm, std::ostream &out, int run)
{
	std::vector<KNNGApproximation<TimeSeries>*> algorithms {algorithm};
	start_simulation(algorithms, out, run);
}

void Simulator::start_simulation(std::vector<KNNGApproximation<TimeSeries>*>& algorithms, KNNGApproximation<TimeSeries>* reference_algorithm, std::ostream &out, int run)
{
	start_simulation(algorithms, out, run, reference_algorithm);
}

void Simulator::clear_temp()
{
	boost::filesystem::remove_all(get_simulation_cache_path(true));
}

void Simulator::clear_ds_cache(std::string dataset, bool temp)
{
	boost::filesystem::remove_all(get_simulation_cache_ds_path(dataset, temp));
}

void Simulator::start_simulation(std::vector<KNNGApproximation<TimeSeries>*>& algorithms, std::ostream &out, int run, KNNGApproximation<TimeSeries>* reference_algorithm)
{
	if (algorithms.size() == 0) return;

	bool calc_recall = reference_algorithm != nullptr;

	Dist<TimeSeries> &dist = algorithms[0]->get_knng().get_dist_cache().get_dist_func();
	int k = algorithms[0]->get_knng().get_k();

	dataset.reset();	
	update_random_seed(dist, k, run);

	std::vector<TimeSeries*>& instances = dataset.get_instances();
	std::vector<TimeSeries*> instances_for_load;
	std::copy(instances.begin(), instances.end(), std::back_inserter(instances_for_load));

	int sim_step = 0;
	if (calc_recall)
	{
		generate_or_load_knng(reference_algorithm, nullptr, run, sim_step, out);
	}

	for (KNNGApproximation<TimeSeries>* algorithm: algorithms)
	{
		out << ";";
		generate_or_load_knng(algorithm, reference_algorithm, run, sim_step, out);
	}
	out << std::endl;

	std::vector<TimeSeries*> changed_nodes;
	std::unordered_set<TimeSeries*> changed_nodes_set;
	while (true)
	{
		if (!load_new_batches(dataset, instances_for_load, changed_nodes)) break;
		changed_nodes_set.clear();
		changed_nodes_set.insert(changed_nodes.begin(), changed_nodes.end());
		sim_step++;
		if (calc_recall)
		{
			generate_or_load_knng(reference_algorithm, nullptr, run, sim_step, out, &changed_nodes_set);
		}
		for (KNNGApproximation<TimeSeries>* algorithm: algorithms)
		{
			out << ";";
			generate_or_load_knng(algorithm, reference_algorithm, run, sim_step, out, &changed_nodes_set);
		}
		out << std::endl;
	}

	boost::filesystem::remove_all(get_simulation_cache_base_path(dist, k, run, true));
}

void Simulator::update_random_seed(Dist<TimeSeries> &dist, int k, int run)
{
	std::string path = get_simulation_cache_base_path(dist, k, run, false) + "seed";
	unsigned int seed;
	if (file_exists(path))
	{
		std::ifstream file;
		file.open(path);
		file >> seed;
		file.close();
	}
	else
	{
		seed = time(0);
		std::ofstream file;
		file.open(path);
		file << seed;
		file.close();
	}
	gen.seed(seed);
}

void Simulator::generate_or_load_knng(KNNGApproximation<TimeSeries>* algorithm, KNNGApproximation<TimeSeries>* reference_algorithm, int run, int sim_step, std::ostream &out, std::unordered_set<TimeSeries*> *changed_nodes)
{
	std::string cached_knng_path = get_algorithm_file_path(algorithm, run, sim_step, reference_algorithm != nullptr);

	if (file_exists(cached_knng_path))
	{
		std::ifstream file;
		
		file.open(cached_knng_path);
		std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		file.close();
		algorithm->from_string(str);

		file.open(cached_knng_path + "_res");
		std::string str2((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		file.close();
		out << str2;
	}
	else
	{
		int dist_calcs_before = algorithm->get_knng().get_dist_calcs();
		auto start = std::chrono::high_resolution_clock::now();
		if (changed_nodes == nullptr)
			algorithm->generate_knng();
		else
			algorithm->update_knng(*changed_nodes);		
		auto finish = std::chrono::high_resolution_clock::now();
		
		std::stringstream sout;
		sout << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
		sout << "," << (algorithm->get_knng().get_dist_calcs() - dist_calcs_before);
		if (reference_algorithm != nullptr)
			sout << "," << algorithm->get_knng().get_recall(reference_algorithm->get_knng());
		
		std::ofstream file;

		file.open(cached_knng_path);
		file << algorithm->to_string();
		file.close();

		file.open(cached_knng_path + "_res");
		file << sout.str();
		file.close();

		out << sout.str();
	}
}

bool Simulator::load_new_batches(TimeSeriesBuffer &dataset, std::vector<TimeSeries*> &points, std::vector<TimeSeries*> &load_points)
{
	load_points.clear();

	std::unordered_map<TimeSeries*, int> actual_loads_cnts;
	std::vector<int> loads_cnts;
	bool all_points;

	if (points_min < 0)
	{
		all_points = true;
		load_points.insert(load_points.end(), points.begin(), points.end());
	}
	else
	{
		int points_num;
		if (points_min == points_max)
		{
			points_num = points_min;
		}
		else
		{
			points_num = points_num_rand(gen);
		}
		if (points_num > points.size())
		{
			points_num = points.size();
		}
		all_points = points_num == points.size();
		boost::random::uniform_int_distribution<> points_rand(0, points.size()-1);
		std::unordered_set<int> rnd_indices = rand_range(gen, points_rand, points_num);
		std::unordered_set<int>::iterator rnd_indices_it = rnd_indices.begin();
		while (rnd_indices_it != rnd_indices.end())
		{
			load_points.push_back(points[*rnd_indices_it]);
			rnd_indices_it++;
		}
	}

	for (int i = 0; i < load_points.size(); i++)
		loads_cnts.push_back(batch_rand(gen));
	
	dataset.load(load_points, loads_cnts, &actual_loads_cnts);

	int loads_cnts_total = 0;
	std::vector<TimeSeries*>::iterator load_points_it = load_points.begin();
	std::vector<int>::iterator loads_cnts_it = loads_cnts.begin();
	while (load_points_it != load_points.end())
	{
		int point_loads_cnt = actual_loads_cnts.find(*load_points_it)->second;
		loads_cnts_total += point_loads_cnt;
		if (point_loads_cnt < *loads_cnts_it)
		{
			points.erase(std::find(points.begin(), points.end(), *load_points_it));
		}
		load_points_it++;
		loads_cnts_it++;
	}
	return loads_cnts_total > 0 && points.size() > 0;
}

std::string Simulator::get_simulation_cache_path(bool temp)
{
	return cache_path + "/" + (temp ? "temp/" : "");
}

std::string Simulator::get_simulation_cache_ds_path(std::string dataset, bool temp)
{
	return get_simulation_cache_path(temp) + dataset + "/";
}

std::string Simulator::get_simulation_cache_ds_path(bool temp)
{
	PathDescriptor ds_path_desc = splitpath(dataset.get_file_path());
	return Simulator::get_simulation_cache_ds_path(ds_path_desc.filename_without_ext, temp);
}

std::string Simulator::get_simulation_cache_base_path(Dist<TimeSeries> &dist, int k, int run, bool temp)
{
	std::stringstream sout;
	sout << get_simulation_cache_ds_path(temp) << typeid(dist).name() << "_k" << k << "/";
	
	sout << "sw" << dataset.get_sliding_window() << "_";
	sout << "bs" << batch_size_min;
	if (batch_size_min != batch_size_max)
		sout << "-" << batch_size_max;
	sout << "_";
	sout << "p";
	if (points_min < 0)
	{
		sout << "-all";
	}
	else
	{
		sout << points_min;
		if (points_min != points_max)
			sout << "-" << points_max;
	}
	sout << "/" << run << "/";;
	
	std::string path = sout.str();
	boost::filesystem::create_directories(path);
	
	return path;
}

std::string Simulator::get_algorithm_file_path(KNNGApproximation<TimeSeries>* algorithm, int run, int sim_step, bool temp)
{
	std::stringstream out;
	out << get_simulation_cache_base_path(algorithm->get_knng().get_dist_cache().get_dist_func(), algorithm->get_knng().get_k(), run, temp) << algorithm->get_name() << sim_step;
	return out.str();
}

} // namespace knng
#endif