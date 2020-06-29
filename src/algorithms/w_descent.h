#ifndef W_DESCENT_H
#define W_DESCENT_H

#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include "../graph/knngraph.h"
#include "../algorithms/w_distributions.h"
#include "../exceptions/not_implemented_exception.h"
#include "../common/utils.h"
#include "knng_approx.h"

namespace knng
{

// Abstract class for w-descent algorithms.
template <typename NodeType>
class WDescent : public KNNGApproximation<NodeType>
{
  public:
	// Enumerates algorithms for determination when the w-descent should stop updating KNN-Graph approximation.
	enum class TerminationCriteria
	{
		FixedIterationsCount, // w-descent stops after fixed number of iterations.
		NodesConvergence      // w-descent stops when all nodes converged (meaning that random walks that start from the nodes update very few NN-lists)
	};
  
  protected:
	TerminationCriteria term_criteria;       // Termination criteria.
	int ws_count;                            // Number of walks that start from one node.
	int rand_count;							 // Number of random comparisons in randomization phase.
	int its_count;                           // Number of iterations.
	double conv_ratio;                       // Convergence ratio. Value obtained by multiplying ws_count and conv_ratio represents the treshold of updates obtained by random walks from a given node.
	int current_iteration = 0;               // Current iteration.
	KNNGraph<NodeType> knng;                 // Current KNN-Graph approximation.
	std::vector<NodeType*>& nodes;           // List of all nodes.

  public:
	WDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int ws_count, int rand_count, int its_count);
	WDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int ws_count, int rand_count, int max_its_count, double conv_ratio);
	void generate_knng() override;                              			 // Creates KNN-Graph approximation.
	void update_knng(std::unordered_set<NodeType*> &changed_nodes) override; // Improves current KNN-Graph approximation due to nodes change. 
	void generate_real_knng();                                 				 // Creates real KNN-Graph.
	KNNGraph<NodeType>& get_knng() override;                    			 // Returns current KNN-Graph approximation.
	virtual std::string get_name() override;								 // Returns name of the algorithm.

	// StringConverter overrides
	virtual std::string to_string(bool brief = true) override;
	virtual void from_string(std::string str, bool brief = true) override;
  
  protected:
  	virtual std::list<NodeType*> get_nodes_for_comparison(NodeType* node, int count) = 0;
	virtual void on_initial_graph_initialized() = 0;
	virtual void on_graph_changed() = 0;
	virtual void on_iteration_changed() = 0;
	virtual void on_edges_added(NodeType* node1, NodeType* node2, int edges_count) = 0;
	virtual void on_edges_removed(std::list<std::pair<NodeType*, NodeType*> > &removed_edges) = 0;

  private:
	void init_random_knng(); // Initializes random KNN-Graph.
	void iterate(std::list<NodeType*> &nodes_list);			 // The core of the algorithm.
	int improve_knng(WDistribution<NodeType>* distribution); // Improves current KNN-Graph approximation by applying one iteration of the algorithm. Returns number of updates of nodes' nn lists. Parameter distribution provides the number of walks from each node.
	void randomize(WDistribution<NodeType>* distribution); 	 // Randomize points.
};

// WDescent class implementation.

template <typename NodeType>
WDescent<NodeType>::WDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int ws_count, int rand_count, int its_count) : knng(k, dist, nodes), ws_count(ws_count), rand_count(rand_count), its_count(its_count), nodes(nodes)
{
	term_criteria = TerminationCriteria::FixedIterationsCount;
}

template <typename NodeType>
WDescent<NodeType>::WDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int ws_count, int rand_count, int max_its_count, double conv_ratio) : knng(k, dist, nodes), ws_count(ws_count), rand_count(rand_count), its_count(max_its_count), conv_ratio(conv_ratio), nodes(nodes)
{
	term_criteria = TerminationCriteria::NodesConvergence;
}

template <typename NodeType>
void WDescent<NodeType>::generate_knng()
{
	std::list<NodeType*> nodes_list = knng.get_nodes();
	init_random_knng();
	iterate(nodes_list);	
}

template <typename NodeType>
void WDescent<NodeType>::update_knng(std::unordered_set<NodeType*> &nodes)
{
	std::unordered_set<NodeType*> affected_nodes = knng.nodes_changed(nodes);
	on_graph_changed();
	std::list<NodeType*> nodes_list(affected_nodes.begin(), affected_nodes.end());
	std::copy(nodes.begin(), nodes.end(), std::back_inserter(nodes_list));
	iterate(nodes_list);
}

template <typename NodeType>
void WDescent<NodeType>::generate_real_knng()
{
	knng.generate();
	on_initial_graph_initialized();
}

template <typename NodeType>
KNNGraph<NodeType>& WDescent<NodeType>::get_knng()
{
	return knng;
}

template <typename NodeType>
std::string WDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "wdes";
	sout << "_wcnt" << ws_count;
	sout << "_rndcnt" << rand_count;
	sout << "_it" << its_count;
	if (term_criteria == WDescent<NodeType>::TerminationCriteria::FixedIterationsCount)
		sout << "_tc-fixed-it";
	else
		sout << "_tc-conv-" << conv_ratio;

	std::string path = sout.str();
	
	return path;
}

template <typename NodeType>
std::string WDescent<NodeType>::to_string(bool brief)
{
	std::stringstream sout;
	if (!brief)
		sout << "Current iteration: ";
	sout << current_iteration << std::endl;
	if (!brief)
		sout << "======== KNN-Graph ========" << std::endl;
	sout << knng.to_string(brief);

	return sout.str();
}

template <typename NodeType>
void WDescent<NodeType>::from_string(std::string str, bool brief)
{
	if (!brief) throw NotImplementedException();
	
	std::string line;
	std::stringstream s_in(str);
	
	s_in >> current_iteration;
	
	std::stringstream s_knng;
	while (std::getline(s_in, line))
	{
		s_knng << line << std::endl;
	}
	knng.from_string(s_knng.str());
}

template <typename NodeType>
void WDescent<NodeType>::init_random_knng()
{
	current_iteration = 0;
	knng.init_randomly();
	on_initial_graph_initialized();
}

template <typename NodeType>
void WDescent<NodeType>::iterate(std::list<NodeType*> &nodes_list)
{
	WDistribution<NodeType> *dist_wdesc, *dist_rand;
	if (term_criteria == TerminationCriteria::NodesConvergence)
	{
		dist_wdesc = new NodesSubsetDistributionWithConvergence<NodeType>(nodes_list, ws_count, conv_ratio);
		dist_rand = new NodesSubsetDistributionWithConvergence<NodeType>(nodes_list, rand_count, conv_ratio);
	}
	else
	{
		dist_wdesc = new NodesSubsetDistribution<NodeType>(nodes_list, ws_count);
		dist_rand = new NodesSubsetDistribution<NodeType>(nodes_list, rand_count);
	}
	int i = 0;
	while (i < its_count && dist_wdesc->has_more_ws())
	{
		randomize(dist_rand);
		improve_knng(dist_wdesc);
		i++;
	}
	delete dist_wdesc;
	delete dist_rand;
}

template <typename NodeType>
int WDescent<NodeType>::improve_knng(WDistribution<NodeType>* distribution)
{
	current_iteration++;
	on_iteration_changed();
	DistCache<NodeType>& dist_cache = this->knng.get_dist_cache();
	int updates_cnt = 0;
	std::unordered_map<NodeType*, int> updates_cnt_map;
	NodeType* node;
	std::list<std::pair<NodeType*, NodeType*> > removed_edges;
	int cnt = 0;
	knng.reset_iterator();
	while ((node = knng.next_node()) != nullptr)
	{
		int w_cnt = distribution->get_w_cnt(node);
		if (w_cnt == 0) continue;
		cnt++;
		for (NodeType* node2: get_nodes_for_comparison(node, w_cnt))
		{
			double d = dist_cache.get_and_store_distance(node, node2, 2);
			int successful_additions = 0;
			if (this->knng.try_add_edge(node, node2, d, &removed_edges))
			{
				updates_cnt_map[node]++;
				successful_additions++;
			}
			if (this->knng.try_add_edge(node2, node, d, &removed_edges))
			{
				updates_cnt_map[node2]++;
				successful_additions++;
			}
			dist_cache.remove_distance(node, node2, 2-successful_additions);
			
			on_edges_added(node, node2, successful_additions);
			updates_cnt += successful_additions;
		}
	}

	distribution->report_nn_updates(updates_cnt_map);
	on_edges_removed(removed_edges);

	return updates_cnt;
}

template <typename NodeType>
void WDescent<NodeType>::randomize(WDistribution<NodeType>* distribution)
{
	DistCache<NodeType>& dist_cache = this->knng.get_dist_cache();

	int cnt = 0;

	std::unordered_map<NodeType*, int> updates_cnt_map;

	typename std::vector<NodeType*>::iterator it = nodes.begin();
	for (NodeType* node: nodes)
	{
		int r = distribution->get_w_cnt(node);
		if (r <= 0) continue;
		cnt++;
		std::unordered_set<int> range = rand_range(0, this->nodes.size()-1, std::min((int)(this->nodes.size()), r));
		std::vector<int> range_vector(range.begin(), range.end());
		std::sort(range_vector.begin(), range_vector.end());
		typename std::vector<NodeType*>::iterator it2 = this->nodes.begin();
		int prev_index = 0;
		for (int rnd_index: range_vector)
		{
			advance(it2, rnd_index - prev_index);
			prev_index = rnd_index;
			
			NodeType* random_node = *it2;

			if (random_node == node) continue;

			double d = dist_cache.get_and_store_distance(node, random_node, 2);
			int added = 0;

			if (this->knng.try_add_edge(node, random_node, d))
			{
				if (this->knng.get_edges(node).front() == random_node)
					updates_cnt_map[node]++;
				added++;
			}

			if (this->knng.try_add_edge(random_node, node, d))
			{
				if (this->knng.get_edges(random_node).front() == node)
					updates_cnt_map[random_node]++;
				added++;
			}

			dist_cache.remove_distance(node, random_node, 2-added);
		}
	}

	distribution->report_nn_updates(updates_cnt_map);
}

} // namespace wdescent
#endif