#ifndef KNNGRAPH_H
#define KNNGRAPH_H

#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <algorithm>
#include <utility>
#include <random>
#include <chrono>
#include "graph.h"
#include "../data_objects/dist.h"
#include "../data_objects/dists_pool.h"

namespace knng
{

// Model class for KNN-Graph (K Nearest Neighbors Graph).
template <typename NodeType>
class KNNGraph : public Graph<NodeType>
{
	// Comparator used for sorting the edges of the given node.
	struct EdgeComparator
	{
		NodeType* node; // Node whose edges need to be sorted.
		DistCache<NodeType> &dist_cache;

		EdgeComparator(NodeType* node, DistCache<NodeType> &dist_cache) : node(node), dist_cache(dist_cache) { }

		// Compares two node's edges by their distance from node.
		bool operator ()(NodeType* &edge1_node, NodeType* &edge2_node)
		{
			return dist_cache.get_distance(node, edge1_node) < dist_cache.get_distance(node, edge2_node);
		}
	};

	int k;                                      // Max out-degree.
	DistCache<NodeType> dist_cache;             // Cache used for distance values retreival and storage.
	std::unordered_map<NodeType*, int> hubness; // Indegrees of graph nodes (also known as hubness values).

  public:
	KNNGraph(int k, Dist<NodeType> &dist);                                                 // Initializes empty graph (without nodes and without edges).
	KNNGraph(int k, Dist<NodeType> &dist, std::vector<NodeType*> &nodes);                  // Initializes graph with the given node set (without edges).
	void add_node(NodeType *node) override;                                                // Adds the node to the graph.
	bool add_edge(NodeType* node1, NodeType* node2) override;                              // Adds directed edge (node1, node2) at the beginning of the edges list.
	void generate();                                                                       // Generates exact KNN-Graph.
	void generate_for_subset(const std::unordered_set<NodeType*>& nodes, std::ostream& out); // Generates nearest neighbors list for a subset of graph's node. As it generates one list it imidiately outputs it.
	void regenerate(std::unordered_set<NodeType*> &changed_nodes);                         // Regenerates exact KNN-Graph due to nodes changes.
	void init_randomly();                                                                  // Initializes k-regular graph with randomly chosen edges.
	void init_randomly(std::vector<NodeType*>& nodes);                                     // Initializes k-regular graph with edges randomly chosen from the given list.
	int get_k();                                                                           // Returns k value.
	double get_recall(KNNGraph<NodeType> &real_knng);                                      // Returns recall (correctness of KNN-Graph approximation).
	double get_scan_rate();                                                                // Ratio between number of actual calls to distance function to number of calls to distance function in case of the naive KNN-Graph generation.
	DistCache<NodeType>& get_dist_cache();
	int get_dist_calcs();                                                                  // Returns total number of calls to distance function.
	KNNGraph<NodeType>* get_reduced_to_k(int k);                                           // Creates and returns a new instance of KNNGraph for new k value. This k value must not be greater than original k value.
	
	// Tries to add two directed edges: (node1,node2) and (node2,node1). Insertion of edge (x,y) fails if
	// the distance between the nodes is larger than a distance from x to the x's current k-th neighbor.
	// Returns the number of succesfully inserted edges and fills list removed_edges (if it is not nullptr)
	// with the edges that are expelled from NN lists during the process.
	int try_add_edges(NodeType* node1, NodeType* node2, std::list<std::pair<NodeType*, NodeType*> >* removed_edges = nullptr);
	
	// Tries to add directed edge (node1,node2). Insertion of the edge fails if the distance
	// between the nodes (d) is larger than a distance from node1 to the node1's current k-th neighbor.
	// Returns true if the insertion was succesfull and false otherwise. If some edge was expelled from
	// node1's NN list, this method adds it to removed_edge list.
	bool try_add_edge(NodeType* node1, NodeType* node2, double d, std::list<std::pair<NodeType*, NodeType*> >* removed_edge = nullptr);
	void remove_edges() override;                                                      // Removes all edges from the graph.
	std::unordered_set<NodeType*> nodes_changed(std::unordered_set<NodeType*> &nodes); // Updates cached distances for changed nodes, and sorts NN lists accordingly. Returns a set of nodes that are affected by changed nodes.
	int get_hubness(NodeType* node);                                                   // Returns indegree (hubness value) of the given node.

	// StringConverter overrides
	std::string to_string(bool brief = true) override;
	void from_string(std::string str, bool brief = true) override;

  private:
  	void init(int k);                                                                                     // Initializes graph. Used in constructors.
	bool add_edge(NodeType* node1, NodeType* node2, typename std::list<NodeType*>::iterator it) override; // Adds directed edge (node1, node2) at the position it.
	NodeType* remove_last_edge(NodeType *node) override;                                                  // Removes last edge from the node and returns it.
};

template <typename NodeType>
KNNGraph<NodeType>::KNNGraph(int k, Dist<NodeType> &dist) : dist_cache(dist)
{
	init(k);
}

template <typename NodeType>
KNNGraph<NodeType>::KNNGraph(int k, Dist<NodeType> &dist, std::vector<NodeType*> &nodes) : dist_cache(dist)
{	
	init(k);
	for (typename std::vector<NodeType*>::iterator it = nodes.begin(); it != nodes.end(); ++it)
		add_node(*it);
}

template <typename NodeType>
void KNNGraph<NodeType>::add_node(NodeType* node)
{
	Graph<NodeType>::add_node(node);
	hubness[node] = 0;
}

template <typename NodeType>
bool KNNGraph<NodeType>::add_edge(NodeType* node1, NodeType* node2)
{
	bool success = Graph<NodeType>::add_edge(node1, node2);
	if (success)
		hubness[node2]++;
	return success;
}

template <typename NodeType>
void KNNGraph<NodeType>::generate()
{
	if (k + 1 >= this->nodes_count())
		throw std::length_error("Not enough vertices for given k.");
	
	remove_edges();

	std::list<NodeType*> nodes = this->get_nodes();
	typename std::list<NodeType*>::iterator it = nodes.begin();
	while (it != nodes.end())
	{
		typename std::list<NodeType*>::iterator it2 = it;
		it2++;
		while (it2 != nodes.end())
		{
			try_add_edges(*it, *it2);
			it2++;
		}
		it++;
	}
}

template <typename NodeType>
void KNNGraph<NodeType>::generate_for_subset(const std::unordered_set<NodeType*>& nodes, std::ostream& out)
{
	if (k + 1 >= this->nodes_count())
		throw std::length_error("Not enough vertices for given k.");

	std::list<NodeType*> all_nodes = this->get_nodes();
	typename std::unordered_set<NodeType*>::const_iterator it = nodes.begin();
	while (it != nodes.end())
	{
		typename std::list<NodeType*>::iterator it2 = all_nodes.begin();
		while (it2 != all_nodes.end())
		{
			if (*it != *it2)
			{
				double d = dist_cache.get_and_store_distance(*it, *it2);
				if (!try_add_edge(*it, *it2, d))
				{
					dist_cache.remove_distance(*it, *it2);
				}
			}
			it2++;
		}

		out << ((Identifiable*)*it)->get_id() << " ";
		std::list<NodeType*> edges = this->get_edges(*it);
		typename std::list<NodeType*>::iterator it_edges = edges.begin();
		while(it_edges != edges.end())
		{
			NodeType* node = *it_edges;
			out << ((Identifiable*)node)->get_id() << "(" << dist_cache.get_distance(*it, node) << ")" << " ";
			it_edges++;
		}
		out << std::endl;

		it++;
	}
}

template <typename NodeType>
void KNNGraph<NodeType>::regenerate(std::unordered_set<NodeType*> &changed_nodes)
{
	if (k + 1 >= this->nodes_count())
		throw std::length_error("Not enough vertices for given k.");
	
	std::unordered_set<NodeType*> affected_nodes = nodes_changed(changed_nodes);
	affected_nodes.insert(changed_nodes.begin(), changed_nodes.end());

	DistCache<NodeType> dist_cache_cp = dist_cache;

	std::list<NodeType*> nodes = this->get_nodes();
	for (NodeType* node1: affected_nodes)
		for (NodeType* node2: nodes)
			if (affected_nodes.find(node2) == affected_nodes.end())
				try_add_edges(node1, node2);

	typename std::unordered_set<NodeType*>::iterator it = affected_nodes.begin();
	while (it != affected_nodes.end())
	{
		typename std::unordered_set<NodeType*>::iterator it2 = it;
		it2++;
		while (it2 != affected_nodes.end())
		{
			try_add_edges(*it, *it2);
			it2++;
		}
		it++;
	}
}

template <typename NodeType>
void KNNGraph<NodeType>::init_randomly()
{
	std::vector<NodeType*> nodes = this->get_sorted_nodes();
	init_randomly(nodes);
}

template <typename NodeType>
void KNNGraph<NodeType>::init_randomly(std::vector<NodeType*>& nodes)
{
	int nodes_cnt = nodes.size();
	if (k + 1 >= nodes_cnt)
		throw std::length_error("Not enough vertices for KNN-Graph initialization.");
	
	remove_edges();
	
	this->reset_iterator();

	std::uniform_int_distribution<std::mt19937::result_type> udist(0, nodes_cnt - 1);
	std::mt19937 rng;
	rng.seed(std::chrono::system_clock::now().time_since_epoch().count());

	this->reset_iterator();
	NodeType* curr_node;
	while ((curr_node = this->next_node()) != nullptr)
	{
		std::unordered_set<NodeType*> added_nodes;
		added_nodes.insert(curr_node);
		
		while (added_nodes.size() <= k)
		{
			int node_index = udist(rng);
			NodeType* node_to_add;
			do
			{
				node_to_add = nodes[node_index];
				node_index = (node_index + 1) % nodes_cnt;
			} while (added_nodes.find(node_to_add) != added_nodes.end());
 			try_add_edges(curr_node, node_to_add);
			added_nodes.insert(node_to_add);
		}
	}
}

template <typename NodeType>
int KNNGraph<NodeType>::get_k()
{
	return k;
}

template <typename NodeType>
double KNNGraph<NodeType>::get_recall(KNNGraph<NodeType> &real_knng)
{
	if (k != real_knng.k)
		throw std::invalid_argument("KNNGraph<NodeType>::get_recall - K values are not equal.");

	if (this->nodes_count() != real_knng.nodes_count())
		throw std::invalid_argument("KNNGraph<NodeType>::get_recall - Node sets are not equal.");
	
	double epsilon = get_epsilon();

	this->reset_iterator();
	int hits_count = 0;
	int nodes_count = 0;
	std::pair<NodeType*, std::list<NodeType*>*> curr;
	while ((curr = this->next_node_with_edges()).first != nullptr)
	{
		std::list<NodeType*> real_knng_edges;
		try
		{
			real_knng_edges = real_knng.get_edges(curr.first);
		}
		catch (std::invalid_argument)
		{
			throw std::invalid_argument("KNNGraph<NodeType>::get_recall - Node sets are not equal.");
		}
		
		if (real_knng_edges.size() < k) continue;

		nodes_count++;

		typename std::list<NodeType*>::iterator real_nn_list_it = real_knng_edges.begin();
		typename std::list<NodeType*>::iterator approx_nn_list_it = curr.second->begin();
		double real_dist = real_knng.dist_cache.get_distance(curr.first, *real_nn_list_it);
		double approx_dist = dist_cache.get_distance(curr.first, *approx_nn_list_it);
		while (true)
		{
			if (std::fabs(approx_dist-real_dist) < epsilon)
			{
				hits_count++;
				real_nn_list_it++;
				approx_nn_list_it++;
				if (real_nn_list_it == real_knng_edges.end() || approx_nn_list_it == curr.second->end())
					break;
				real_dist = real_knng.dist_cache.get_distance(curr.first, *real_nn_list_it);
				approx_dist = dist_cache.get_distance(curr.first, *approx_nn_list_it);
			}
			else if (real_dist < approx_dist)
			{
				real_nn_list_it++;
				if (real_nn_list_it == real_knng_edges.end())
					break;
				real_dist = real_knng.dist_cache.get_distance(curr.first, *real_nn_list_it);
			}
			else
			{
				approx_nn_list_it++;
				if (approx_nn_list_it == curr.second->end())
					break;
				approx_dist = dist_cache.get_distance(curr.first, *approx_nn_list_it);
			}
		}
	}

	return (double)hits_count/(k*nodes_count);
}

template <typename NodeType>
double KNNGraph<NodeType>::get_scan_rate()
{
	int dist_calcs = dist_cache.get_dist_calcs();
	int nodes_cnt = this->nodes_count();
	int base_dist_calcs = (nodes_cnt * (nodes_cnt - 1)) / 2;
	return (double)dist_calcs / base_dist_calcs;
}

template <typename NodeType>
DistCache<NodeType>& KNNGraph<NodeType>::get_dist_cache()
{
	return dist_cache;
}

template <typename NodeType>
int KNNGraph<NodeType>::get_dist_calcs()
{
	return dist_cache.get_dist_calcs();
}

template <typename NodeType>
KNNGraph<NodeType>* KNNGraph<NodeType>::get_reduced_to_k(int k)
{
	if (k > this->k)
		throw std::invalid_argument("KNNGraph<NodeType>::get_reduced_to_k - Given k value must not be greater than original KNN graph's k.");
	
	std::vector<NodeType*> nodes = Graph<NodeType>::get_nodes_vector();
	KNNGraph<NodeType>* reduced_knng = new KNNGraph<NodeType>(k, dist_cache.get_dist_func(), nodes);
	reduced_knng->dist_cache.set_dist_calcs(dist_cache.get_dist_calcs());

	Graph<NodeType>::reset_iterator();

	while (true)
	{
		std::pair<NodeType*, std::list<NodeType*>*> node_edges = Graph<NodeType>::next_node_with_edges();
		
		if (node_edges.first == nullptr) break;

		for (auto edge: *(node_edges.second))
		{
			double dist = dist_cache.get_distance(node_edges.first, edge);
			reduced_knng->dist_cache.store_distance(node_edges.first, edge, dist, 1);
			reduced_knng->try_add_edge(node_edges.first, edge, dist);
		}
	}

	return reduced_knng;
}

template <typename NodeType>
int KNNGraph<NodeType>::try_add_edges(NodeType* node1, NodeType* node2, std::list<std::pair<NodeType*, NodeType*> >* removed_edges)
{
	if (node1 == node2)
		return 0;
	
	double d = dist_cache.get_and_store_distance(node1, node2, 2);
	int successful_additions = 0;
	if (try_add_edge(node1, node2, d, removed_edges))
		successful_additions++;
	if (try_add_edge(node2, node1, d, removed_edges))
		successful_additions++;
	
	dist_cache.remove_distance(node1, node2, 2-successful_additions);

	return successful_additions;
}

template <typename NodeType>
bool KNNGraph<NodeType>::try_add_edge(NodeType* node1, NodeType* node2, double d, std::list<std::pair<NodeType*, NodeType*> >* removed_edge)
{
	// Predicate that is instantiated for the certain node (n) and certain distance (d). For the node n_test,
	// this predicate returns true if dist(n, n_test) is smaller than d or equal to d, and false otherwise.
	struct SmallerOrEqualDistancePredicate
	{
		double d;    // Limit distance.
		NodeType* n; // Node from which distance is measured.
		DistCache<NodeType> &dist_cache;

		SmallerOrEqualDistancePredicate(NodeType* node, double distance, DistCache<NodeType> &dist_cache) : n(node), d(distance), dist_cache(dist_cache) { }

		// Compares two node's edges by their distance from node.
		bool operator ()(NodeType* &node)
		{
			return dist_cache.get_distance(n, node) <= d;
		}
	};

	// Predicate that is instantiated for the certain node (n) and certain distance (d). For the node n_test,
	// this predicate returns true if dist(n, n_test) is not equal to d, and false otherwise.
	struct UnequalDistancePredicate
	{
		double d;    // Distance.
		NodeType* n; // Node from which distance is measured.
		DistCache<NodeType> &dist_cache;

		UnequalDistancePredicate(NodeType* node, double distance, DistCache<NodeType> &dist_cache) : n(node), d(distance), dist_cache(dist_cache) { }

		// Compares two node's edges by their distance from node.
		bool operator ()(NodeType* &node)
		{
			return dist_cache.get_distance(n, node) != d;
		}
	};

	// Predicate that is instantiated for the certain node (n). For the node n_test
	// this predicate returns true if n == n_test, and false otherwise.
	struct EqualNodesPredicate
	{
		NodeType* n; // Node from which distance is measured.

		EqualNodesPredicate(NodeType* node) : n(node) { }

		// Compares two node's edges by their distance from node.
		bool operator ()(NodeType* &node)
		{
			return n == node;
		}
	};

	// Predicate that always returns false.
	struct FalsePredicate
	{
		bool operator ()(NodeType* &node)
		{
			return false;
		}
	};

	if (node1 == node2) return false;

	typename std::list<NodeType*> edges = this->get_edges(node1);

	if (edges.empty())
	{
		this->add_edge(node1, node2);
		return true;
	}

	int found_in_steps;
	typename std::list<NodeType*>::iterator current_edge = this->r_find_edge(node1, SmallerOrEqualDistancePredicate(node1, d, dist_cache), FalsePredicate(), found_in_steps);
	
	// Check if the list contains the same node we want to add.
	if (found_in_steps > 0)
	{
		int edge_exists;
		this->r_find_edge(node1, EqualNodesPredicate(node2), UnequalDistancePredicate(node1, d, dist_cache), current_edge, edge_exists);
		if (edge_exists) // Edge is already there, don't add it again.
		{
			return false;
		}
	}

	// Last element of NN list already had smaller or equal distance than node2, so we will not add it
	if (edges.size() == k && found_in_steps == 1)
	{
		return false;
	}

	this->add_edge(node1, node2, current_edge);

	if (edges.size() == k)
	{
		NodeType* removed_node = this->remove_last_edge(node1);
		if (removed_edge != nullptr)
			(*removed_edge).push_back(std::make_pair(node1, removed_node));
	}

	return true;
}

template <typename NodeType>
void KNNGraph<NodeType>::remove_edges()
{
	dist_cache.clear();
	this->reset_iterator();
	NodeType* curr;
	while ((curr = this->next_node()) != nullptr)
	{
		hubness[curr] = 0;
	}

	Graph<NodeType>::remove_edges();
}

template <typename NodeType>
std::unordered_set<NodeType*> KNNGraph<NodeType>::nodes_changed(std::unordered_set<NodeType*> &nodes)
{
	dist_cache.nodes_changed(nodes);
	std::unordered_set<NodeType*> to_be_updated;
	typename std::unordered_set<NodeType*>::iterator nodes_it = nodes.begin();
	while (nodes_it != nodes.end())
	{
		this->sort_edges(*nodes_it, EdgeComparator(*nodes_it, dist_cache));
		std::list<NodeType*> edges = this->get_r_edges(*nodes_it);

		typename std::list<NodeType*>::iterator edges_it = edges.begin();
		while (edges_it != edges.end())
		{
			if (nodes.find(*edges_it) == nodes.end())
				to_be_updated.insert(*edges_it);
			edges_it++;
		}
		nodes_it++;
	}

	nodes_it = to_be_updated.begin();
	while (nodes_it != to_be_updated.end())
	{
		this->sort_edges(*nodes_it, EdgeComparator(*nodes_it, dist_cache));
		nodes_it++;
	}

	return to_be_updated;
}

template <typename NodeType>
int KNNGraph<NodeType>::get_hubness(NodeType* node)
{
	return hubness[node];
}

template <typename NodeType>
std::string KNNGraph<NodeType>::to_string(bool brief)
{
	std::stringstream sout;
	sout << k << "-" << dist_cache.get_dist_calcs() << "-" << get_dist_name(dist_cache.get_dist_func()) << std::endl;
	for (auto key : this->get_sorted_nodes())
	{
		sout << ((Identifiable*)key)->get_id() << " ";
		std::list<NodeType*> edges = this->get_edges(key);
		typename std::list<NodeType*>::iterator it_edges = edges.begin();
		while(it_edges != edges.end())
		{
			NodeType* node = *it_edges;
			sout << ((Identifiable*)node)->get_id() << "(" << dist_cache.get_distance(key, node) << ")" << " ";
			it_edges++;
		}
		sout << std::endl;
	}
	return sout.str();
}

template <typename NodeType>
void KNNGraph<NodeType>::from_string(std::string str, bool brief)
{
	std::stringstream sin_all(str);

	remove_edges();
	dist_cache.clear();

	sin_all >> k;
	sin_all.ignore();
	int dist_calcs;
	sin_all >> dist_calcs;
	sin_all.ignore();
	std::string dist_name;
	std::getline(sin_all, dist_name);
	dist_cache.update_dist_func(get_dist<NodeType>(dist_name));
	dist_cache.set_dist_calcs(dist_calcs);

	std::string line;
	while (std::getline(sin_all, line))
	{
		std::stringstream sin_line(line);
		int node_id;
		sin_line >> node_id;
		sin_line.ignore();
		NodeType* current_node = this->find_node_by_id(node_id);
		if (current_node == nullptr)
			throw std::domain_error("KNNGraph<NodeType>::from_string - Node with given id is not present in the graph.");
		while (sin_line >> node_id)
		{
			NodeType* neighbor = this->find_node_by_id(node_id);
			if (neighbor == nullptr)
				throw std::domain_error("KNNGraph<NodeType>::from_string - Node with given id is not present in the graph.");
			
			double dist;
			if (sin_line.peek() == '(')
			{
				sin_line.ignore();
				sin_line >> dist;
				sin_line.ignore();
			}
			else
			{
				dist = dist_cache.get_dist_func().calc(*current_node, *neighbor);
				std::stringstream s;
				s << dist;
				std::stringstream s2(s.str());
				s2 >> dist;
			}
			
			dist_cache.store_distance(current_node, neighbor, dist, 1);
			try_add_edge(current_node, neighbor, dist);
		}
	}
}

template <typename NodeType>
void KNNGraph<NodeType>::init(int k)
{	
	Graph<NodeType>::init();
	if (k <= 0)
		throw std::invalid_argument("K must be greater than zero.");
	this->k = k;
}

template <typename NodeType>
bool KNNGraph<NodeType>::add_edge(NodeType* node1, NodeType* node2, typename std::list<NodeType*>::iterator it)
{
	bool success = Graph<NodeType>::add_edge(node1, node2, it);
	if (success)
		hubness[node2]++;
	return success;
}

template <typename NodeType>
NodeType* KNNGraph<NodeType>::remove_last_edge(NodeType* node)
{
	NodeType* removed_node = Graph<NodeType>::remove_last_edge(node);
	dist_cache.remove_distance(node, removed_node);
	hubness[removed_node]--;
	return removed_node;
}

} // namespace knng
#endif