#ifndef NW_DESCENT_H
#define NW_DESCENT_H

#include <cmath>
#include <queue>
#include "w_descent.h"

namespace knng
{

// Class that implements nw-descent algorithm.
template <typename NodeType>
class NWDescent : public WDescent<NodeType>
{  
	struct NodeComp
	{
		std::unordered_map<NodeType*, double> &angles;

		NodeComp(std::unordered_map<NodeType*, double> &angles) : angles(angles) { }

		bool operator() (NodeType* lhs, NodeType* rhs)
		{
			return angles[lhs] < angles[rhs];
		}
	};

  private:
	std::unordered_map<NodeType*, double> min_angles;
	int new_neighbors_addition_map = 0;
	std::unordered_map<NodeType*, std::unordered_set<NodeType*>> new_neighbors_1;
	std::unordered_map<NodeType*, std::unordered_set<NodeType*>> new_neighbors_2;

  public:
	NWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int its_count);
	NWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int max_its_count, double conv_ratio);
	std::string get_name() override; // Returns name of the algorithm.

  protected:
	std::list<NodeType*> get_nodes_for_comparison(NodeType* node, int count) override;
	void on_initial_graph_initialized() override;
	void on_graph_changed() override;
	void on_iteration_changed() override;
	void on_edges_added(NodeType* node1, NodeType* node2, int edges_count) override;
	void on_edges_removed(std::list<std::pair<NodeType*, NodeType*> > &removed_edges) override { }

  private:
	void update_angles(DistCache<NodeType>& dist_cache, double dist1, double max_dist, double min_angle, bool is_new, NodeType* n1, NodeType* n2, std::unordered_map<NodeType*, double> &angles);
	void reset();
};

// NWDescent class implementation.

template <typename NodeType>
NWDescent<NodeType>::NWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int its_count) : WDescent<NodeType>(k, dist, nodes, rws_count, rand_count, its_count) { }

template <typename NodeType>
NWDescent<NodeType>::NWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int max_its_count, double conv_ratio) : WDescent<NodeType>(k, dist, nodes, rws_count, rand_count, max_its_count, conv_ratio) { }

template <typename NodeType>
std::string NWDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "n" << WDescent<NodeType>::get_name();
	return sout.str();
}

template <typename NodeType>
std::list<NodeType*> NWDescent<NodeType>::get_nodes_for_comparison(NodeType* node, int count)
{
	DistCache<NodeType>& dist_cache = this->knng.get_dist_cache();
	std::unordered_set<NodeType*>* new_neighbors = new_neighbors_addition_map == 0 ? &(new_neighbors_2[node]) : &(new_neighbors_1[node]);
	std::list<NodeType*> nn_list = this->knng.get_edges(node);
	std::unordered_set<NodeType*> neighbors;
	for (NodeType* n: nn_list)
	{
		neighbors.insert(n);
	}
	for (NodeType* n: this->knng.get_r_edges(node))
	{
		neighbors.insert(n);
	}
	double max_dist = dist_cache.get_distance(node, nn_list.back());
	double min_angle = min_angles[node];	
	std::unordered_map<NodeType*, double> angles;
	for (NodeType* n: neighbors)
	{
		bool is_new = new_neighbors->find(n) != new_neighbors->end();
		std::unordered_set<NodeType*>* new_neighbors_n = new_neighbors_addition_map == 0 ? &(new_neighbors_2[n]) : &(new_neighbors_1[n]);
		double dist = dist_cache.get_distance(node, n);
		for (NodeType* n_n: this->knng.get_edges(n))
		{
			if (n_n == node || neighbors.find(n_n) != neighbors.end()) continue;
			update_angles(dist_cache, dist, max_dist, min_angle, is_new || new_neighbors_n->find(n_n) != new_neighbors_n->end(), n, n_n, angles);
		}
		for (NodeType* n_n: this->knng.get_r_edges(n))
		{
			if (n_n == node || neighbors.find(n_n) != neighbors.end()) continue;
			update_angles(dist_cache, dist, max_dist, min_angle, is_new || new_neighbors_n->find(n_n) != new_neighbors_n->end(), n, n_n, angles);
		}
	}

	std::vector<NodeType*> candidates;
	for (typename std::unordered_map<NodeType*, double>::iterator it = angles.begin(); it != angles.end(); ++it)
	{
		candidates.push_back(it->first);
	}
	std::priority_queue<NodeType*, std::vector<NodeType*>, NodeComp> pq(NodeComp(angles), candidates);
	std::list<NodeType*> nodes;
	while (pq.size() > 0 && nodes.size() < count)
	{
		NodeType* top = pq.top(); pq.pop();			
		min_angle = angles[top];
		nodes.push_back(top);
	}
	min_angles[node] = min_angle;

	return nodes;
}

template <typename NodeType>
void NWDescent<NodeType>::on_initial_graph_initialized()
{
	reset();
}

template <typename NodeType>
void NWDescent<NodeType>::on_graph_changed()
{
	reset();
}

template <typename NodeType>
void NWDescent<NodeType>::on_iteration_changed()
{
	new_neighbors_addition_map = (new_neighbors_addition_map + 1) % 2;
	std::unordered_map<NodeType*, std::unordered_set<NodeType*>>* new_neighbors = new_neighbors_addition_map == 0 ? &new_neighbors_1 : &new_neighbors_2;
	for (auto &entry: *new_neighbors) {
		entry.second.clear();
	}	
}

template <typename NodeType>
void NWDescent<NodeType>::on_edges_added(NodeType* node1, NodeType* node2, int edges_count)
{
	if (edges_count == 0) return;
	std::unordered_map<NodeType*, std::unordered_set<NodeType*>>* new_neighbors = new_neighbors_addition_map == 0 ? &new_neighbors_1 : &new_neighbors_2;
	(*new_neighbors)[node1].insert(node2);
	(*new_neighbors)[node2].insert(node1);
}

template <typename NodeType>
void NWDescent<NodeType>::update_angles(DistCache<NodeType>& dist_cache, double dist1, double max_dist, double min_angle, bool is_new, NodeType* n1, NodeType* n2, std::unordered_map<NodeType*, double> &angles)
{
	double dist2 = dist_cache.get_distance(n1, n2);
	if (dist2 - dist1 >= max_dist || dist1 - dist2 >= max_dist) return;

	double angle = 0;
	if (dist1 + dist2 < max_dist)
		angle = 3.1416;
	else
		angle = std::acos((dist1*dist1 + dist2*dist2 - max_dist*max_dist) / (2*dist1*dist2));

	if (angle == 0 || ((angle > min_angle) && !is_new) || (angles.find(n2) != angles.end() && angles[n2] >= angle))		
		return;

	angles[n2] = angle;
}

template <typename NodeType>
void NWDescent<NodeType>::reset()
{
	this->knng.reset_iterator();
	NodeType* node;
	while ((node = this->knng.next_node()) != nullptr)
	{
		min_angles[node] = 3.1416;
		new_neighbors_1.emplace(node, std::unordered_set<NodeType*>());
		new_neighbors_2.emplace(node, std::unordered_set<NodeType*>());
	}
}

} // namespace knng
#endif