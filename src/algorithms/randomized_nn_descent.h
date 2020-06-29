#ifndef R_NN_DESCENT_H
#define R_NN_DESCENT_H

#include "nn_descent.h"

namespace knng
{

// Class that implements hubness aware variant of the nn-descent algorithm.
template <typename NodeType>
class RandomizedNNDescent : public NNDescent<NodeType>
{
	int iteration;
	int r;
	std::vector<NodeType*>& nodes;
	std::vector<NodeType*> antihubs;
	std::list<NodeType*> nodes_to_randomize;
	UniformWeights<NodeType> uniform_distribution;
	
  public:
	RandomizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling, int r);
	RandomizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling, int r);

	virtual std::string get_name() override; // Returns name of the algorithm.

  protected:
  	void before_iteration() override;
	NodeType* get_random_node();
};

template <typename NodeType>
RandomizedNNDescent<NodeType>::RandomizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling, int r) : NNDescent<NodeType>(k, dist, nodes, its_count, sampling), iteration(0), r(r), nodes(nodes)
{
	std::copy(nodes.begin(), nodes.end(), std::back_inserter(nodes_to_randomize));
}

template <typename NodeType>
RandomizedNNDescent<NodeType>::RandomizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling, int r) : NNDescent<NodeType>(k, dist, nodes, conv_ratio, sampling), iteration(0), r(r), nodes(nodes)
{
	std::copy(nodes.begin(), nodes.end(), std::back_inserter(nodes_to_randomize));
}

template <typename NodeType>
std::string RandomizedNNDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "rnd-" << NNDescent<NodeType>::get_name();
	sout << "_r" << r;
	return sout.str();
}

template <typename NodeType>
void RandomizedNNDescent<NodeType>::before_iteration()
{
	iteration++;
	if (iteration == 1) return;

	DistCache<NodeType>& dist_cache = this->knng.get_dist_cache();

	std::unordered_map<NodeType*, int> r_vals;
	std::unordered_map<NodeType*, int> updates;

	std::list<std::pair<NodeType*,NodeType*>> removed_edges;

	for (typename std::list<NodeType*>::iterator it = nodes_to_randomize.begin(); it != nodes_to_randomize.end(); it++)
	{
		NodeType* node = *it;
		
		int r_val;
		typename std::unordered_map<NodeType*, int>::iterator r_val_it = r_vals.find(node);
		if (r_val_it == r_vals.end())
			r_val = r;
		else
			r_val = r_val_it->second;

		typename std::unordered_map<NodeType*, int>::iterator updates_it = updates.find(node);
		if (updates_it == updates.end())
			updates_it = updates.insert({node, 0}).first;

		for (int i = 0; i < r_val; i++)
		{
			NodeType* random_node = get_random_node();
			while (node == random_node)
				random_node = get_random_node();

			typename std::unordered_map<NodeType*, int>::iterator rnd_node_r_val_it = r_vals.find(random_node);
			if (rnd_node_r_val_it == r_vals.end())
				r_vals.insert({random_node, r-1});
			else
				(rnd_node_r_val_it->second)--;

			double d = dist_cache.get_and_store_distance(node, random_node, 2);
			int added = 0;

			if (this->knng.try_add_edge(node, random_node, d, &removed_edges))
			{
				added++;
				this->nn[node][random_node] = true;
				(updates_it->second)++;
			}

			if (this->knng.try_add_edge(random_node, node, d, &removed_edges))
			{
				added++;
				this->nn[random_node][node] = true;
				typename std::unordered_map<NodeType*, int>::iterator rnd_node_updates_it = updates.find(random_node);
				if (rnd_node_updates_it == updates.end())
					updates.insert({random_node, 1});
				else
					(rnd_node_updates_it->second)++;
			}

			dist_cache.remove_distance(node, random_node, 2-added);
		}
	}

	for (auto& removed_edge: removed_edges)
	{
		this->nn[removed_edge.first].erase(removed_edge.second);
	}

	typename std::list<NodeType*>::iterator it = nodes_to_randomize.begin();
	while (it != nodes_to_randomize.end())
	{
		NodeType* node = *it;
		if (updates.find(node)->second == 0)
			it = nodes_to_randomize.erase(it);
		else
			it++;
	}
}

template <typename NodeType>
NodeType* RandomizedNNDescent<NodeType>::get_random_node()
{
	return nodes[rand() % nodes.size()];
}

} // namespace knng
#endif