#ifndef NN_DESCENT_H
#define NN_DESCENT_H

#include "../data_objects/dist.h"
#include "../exceptions/not_implemented_exception.h"
#include "knng_approx.h"

namespace knng
{

// Class that implements nn-descent algorithm.
template <typename NodeType>
class NNDescent : public KNNGApproximation<NodeType>
{
	// Enumerates algorithms for determination when the nn-descent should stop updating KNN-Graph approximation.
	enum class TerminationCriteria
	{
		FixedIterationsCount, // nn-descent stops after fixed number of iterations.
		Convergence           // nn-descent stops on convergence (see conv_ratio explanation for more details)
	};
	
  protected:
	TerminationCriteria term_criteria;                                      // Termination criteria.
	int its_count;                                                          // Number of iterations.
	double conv_ratio;                                                      // Convergence ratio. If there are less than conv_ratio*N*K updates of NN lists in current iteration, the algorithm terminates.
	int sampling;                                                           // The count of neighbors to use in local joins.
	KNNGraph<NodeType> knng;                                                // Current KNN-Graph approximation.
	std::unordered_map<NodeType*, std::unordered_map<NodeType*, bool>> nn;  // Current NN-lists with the boolean value that is equal to true if the node is newly added to NN-list, and false otherwise.

  public:
	NNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling);
	NNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling);
	void generate_knng() override;                   						 // Creates KNN-Graph approximation.
	void update_knng(std::unordered_set<NodeType*> &changed_nodes) override; // Updates KNN-Graph approximation by recreating it from scratch.
	virtual KNNGraph<NodeType>& get_knng() override; 						 // Returns current KNN-Graph approximation.
	virtual std::string get_name() override;								 // Returns name of the algorithm.

	// StringConverter overrides
	std::string to_string(bool brief = true) override;
	void from_string(std::string str, bool brief = true) override;

  protected:
  	void init_nn();                                 // Initialize nn list.
	int improve_knng();                             // Improves current KNN-Graph approximation by applying one iteration of the algorithm. Returns number of updates of nodes' nn lists.
	virtual void before_iteration();                // Called before each NN-Descent iteration.
	virtual void after_iteration();                 // Called after each NN-Descent iteration.
	virtual NodeType* replace_node(NodeType* node); // Used in subclass (for NNDescent variants). Method should replace given node with some other node that is more appropriate.
	int local_joins(NodeType* node, std::unordered_map<NodeType*, bool>& edges, std::unordered_map<NodeType*, bool>& r_edges, std::list<std::pair<NodeType*,NodeType*>>& removed_edges);
	void output_nn(); // For debug purposes during development.
};

template <typename NodeType>
NNDescent<NodeType>::NNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling) : knng(k, dist, nodes), its_count(its_count)
{
	if (sampling <= 0 || sampling > 1)
		throw std::invalid_argument("NNDescent<NodeType>::NNDescent - Sampling value must be a number in range (0,1].");
	this->sampling = (int)round(sampling*k);
	if (this->sampling == 0) this->sampling = 1;
	term_criteria = TerminationCriteria::FixedIterationsCount;
}

template <typename NodeType>
NNDescent<NodeType>::NNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling) : knng(k, dist, nodes), conv_ratio(conv_ratio), sampling(sampling)
{
	if (sampling <= 0 || sampling > 1)
		throw std::invalid_argument("NNDescent<NodeType>::NNDescent - Sampling value must be a number in range (0,1].");
	this->sampling = (int)round(sampling*k);
	if (this->sampling == 0) this->sampling = 1;
	term_criteria = TerminationCriteria::Convergence;
}

template <typename NodeType>
void NNDescent<NodeType>::generate_knng()
{
	knng.init_randomly();
	init_nn();
	int updates_cnt;
	double updates_limit = knng.nodes_count() * knng.get_k() * conv_ratio;
	int i = 1;
	do
	{
		before_iteration();
		updates_cnt = improve_knng();
		after_iteration();
		i++;
	} while (term_criteria == TerminationCriteria::FixedIterationsCount && i <= its_count || term_criteria == TerminationCriteria::Convergence && updates_cnt > updates_limit);
}

template <typename NodeType>
void NNDescent<NodeType>::update_knng(std::unordered_set<NodeType*> &changed_nodes)
{
	generate_knng();
}

template <typename NodeType>
KNNGraph<NodeType>& NNDescent<NodeType>::get_knng()
{
	return knng;
}

template <typename NodeType>
std::string NNDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "nndes";
	if (term_criteria == NNDescent<NodeType>::TerminationCriteria::FixedIterationsCount)
		sout << "_tc-fixed-it-" << its_count;
	else
		sout << "_tc-conv-" << conv_ratio;
	sout << "_sampl-" << sampling;
	return sout.str();
}

template <typename NodeType>
std::string NNDescent<NodeType>::to_string(bool brief)
{
	std::stringstream sout;
	if (!brief)
		sout << "======== KNN-Graph ========" << std::endl;
	sout << knng.to_string(brief);
	return sout.str();
}

template <typename NodeType>
void NNDescent<NodeType>::from_string(std::string str, bool brief)
{
	if (!brief) throw NotImplementedException();
	
	knng.from_string(str);
}

template <typename NodeType>
void NNDescent<NodeType>::init_nn()
{
	nn.clear();
	knng.reset_iterator();
	std::pair<NodeType*, std::list<NodeType*>*> node;
	while ((node = knng.next_node_with_edges()).first != nullptr)
	{
		typename std::unordered_map<NodeType*, std::unordered_map<NodeType*, bool>>::iterator it_nn = nn.insert({node.first, std::unordered_map<NodeType*, bool>()}).first;
		for (typename std::list<NodeType*>::iterator it = node.second->begin(); it != node.second->end(); it++)
			it_nn->second.insert({*it, true});
	}
}

template <typename NodeType>
int NNDescent<NodeType>::improve_knng()
{
	int updates_cnt = 0;
	std::unordered_map<NodeType*, std::unordered_map<NodeType*, bool>> lj_nn;  // NN lists for local joins
	std::unordered_map<NodeType*, std::unordered_map<NodeType*, bool>> lj_rnn; // RNN lists for local joins
	for (auto& node_edges_pair: nn)
	{
		typename std::unordered_map<NodeType*, std::unordered_map<NodeType*, bool>>::iterator new_entry = lj_nn.insert({node_edges_pair.first, std::unordered_map<NodeType*, bool>()}).first;
		std::list<NodeType*> new_nodes_list;
		for (auto& edge: node_edges_pair.second)
		{
			if (edge.second)
			{
				new_nodes_list.push_back(edge.first);
			}
			else
			{
				new_entry->second.insert(edge);
				typename std::unordered_map<NodeType*,std::unordered_map<NodeType*,bool>>::iterator rr_entry = lj_rnn.find(edge.first);
				if (rr_entry == lj_rnn.end())
					rr_entry = lj_rnn.insert({edge.first, std::unordered_map<NodeType*,bool>()}).first;
				rr_entry->second.insert({node_edges_pair.first,false});
			}
		}
		
		if (new_nodes_list.size() > sampling)
		{
			std::unordered_set<int> rand_nodes_set = rand_range(0, new_nodes_list.size()-1, new_nodes_list.size() - sampling);
			std::vector<int> rand_nodes(rand_nodes_set.begin(), rand_nodes_set.end());
			std::sort(rand_nodes.begin(), rand_nodes.end(), std::greater<int>());
			for (auto& index: rand_nodes)
			{
				typename std::list<NodeType*>::iterator it = new_nodes_list.begin();
				advance(it, index);
				new_nodes_list.erase(it);
			}
		}

		for (auto& node: new_nodes_list)
		{
			new_entry->second.insert({node, true});
			node_edges_pair.second[node] = false;
			typename std::unordered_map<NodeType*,std::unordered_map<NodeType*,bool>>::iterator rr_entry = lj_rnn.find(node);
			if (rr_entry == lj_rnn.end())
				rr_entry = lj_rnn.insert({node, std::unordered_map<NodeType*,bool>()}).first;
			rr_entry->second.insert({node_edges_pair.first,true});
		}
	}

	std::list<std::pair<NodeType*,NodeType*>> removed_edges;

	for (auto& node_edges_pair: lj_nn)
	{
		NodeType* node = node_edges_pair.first;
		std::unordered_map<NodeType*, bool>& edges = node_edges_pair.second;
		
		std::unordered_map<NodeType*, bool> r_edges = lj_rnn.find(node) != lj_rnn.end() ? lj_rnn.find(node)->second : std::unordered_map<NodeType*, bool>();
		
		std::vector<NodeType*> r_edges_nodes_new;
		std::vector<NodeType*> r_edges_nodes_old;
		for (auto r_edges_pair: r_edges)
			if (r_edges_pair.second)
				r_edges_nodes_new.push_back(r_edges_pair.first);
			else
				r_edges_nodes_old.push_back(r_edges_pair.first);
		
		if (r_edges_nodes_new.size() > sampling)
		{
			std::unordered_set<int> rand_new_nodes_set = rand_range(0, r_edges_nodes_new.size()-1, r_edges_nodes_new.size()-sampling);
			for (auto& index: rand_new_nodes_set)
				r_edges.erase(r_edges_nodes_new[index]);
		}

		if (r_edges_nodes_old.size() > sampling)
		{
			std::unordered_set<int> rand_old_nodes_set = rand_range(0, r_edges_nodes_old.size()-1, r_edges_nodes_old.size()-sampling);
			for (auto& index: rand_old_nodes_set)
				r_edges.erase(r_edges_nodes_old[index]);
		}

		updates_cnt += local_joins(node, edges, r_edges, removed_edges);
	}

	for (auto& removed_edge: removed_edges)
	{
		nn[removed_edge.first].erase(removed_edge.second);
	}

	return updates_cnt;
}

template <typename NodeType>
void NNDescent<NodeType>::before_iteration() { }

template <typename NodeType>
void NNDescent<NodeType>::after_iteration() { }

template <typename NodeType>
NodeType* NNDescent<NodeType>::replace_node(NodeType* node)
{
	return node;
}

template <typename NodeType>
int NNDescent<NodeType>::local_joins(NodeType* node, std::unordered_map<NodeType*, bool>& edges, std::unordered_map<NodeType*, bool>& r_edges, std::list<std::pair<NodeType*,NodeType*>>& removed_edges)
{
	DistCache<NodeType>& dist_cache = this->knng.get_dist_cache();
	int updates_cnt = 0;
	for (typename std::unordered_map<NodeType*, bool>::iterator edge_it = edges.begin(); edge_it != edges.end(); edge_it++)
	{
		NodeType* node1 = replace_node(edge_it->first);
		bool is_new = edge_it->second;
		typename std::unordered_map<NodeType*, bool>::iterator edge_it_2 = edge_it;
		edge_it_2++;
		while (edge_it_2 != edges.end())
		{
			bool is_new_2 = edge_it_2->second;
			if (is_new || is_new_2)
			{
				NodeType* node2 = replace_node(edge_it_2->first);
				if (node1 == node2) 
				{
					edge_it_2++;
					continue;
				}
				int old_updates_cnt = updates_cnt;
				double d = dist_cache.get_and_store_distance(node1, node2, 2);
				if (knng.try_add_edge(node1, node2, d, &removed_edges))
				{
					nn[node1][node2] = true;					
					updates_cnt++;
				}
				if (knng.try_add_edge(node2, node1, d, &removed_edges))
				{
					nn[node2][node1] = true;
					updates_cnt++;
				}
				dist_cache.remove_distance(node1, node2, 2-updates_cnt+old_updates_cnt);
			}
			edge_it_2++;
		}

		for (typename std::unordered_map<NodeType*,bool>::iterator r_edge_it = r_edges.begin(); r_edge_it != r_edges.end(); r_edge_it++)
		{
			bool is_new_2 = r_edge_it->second;
			if (is_new || is_new_2)
			{
				NodeType* node2 = replace_node(r_edge_it->first);
				if (node1 == node2) continue;
				int old_updates_cnt = updates_cnt;
				double d = dist_cache.get_and_store_distance(node1, node2, 2);
				if (knng.try_add_edge(node1, node2, d, &removed_edges))
				{
					nn[node1][node2] = true;
					updates_cnt++;
				}
				if (knng.try_add_edge(node2, node1, d, &removed_edges))
				{
					nn[node2][node1] = true;
					updates_cnt++;
				}
				dist_cache.remove_distance(node1, node2, 2-updates_cnt+old_updates_cnt);
			}
		}
	}

	for (typename std::unordered_map<NodeType*,bool>::iterator r_edge_it = r_edges.begin(); r_edge_it != r_edges.end(); r_edge_it++)
	{
		bool is_new = r_edge_it->second;
		NodeType* node1 = replace_node(r_edge_it->first);
		typename std::unordered_map<NodeType*,bool>::iterator r_edge_it_2 = r_edge_it;
		r_edge_it_2++;
		while (r_edge_it_2 != r_edges.end())
		{
			bool is_new_2 = r_edge_it_2->second;
			if (is_new || is_new_2)
			{
				NodeType* node2 = replace_node(r_edge_it_2->first);
				if (node1 == node2)
				{
					r_edge_it_2++;
					continue;
				}
				int old_updates_cnt = updates_cnt;
				double d = dist_cache.get_and_store_distance(node1, node2, 2);
				if (knng.try_add_edge(node1, node2, d, &removed_edges))
				{
					nn[node1][node2] = true;
					updates_cnt++;
				}
				if (knng.try_add_edge(node2, node1, d, &removed_edges))
				{
					nn[node2][node1] = true;
					updates_cnt++;
				}
				dist_cache.remove_distance(node1, node2, 2-updates_cnt+old_updates_cnt);
			}
			r_edge_it_2++;
		}
	}

	return updates_cnt;
}

template <typename NodeType>
void NNDescent<NodeType>::output_nn()
{
	for (auto& node_edges: nn)
	{
		std::cout << ((StringConverter*)node_edges.first)->to_string() << ":";
		for (auto& edge: node_edges.second)
		{
			std::cout << " " << ((StringConverter*)edge.first)->to_string() << "(" << (edge.second ? "new" : "old") << ")";
		}
		std::cout << std::endl;
	}
}


} // namespace knng
#endif