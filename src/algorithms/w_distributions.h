#ifndef RW_DISTRIBUTIONS_H
#define RW_DISTRIBUTIONS_H

#include <list>
#include <unordered_map>

namespace knng
{

// Abstract class that models distribution of walks.
template <typename NodeType>
class WDistribution
{
  public:
	virtual int get_w_cnt(NodeType* node) = 0;                                       // Returns the number of walks for the given point.
	virtual void report_nn_updates(std::unordered_map<NodeType*, int> &updates) = 0; // During the rw-descent algorithm, this method is called in order to report the number of points' updates.
	virtual bool has_more_ws() = 0;                                                  // Returns true if there is at least one point that has assigned walks.
};

// Uniform walk distribution.
// This distribution assigns node_w_cnt walks to each point.
template <typename NodeType>
class UniformWDistribution : public WDistribution<NodeType>
{
	int node_w_cnt;

  public:
	UniformWDistribution(int node_w_cnt);
	int get_w_cnt(NodeType* node) override;
	void report_nn_updates(std::unordered_map<NodeType*, int> &updates) override;
	bool has_more_ws() override;
};

// Distribution that assigns walks only to the subset of dataset's nodes.
// Each node from this subset has equal number of walks.
template <typename NodeType>
class NodesSubsetDistribution : public WDistribution<NodeType>
{
	int node_w_cnt;
	std::list<NodeType*> nodes;

  public:
	NodesSubsetDistribution(std::list<NodeType*> nodes, int node_w_cnt);
	int get_w_cnt(NodeType* node) override;
	void report_nn_updates(std::unordered_map<NodeType*, int> &updates) override;
	bool has_more_ws() override;
};


// Simillarly to NodesSubsetDistribution, this distribution also assigns walks
// only to the subset of dataset's nodes. Each node from this subset has equal number of
// walks. But unlike NodesSubsetDistribution, this distribution expelles certain
// nodes during the time. The node is expelled when the recent updates counts are small
// enough. That actually means that the node converged, and hence it does not need walks.
template <typename NodeType>
class NodesSubsetDistributionWithConvergence : public WDistribution<NodeType>
{
	const int history_depth = 3;
	int node_w_cnt;
	double node_updates_treshold;
	std::list<NodeType*> nodes;
	std::list<std::list<int> > updates_history;

  public:
	NodesSubsetDistributionWithConvergence(std::list<NodeType*> nodes, int node_w_cnt, double conv_ratio = 0.05);
	int get_w_cnt(NodeType* node) override;
	void report_nn_updates(std::unordered_map<NodeType*, int> &updates) override;
	bool has_more_ws() override;
};

// UniformWDistribution class implementation.

template <typename NodeType>
UniformWDistribution<NodeType>::UniformWDistribution(int node_w_cnt) : node_w_cnt(node_w_cnt) { }

template <typename NodeType>
int UniformWDistribution<NodeType>::get_w_cnt(NodeType* node)
{
	return node_w_cnt;
}

template <typename NodeType>
void UniformWDistribution<NodeType>::report_nn_updates(std::unordered_map<NodeType*, int> &updates) { }

template <typename NodeType>
bool UniformWDistribution<NodeType>::has_more_ws()
{
	return true;
}

// NodesSubsetDistribution class implementation.

template <typename NodeType>
NodesSubsetDistribution<NodeType>::NodesSubsetDistribution(std::list<NodeType*> nodes, int node_w_cnt) : nodes(nodes), node_w_cnt(node_w_cnt) { }

template <typename NodeType>
int NodesSubsetDistribution<NodeType>::get_w_cnt(NodeType* node)
{
	if (std::find(nodes.begin(), nodes.end(), node) != nodes.end())
		return node_w_cnt;
	return 0;
}

template <typename NodeType>
void NodesSubsetDistribution<NodeType>::report_nn_updates(std::unordered_map<NodeType*, int> &updates) { }

template <typename NodeType>
bool NodesSubsetDistribution<NodeType>::has_more_ws()
{
	return true;
}

// NodesSubsetDistributionWithConvergence class implementation.

template <typename NodeType>
NodesSubsetDistributionWithConvergence<NodeType>::NodesSubsetDistributionWithConvergence(std::list<NodeType*> nodes, int node_w_cnt, double conv_ratio) : nodes(nodes), node_w_cnt(node_w_cnt), node_updates_treshold(conv_ratio*node_w_cnt), updates_history(nodes.size()) { }

template <typename NodeType>
int NodesSubsetDistributionWithConvergence<NodeType>::get_w_cnt(NodeType* node)
{
	if (std::find(nodes.begin(), nodes.end(), node) != nodes.end())
		return node_w_cnt;
	return 0;
}

template <typename NodeType>
void NodesSubsetDistributionWithConvergence<NodeType>::report_nn_updates(std::unordered_map<NodeType*, int> &updates)
{
	typename std::list<NodeType*>::iterator nodes_it = nodes.begin();
	typename std::list<std::list<int> >::iterator updates_history_it = updates_history.begin();
	while (nodes_it != nodes.end())
	{
		updates_history_it->push_back(updates[*nodes_it]);
		if (updates_history_it->size() > history_depth)
			updates_history_it->pop_front();
		int total_updates = 0;
		std::list<int>::iterator updates_history_items_it = updates_history_it->begin();
		while (updates_history_items_it != updates_history_it->end())
		{
			total_updates += *updates_history_items_it;
			updates_history_items_it++;
		}
		double avg_updates = (double)total_updates / updates_history_it->size();
		if (avg_updates < node_updates_treshold)
		{
			nodes.erase(nodes_it++);
			updates_history.erase(updates_history_it++);
		}
		else
		{
			nodes_it++;
			updates_history_it++;
		}
	}
}

template <typename NodeType>
bool NodesSubsetDistributionWithConvergence<NodeType>::has_more_ws()
{
	return nodes.size() > 0;
}

} // namespace knng
#endif