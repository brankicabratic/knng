#ifndef HA_NN_DESCENT_H
#define HA_NN_DESCENT_H

#include "nn_descent.h"

namespace knng
{

// Class that implements hubness aware variant of the nn-descent algorithm.
template <typename NodeType>
class HANNDescent : public NNDescent<NodeType>
{
	int h_min, h_max;
	std::vector<NodeType*>& nodes;

  public:
	HANNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling, int h_min, int h_max);
	HANNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling, int h_min, int h_max);

	virtual std::string get_name() override; // Returns name of the algorithm.

  protected:
  	NodeType* replace_node(NodeType* node) override; // If the node is hub, it is replaced with some other, randomly chosen point.
	NodeType* get_random_node();
};

template <typename NodeType>
HANNDescent<NodeType>::HANNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling, int h_min, int h_max) : NNDescent<NodeType>(k, dist, nodes, its_count, sampling), h_min(h_min), h_max(h_max), nodes(nodes) { }

template <typename NodeType>
HANNDescent<NodeType>::HANNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling, int h_min, int h_max) : NNDescent<NodeType>(k, dist, nodes, conv_ratio, sampling), h_min(h_min), h_max(h_max), nodes(nodes) { }

template <typename NodeType>
std::string HANNDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "ha-" << NNDescent<NodeType>::get_name();
	sout << "_h_min" << h_min;
	sout << "_h_max" << h_max;
	return sout.str();
}

template <typename NodeType>
NodeType* HANNDescent<NodeType>::replace_node(NodeType* node)
{
	int h = this->knng.get_hubness(node);
	bool replace;
	if (h <= h_min)
	{
		replace = false;
	}
	else if (h >= h_max)
	{
		replace = true;
	}
	else
	{
		double prob = ((double)h - h_min) / (h_max - h_min);
		replace = rand() < prob * RAND_MAX;
	}

	if (replace) node = get_random_node();

	return node;
}

template <typename NodeType>
NodeType* HANNDescent<NodeType>::get_random_node()
{
	return nodes[rand() % nodes.size()];
}

} // namespace knng
#endif