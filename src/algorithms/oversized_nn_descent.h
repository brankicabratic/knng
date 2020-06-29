#ifndef OVERSIZED_NN_DESCENT_H
#define OVERSIZED_NN_DESCENT_H

#include "nn_descent.h"

namespace knng
{

// Class that implements oversized nn-list nn-descent algorithm.
template <typename NodeType>
class OversizedNNDescent : public NNDescent<NodeType>
{
	int k;
	KNNGraph<NodeType>* reduced_knng = nullptr;

  public:
	OversizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling, int k2);
	OversizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling, int k2);
	~OversizedNNDescent();
	KNNGraph<NodeType>& get_knng() override; // Returns current KNN-Graph approximation.
	virtual std::string get_name() override; // Returns name of the algorithm.
};

template <typename NodeType>
OversizedNNDescent<NodeType>::OversizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int its_count, double sampling, int k2) : NNDescent<NodeType>(k2, dist, nodes, its_count, sampling), k(k) 
{
	if (k2 <= k)
		throw std::invalid_argument("OversizedNNDescent<NodeType>::OversizedNNDescent - k2 value must be strictly larger than k value.");
}

template <typename NodeType>
OversizedNNDescent<NodeType>::OversizedNNDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, double conv_ratio, double sampling, int k2) : NNDescent<NodeType>(k2, dist, nodes, conv_ratio, sampling), k(k)
{
	if (k2 <= k)
		throw std::invalid_argument("OversizedNNDescent<NodeType>::OversizedNNDescent - k2 value must be strictly larger than k value.");
}

template <typename NodeType>
OversizedNNDescent<NodeType>::~OversizedNNDescent()
{
	if (reduced_knng != nullptr)
		delete reduced_knng;
}

template <typename NodeType>
KNNGraph<NodeType>& OversizedNNDescent<NodeType>::get_knng()
{
	if (reduced_knng != nullptr)
		delete reduced_knng;

	reduced_knng = NNDescent<NodeType>::knng.get_reduced_to_k(k);
	return *reduced_knng;
}

template <typename NodeType>
std::string OversizedNNDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "osz-" << NNDescent<NodeType>::get_name();
	sout << "_k2" << k;
	return sout.str();
}

} // namespace knng
#endif