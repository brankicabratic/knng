#ifndef NAIVE_KNNG_H
#define NAIVE_KNNG_H

#include "../data_objects/dist.h"
#include "../exceptions/not_implemented_exception.h"
#include "knng_approx.h"

namespace knng
{

// Class that implements nn-descent algorithm.
template <typename NodeType>
class NaiveKNNGraph : public KNNGApproximation<NodeType>
{
  protected:
	KNNGraph<NodeType> knng;

  public:
	NaiveKNNGraph(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes);
	void generate_knng() override;                   						 // Creates KNN-Graph approximation.
	void update_knng(std::unordered_set<NodeType*> &changed_nodes) override; // Updates KNN-Graph approximation by recreating it from scratch.
	KNNGraph<NodeType>& get_knng() override; 						 		 // Returns current KNN-Graph approximation.
	std::string get_name() override; 										 // Returns name of the algorithm.

	// StringConverter overrides
	std::string to_string(bool brief = true) override;
	void from_string(std::string str, bool brief = true) override;
};

template <typename NodeType>
NaiveKNNGraph<NodeType>::NaiveKNNGraph(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes) : knng(k, dist, nodes) { }

template <typename NodeType>
void NaiveKNNGraph<NodeType>::generate_knng()
{
	knng.generate();
}

template <typename NodeType>
void NaiveKNNGraph<NodeType>::update_knng(std::unordered_set<NodeType*> &changed_nodes)
{
	knng.regenerate(changed_nodes);
}

template <typename NodeType>
KNNGraph<NodeType>& NaiveKNNGraph<NodeType>::get_knng()
{
	return knng;
}

template <typename NodeType>
std::string NaiveKNNGraph<NodeType>::get_name()
{
	return "knng";
}

template <typename NodeType>
std::string NaiveKNNGraph<NodeType>::to_string(bool brief)
{
	return knng.to_string(brief);
}

template <typename NodeType>
void NaiveKNNGraph<NodeType>::from_string(std::string str, bool brief)
{
	knng.from_string(str, brief);
}

} // namespace knng
#endif