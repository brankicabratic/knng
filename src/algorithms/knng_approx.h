#ifndef KNNG_APPROX_H
#define KNNG_APPROX_H

#include "../graph/knngraph.h"

namespace knng
{

// Interface that should be extended by each class that implements some algorithm for creation KNN graph approximation.
template <typename NodeType>
class KNNGApproximation : public StringConverter
{
  public:
	virtual void generate_knng() = 0;           								// Creates KNN-Graph approximation.
	virtual void update_knng(std::unordered_set<NodeType*> &changed_nodes) = 0; // Updates KNN-Graph approximation.
	virtual KNNGraph<NodeType>& get_knng() = 0; 								// Returns current KNN-Graph approximation.
	virtual std::string get_name() = 0;											// Returns name of the algorithm.
};

} // namespace knng
#endif