#ifndef RW_DESCENT_H
#define RW_DESCENT_H

#include "w_descent.h"

namespace knng
{

// Edge maturity graph weights. Edge has greater weight
// if it was added to the graph more recently.
template <typename NodeType>
class EdgeMaturityWeights : public GraphWeights<NodeType>
{
	// Type definition for the hash map that will keep data about moments in time when the edges are added in the graph.
	// The edge weight is then a function of the time of edge addition.
	// Key:   Edge
	// Value: Pair of integers. First integer represents the moment when the edge is added in the graph.
	//        The second one represents the number of links that depend on this map entry. Namely,
	//        if the number of dependencies becomes zero, edges's information is no longer needed, in
	//        which case this map entry is removed.
	typedef std::unordered_map<std::pair<NodeType*, NodeType*>, std::pair<int, int>, PointersPairHash<NodeType>, PointersPairEqual<NodeType> > iterations_map;
	int current_iteration;     // Current interation. Could also be viewed as a current moment in time.
	iterations_map iterations; // Map containing the data.

  public:
	void set_current_iteration(int i);                                 // Sets the current iteration.
	void add_pair(NodeType* node1, NodeType* node2, int dependencies); // Adds an edge. Time of this edge's addition becomes current_iteration.
	void remove_dependency(NodeType* node1, NodeType* node2);          // Removes a dependency of given edge.
	void clear();                                                      // Removes all weights.

	// GraphWeights<NodeType> overrides
	double get_weight(NodeType* node1, NodeType* node2) override;
	void from_string(std::vector<NodeType*> &nodes, std::string str) override;

	// StringConverter overrides
	std::string to_string(bool brief = true) override;
};

// Class that implements rw-descent algorithm.
template <typename NodeType>
class RWDescent : public WDescent<NodeType>
{
  public:
  	// Enumerates algorithms for calculating edge traversal probabilities during the random walks.
	// RWPr stands for Random Walk PRobabilities.
	enum class RWPr
	{
		Uniform,     // All the edges always have the same traversal probability.
		EdgeMaturity // The edge has a higher traversal probability if it was added to the graph more recently.
	};

  protected:
  	RWPr rw_pr;                              // Algorithm for edges' traversal probabilities assignments during the random walks.
	GraphWeights<NodeType>* graph_weights;   // Graph weights used for edges traversal probabilities during the random walks. The edge of greater weight has a higher probability to be traversed.
	Graph<NodeType>* undirected_graph = nullptr;

  public:
	RWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int its_count, RWPr rw_pr = RWPr::Uniform);
	RWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int max_its_count, double conv_ratio, RWPr rw_pr = RWPr::Uniform);
	~RWDescent();

	virtual std::string get_name() override; // Returns name of the algorithm.
	
	// StringConverter overrides
	virtual std::string to_string(bool brief = true) override;
	virtual void from_string(std::string str, bool brief = true) override;

  protected:
  	void init(); // Initialization on the object creation.
	std::list<NodeType*> get_nodes_for_comparison(NodeType* node, int count) override;
	void on_initial_graph_initialized() override;
	void on_graph_changed() override;
	void on_iteration_changed() override;
	void on_edges_added(NodeType* node1, NodeType* node2, int edges_count) override;
	void on_edges_removed(std::list<std::pair<NodeType*, NodeType*> > &removed_edges) override;

  private:
	void update_undirected_graph();
	void reset();
};

// EdgeMaturityWeights class implementation.

template <typename NodeType>
void EdgeMaturityWeights<NodeType>::set_current_iteration(int i)
{
	current_iteration = i;
}

template <typename NodeType>
void EdgeMaturityWeights<NodeType>::add_pair(NodeType* node1, NodeType* node2, int dependencies)
{
	if (dependencies <= 0) return;
	typename iterations_map::iterator entry = iterations.find(std::make_pair(node1, node2));
	if (entry == iterations.end())
		iterations[std::make_pair(node1, node2)] = std::make_pair(current_iteration, dependencies);
	else
		entry->second.second += dependencies;	
}

template <typename NodeType>
void EdgeMaturityWeights<NodeType>::remove_dependency(NodeType* node1, NodeType* node2)
{
	typename iterations_map::iterator entry = iterations.find(std::make_pair(node1, node2));
	if (entry == iterations.end())
		return;
	entry->second.second--;
	if (entry->second.second == 0)
		iterations.erase(entry);
}

template <typename NodeType>
void EdgeMaturityWeights<NodeType>::clear()
{
	iterations.clear();
}

template <typename NodeType>
double EdgeMaturityWeights<NodeType>::get_weight(NodeType* node1, NodeType* node2)
{
	typename iterations_map::iterator entry = iterations.find(std::make_pair(node1, node2));
	if (entry == iterations.end())
	{
		std::cout << to_string();
		throw std::invalid_argument("EdgeMaturityWeights<NodeType>::get_weight - Given edge does not have probability.");
	}
	return 1.0 / (current_iteration - entry->second.first);
}

template <typename NodeType>
void EdgeMaturityWeights<NodeType>::from_string(std::vector<NodeType*> &nodes, std::string str)
{
	iterations.clear();

	std::stringstream s_all(str);
	std::string line;
	while (std::getline(s_all, line))
	{
		std::stringstream s_line(line);
		int id1, id2, it, dep;
		s_line >> id1 >> id2 >> it >> dep;
		NodeType* n1 = nullptr;
		NodeType* n2 = nullptr;
		typename std::vector<NodeType*>::iterator nodes_it = nodes.begin();
		while (nodes_it != nodes.end())
		{
			int current_id = ((Identifiable*)*nodes_it)->get_id();
			if (current_id == id1)
				n1 = *nodes_it;
			else if (current_id == id2)
				n2 = *nodes_it;
			
			if (n1 != nullptr && n2 != nullptr)
				break;
		}
		
		if (n1 != nullptr || n2 != nullptr)
			throw std::invalid_argument("EdgeMaturityWeights<NodeType>::from_string - Node set does not contain saved id.");
		
		iterations[std::make_pair(n1, n2)] = std::make_pair(it, dep);
	}
}

template <typename NodeType>
std::string EdgeMaturityWeights<NodeType>::to_string(bool brief)
{
	std::stringstream sout;
	typename iterations_map::iterator it = iterations.begin();
	while (it != iterations.end())
	{
		if (brief)
		{
			sout << ((Identifiable*)it->first.first)->get_id() << " ";
			sout << ((Identifiable*)it->first.second)->get_id() << " ";
			sout << it->second.first << " ";
			sout << it->second.second;
		}
		else
		{
			sout << "(";
			sout << ((StringConverter*)it->first.first)->to_string();
			sout << ",";
			sout << ((StringConverter*)it->first.second)->to_string();
			sout << ") -> it=" << it->second.first << ", depend=" << it->second.second;
		}		
		sout << std::endl;
		it++;
	}
	return sout.str();
}

// RWDescent class implementation.

template <typename NodeType>
RWDescent<NodeType>::RWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int its_count, RWPr rw_pr) : WDescent<NodeType>(k, dist, nodes, rws_count, rand_count, its_count), rw_pr(rw_pr)
{
	init();
}

template <typename NodeType>
RWDescent<NodeType>::RWDescent(int k, Dist<NodeType> &dist, std::vector<NodeType*>& nodes, int rws_count, int rand_count, int max_its_count, double conv_ratio, RWPr rw_pr) : WDescent<NodeType>(k, dist, nodes, rws_count, rand_count, max_its_count, conv_ratio), rw_pr(rw_pr)
{
	init();
}

template <typename NodeType>
RWDescent<NodeType>::~RWDescent()
{
	if (graph_weights != nullptr)
		delete graph_weights;
	if (undirected_graph != nullptr)
		delete undirected_graph;
}

template <typename NodeType>
std::string RWDescent<NodeType>::get_name()
{
	std::stringstream sout;
	sout << "r" << WDescent<NodeType>::get_name();
	sout << "_" << (rw_pr == RWDescent<NodeType>::RWPr::Uniform ? "rwpr-uniform" : "rwpr-em");
	std::string path = sout.str();	
	return path;
}

template <typename NodeType>
std::string RWDescent<NodeType>::to_string(bool brief)
{
	std::stringstream sout;
	sout << WDescent<NodeType>::to_string(brief);
	if (brief)
		sout << "-" << std::endl;
	if (!brief)
		sout << "======== Graph weights ========" << std::endl;
	sout << graph_weights->to_string(brief);
	return sout.str();
}

template <typename NodeType>
void RWDescent<NodeType>::from_string(std::string str, bool brief)
{
	if (!brief) throw NotImplementedException();
	
	std::string line;
	std::stringstream s_in(str);
	
	std::stringstream s_wdes;
	while (std::getline(s_in, line) && line != "-")
	{
		s_wdes << line << std::endl;
	}
	WDescent<NodeType>::from_string(s_wdes.str(), brief);

	std::stringstream s_gws;
	while (std::getline(s_in, line))
	{
		s_gws << line << std::endl;
	}
	std::vector<NodeType*> nodes = this->knng.get_sorted_nodes();
	graph_weights->from_string(nodes, s_gws.str());
}

template <typename NodeType>
void RWDescent<NodeType>::init()
{
	if (rw_pr == RWPr::Uniform)
		graph_weights = new UniformWeights<NodeType>;
	else if (rw_pr == RWPr::EdgeMaturity)
		graph_weights = new EdgeMaturityWeights<NodeType>;
}

template <typename NodeType>
std::list<NodeType*> RWDescent<NodeType>::get_nodes_for_comparison(NodeType* node, int count)
{
	std::list<NodeType*> nodes;
	for (int i = 0; i < count; i++)
	{
		NodeType* node2 = undirected_graph->random_walk(node, 2, *graph_weights);
		nodes.push_back(node2);
	}
	return nodes;
}

template <typename NodeType>
void RWDescent<NodeType>::on_initial_graph_initialized()
{
	reset();
}

template <typename NodeType>
void RWDescent<NodeType>::on_graph_changed()
{
	reset();
}

template <typename NodeType>
void RWDescent<NodeType>::on_iteration_changed()
{
	update_undirected_graph();

	if (rw_pr == RWPr::EdgeMaturity)
	{
		((EdgeMaturityWeights<NodeType>*)graph_weights)->set_current_iteration(this->current_iteration);
	}
}

template <typename NodeType>
void RWDescent<NodeType>::on_edges_added(NodeType* node1, NodeType* node2, int edges_count)
{
	if (rw_pr != RWPr::EdgeMaturity) return;
	((EdgeMaturityWeights<NodeType>*)graph_weights)->add_pair(node1, node2, edges_count);
}

template <typename NodeType>
void RWDescent<NodeType>::on_edges_removed(std::list<std::pair<NodeType*, NodeType*> > &removed_edges)
{
	if (rw_pr != RWPr::EdgeMaturity) return;
	EdgeMaturityWeights<NodeType>* em_graph_weights = (EdgeMaturityWeights<NodeType>*)graph_weights;
	typename std::list<std::pair<NodeType*, NodeType*> >::iterator removed_edges_it = removed_edges.begin();
	while (removed_edges_it != removed_edges.end())
	{
		em_graph_weights->remove_dependency(removed_edges_it->first, removed_edges_it->second);
		removed_edges_it++;
	}
}

template <typename NodeType>
void RWDescent<NodeType>::update_undirected_graph()
{
	if (undirected_graph != nullptr) delete undirected_graph;
	undirected_graph = this->knng.create_undirected_graph();
}

template <typename NodeType>
void RWDescent<NodeType>::reset()
{
	update_undirected_graph();
	
	if (rw_pr == RWPr::EdgeMaturity)
	{	
		((EdgeMaturityWeights<NodeType>*)graph_weights)->clear();
		on_iteration_changed();
		this->knng.reset_iterator();
		std::pair<NodeType*, std::list<NodeType*>*> current;
		while ((current = this->knng.next_node_with_edges()).first != nullptr)
		{
			typename std::list<NodeType*>::iterator it = (*current.second).begin();
			while (it != (*current.second).end())
			{
				on_edges_added(current.first, *it, 1);
				it++;
			}
		}
		this->knng.reset_iterator();
	}
}
} // namespace knng
#endif