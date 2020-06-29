#ifndef GRAPH_H
#define GRAPH_H

#include <list>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#include <type_traits>
#include "../common/string_converter.h"
#include "../common/identifiable.h"

namespace knng
{

// Abstract class for weighted graph extension.
template <typename NodeType>
class GraphWeights : public StringConverter
{
  public:
	virtual double get_weight(NodeType* node1, NodeType* node2) = 0;              // Returns the weigth of the edge (node1,node2).
	virtual void from_string(std::vector<NodeType*> &nodes, std::string str) = 0; // Loads graph weights from the given string. Needs also set of nodes to be able to obtain pointers, if needed.
};

// Trivial weighted graph extension where each edge has weight equal to one.
template <typename NodeType>
class UniformWeights : public GraphWeights<NodeType>
{
  public:
	double get_weight(NodeType* node1, NodeType* node2) override;
	void from_string(std::vector<NodeType*> &nodes, std::string str) override;
};

// Model class for graph.
template <typename NodeType>
class Graph : StringConverter
{
	static_assert(std::is_base_of<Identifiable, NodeType>::value, "NodeType must implement Identifiable class.");
	static_assert(std::is_base_of<StringConverter, NodeType>::value, "NodeType must implement StringConverter class.");

	struct NodeTypeComparator
	{
		inline bool operator() (NodeType*& n1, NodeType*& n2)
		{
			return *n1 < *n2;
		}
	};

  private:
  	// Type of hash table that stores graph data.
	// Key:   Node of the graph.
	// Value: List of nodes to which key node is connected to.
	typedef std::unordered_map<NodeType*, std::list<NodeType*> > graph_map;
	graph_map graph;                         // Hash table for storing graph data.
	graph_map r_graph;                       // Hash table for storing reverse graph data.
	typename graph_map::iterator current_it; // Iterator that is used for traversing the graph nodes.
	
  public:
	int nodes_count();                                                  // Returns number of the nodes in graph.
	NodeType* next_node();                                              // Returns the next node.
	std::pair<NodeType*, std::list<NodeType*>*> next_node_with_edges(); // Returns the next node with its edges.
	void reset_iterator();                                              // Resets the iterator.
	virtual void add_node(NodeType *node);                              // Adds the node to the graph.
	std::vector<NodeType*> get_sorted_nodes();                          // Returns the sorted vector of all nodes.
	std::list<NodeType*> get_nodes();                                   // Returns all nodes of the graph.
	std::vector<NodeType*> get_nodes_vector();                          // Returns all nodes of the graph.
	virtual bool add_edge(NodeType* node1, NodeType* node2);            // Adds directed edge (node1, node2) at the beginning of the edges list.
	std::list<NodeType*> get_edges(NodeType* node);                     // Returns the edges of the given node.
	std::list<NodeType*> get_r_edges(NodeType* node);                   // Returns the reverse (in) edges of the given node.
	virtual void remove_edges();                                        // Removes all edges from the graph.
	Graph<NodeType>* create_undirected_graph();                         // Returns the graph with following property: edge (x,y) is in the graph iff edge (y,x) is also in the graph.
	// Returns the node in which random walk ends. Random walk starts from start_node and has length walk_length.
	// GraphWeights object should provide the edge weights, where greater weight means higher
	// probability for traversing the edge. 
	NodeType* random_walk(NodeType* start_node, int walk_length, GraphWeights<NodeType> &pr);

	// StringConverter overrides
	std::string to_string(bool brief = true) override;

  protected:
	void init();                                                                                         // Initializes graph. Used in constructors.
	virtual bool add_edge(NodeType* node1, NodeType* node2, typename std::list<NodeType*>::iterator it); // Adds directed edge (node1, node2) at the position it.
	virtual NodeType* remove_last_edge(NodeType *node);                                                  // Removes last edge from the node and returns it.
	// Returns randomly chosen neighbor of the given node. Returning neighbor should be different from forbidden_node.
	// Anyway, forbidden_node is returned if it is the only node's neighbor.
	// Parameter pr contains edge weights - greater weight implies greater probability for the edge's end-node to be chosen.
	NodeType* rnd_neighbor(NodeType* node, NodeType* forbidden_node, GraphWeights<NodeType> &pr);
	NodeType* find_node_by_id(int id); // Finds node by its id.
	
	// Sorts the edges of given node by using given comparator.
	template <class Compare>
	void sort_edges(NodeType* node, Compare comp);
	
	// Returns an iterator that points to the first edge of given node for which given predicate is true.
	// Search starts from the begining of the list and goes in direction begin->end. Parameter Termination t
	// is used for early termination of the search - if t returns true, search stops even if the edge with
	// given predicate is not yet found.
	// Parameter steps will hold the number that tells in how many steps was the edge found. If edge was
	// not found at all, the value will be 0.
	template <class Predicate, class Termination>
	typename std::list<NodeType*>::iterator find_edge(NodeType* node, Predicate p, Termination t, int& steps);

	// Returns an iterator that points to the first edge of given node for which given predicate is true.
	// Search starts from it_from and goes in direction begin->end. Parameter Termination t
	// is used for early termination of the search - if t returns true, search stops even if the edge with
	// given predicate is not yet found.
	// Parameter steps will hold the number that tells in how many steps was the edge found. If edge was
	// not found at all, the value will be 0.
	template <class Predicate, class Termination>
	typename std::list<NodeType*>::iterator find_edge(NodeType* node, Predicate p, Termination t, typename std::list<NodeType*>::iterator it_from, int& steps);

	// Returns an iterator that points to the first edge of given node for which given predicate is true.
	// Search starts from the end of the list and goes in direction end->begin. Parameter Termination t
	// is used for early termination of the search - if t returns true, search stops even if the edge with
	// given predicate is not yet found.
	// Parameter steps will hold the number that tells in how many steps was the edge found. If edge was
	// not found at all, the value will be 0.
	template <class Predicate, class Termination>
	typename std::list<NodeType*>::iterator r_find_edge(NodeType* node, Predicate p, Termination t, int& steps);

	// Returns an iterator that points to the first edge of given node for which given predicate is true.
	// Search starts from it_from and goes in direction end->begin. Parameter Termination t
	// is used for early termination of the search - if t returns true, search stops even if the edge with
	// given predicate is not yet found.
	// Parameter steps will hold the number that tells in how many steps was the edge found. If edge was
	// not found at all, the value will be 0.
	template <class Predicate, class Termination>
	typename std::list<NodeType*>::iterator r_find_edge(NodeType* node, Predicate p, Termination t, typename std::list<NodeType*>::iterator it_from, int& steps);

  private:
	std::list<NodeType*>& get_edges_ref(NodeType* node);   // Returns the reference to the edges list of the given node.
	std::list<NodeType*>& get_r_edges_ref(NodeType* node); // Returns the reference to the reverse edges list of the given node.
};

// UniformWeights<NodeType> class implementation.

template <typename NodeType>
double UniformWeights<NodeType>::get_weight(NodeType* node1, NodeType* node2) 
{
	return 1;
}

template <typename NodeType>
void UniformWeights<NodeType>::from_string(std::vector<NodeType*> &nodes, std::string str) { }

// Graph<NodeType> class implementation.

template <typename NodeType>
int Graph<NodeType>::nodes_count()
{
	return graph.size();
}

template <typename NodeType>
NodeType* Graph<NodeType>::next_node()
{
	if (current_it == graph.end())
		return nullptr;
	NodeType* node = current_it->first;
	current_it++;
	return node;
}

template <typename NodeType>
std::pair<NodeType*, std::list<NodeType*>*> Graph<NodeType>::next_node_with_edges()
{
	if (current_it == graph.end())
		return std::make_pair(nullptr, nullptr);

	std::pair<NodeType*, std::list<NodeType*>*> return_val(current_it->first, &(current_it->second));
	current_it++;
	return return_val;
}

template <typename NodeType>
void Graph<NodeType>::reset_iterator()
{
	current_it = graph.begin();
}

template <typename NodeType>
void Graph<NodeType>::add_node(NodeType* node)
{	
	graph[node].clear();
	r_graph[node].clear();
}

template <typename NodeType>
std::vector<NodeType*> Graph<NodeType>::get_sorted_nodes()
{	
	std::vector<NodeType*> nodes;
	nodes.reserve(graph.size());
	for(auto kv : graph)
		nodes.push_back(kv.first);  
	std::sort(nodes.begin(), nodes.end(), NodeTypeComparator());
	return nodes;
}

template <typename NodeType>
std::list<NodeType*> Graph<NodeType>::get_nodes()
{
	std::list<NodeType*> nodes;
	for (typename graph_map::iterator it = graph.begin(); it != graph.end(); it++)
		nodes.push_back(it->first);
	return nodes;
}

template <typename NodeType>
std::vector<NodeType*> Graph<NodeType>::get_nodes_vector()
{
	std::vector<NodeType*> nodes;
	nodes.reserve(graph.size());
	for(auto kv : graph)
		nodes.push_back(kv.first);  
	return nodes;
}

template <typename NodeType>
bool Graph<NodeType>::add_edge(NodeType* node1, NodeType* node2)
{
	typename std::list<NodeType*>& edges = get_edges_ref(node1);
	Graph<NodeType>::add_edge(node1, node2, edges.begin());
	return true;
}

template <typename NodeType>
std::list<NodeType*> Graph<NodeType>::get_edges(NodeType* node)
{
	return get_edges_ref(node);
}

template <typename NodeType>
std::list<NodeType*> Graph<NodeType>::get_r_edges(NodeType* node)
{
	typename graph_map::iterator entry = r_graph.find(node);
	if (entry == r_graph.end())
		throw std::invalid_argument("Given node is not in the graph.");
	return entry->second;
}

template <typename NodeType>
void Graph<NodeType>::remove_edges()
{
	typename graph_map::iterator entry = graph.begin();
	typename graph_map::iterator entry_r = r_graph.begin();
	while (entry != graph.end())
	{
		entry->second.clear();
		entry_r->second.clear();
		entry++;
		entry_r++;
	}
}

template <typename NodeType>
Graph<NodeType>* Graph<NodeType>::create_undirected_graph()
{
	Graph<NodeType>* new_graph = new Graph<NodeType>;
	typename graph_map::iterator it = this->graph.begin();
	while (it != this->graph.end())
	{
		new_graph->add_node(it->first);
		it++;
	}

	it = this->graph.begin();
	while (it != this->graph.end())
	{
		typename std::list<NodeType*>::iterator edges_it = it->second.begin();
		while (edges_it != it->second.end())
		{
			std::list<NodeType*>& new_graph_edges = new_graph->graph[it->first];
			if (std::find(new_graph_edges.begin(), new_graph_edges.end(), *edges_it) == new_graph_edges.end())
			{
				new_graph_edges.push_back(*edges_it);
				new_graph->graph[*edges_it].push_back(it->first);
			}
			edges_it++;
		}
		it++;
	}
	new_graph->reset_iterator();
	return new_graph;
}

template <typename NodeType>
NodeType* Graph<NodeType>::random_walk(NodeType* start_node, int walk_length, GraphWeights<NodeType> &pr)
{
	NodeType* current = start_node;
	for (int i = 0; i < walk_length; i++)
		current = rnd_neighbor(current, start_node, pr);
	return current;
}

template <typename NodeType>
std::string Graph<NodeType>::to_string(bool brief)
{
	std::stringstream sout;

	for (auto key : get_sorted_nodes())
	{
		typename graph_map::iterator it = graph.find(key);
		sout << ((StringConverter*)it->first)->to_string() << " -> ";
		typename std::list<NodeType*>::iterator it_edges = it->second.begin();
		while(it_edges != it->second.end())
		{
			NodeType* node = *it_edges;
			sout << ((StringConverter*)node)->to_string() << " ";
			it_edges++;
		}
		sout << std::endl;
		it++;
	}
	return sout.str();
}

template <typename NodeType>
void Graph<NodeType>::init()
{	
	reset_iterator();
}

template <typename NodeType>
bool Graph<NodeType>::add_edge(NodeType* node1, NodeType* node2, typename std::list<NodeType*>::iterator it)
{
	typename std::list<NodeType*>& edges = get_edges_ref(node1);
	typename std::list<NodeType*>& r_edges = get_r_edges_ref(node2);
	edges.insert(it, node2);
	r_edges.push_back(node1);
	return true;
}

template <typename NodeType>
NodeType* Graph<NodeType>::remove_last_edge(NodeType *node)
{
	typename std::list<NodeType*>& edges = get_edges_ref(node);
	typename std::list<NodeType*>::reverse_iterator last_edge = edges.rbegin();
	if (last_edge == edges.rend())
		throw std::invalid_argument("Graph<NodeType>::remove_last_edge - Given node does not have edges.");
	NodeType* last_node = *last_edge;
	typename std::list<NodeType*>& r_edges = get_r_edges_ref(last_node);
	edges.pop_back();
	r_edges.remove(node);
	return last_node;
}

template <typename NodeType>
NodeType* Graph<NodeType>::rnd_neighbor(NodeType* node, NodeType* forbidden_node, GraphWeights<NodeType> &pr)
{
	typename graph_map::iterator entry = graph.find(node);

	if (entry == graph.end())
		throw std::invalid_argument("Graph<NodeType>::rnd_neighbor - Given node is not in the graph.");

	int edges_cnt = entry->second.size();
	
	if (edges_cnt == 0)
		throw std::length_error("Given node does not have neighbors");

	if (edges_cnt == 1)
		return entry->second.front();

	typename std::list<NodeType*>::iterator edges_it = entry->second.begin();
	double probs[edges_cnt];
	double cumulative_prob = 0;
	int i = 0;
	while (edges_it != entry->second.end())
	{
		double prob = 0;
		if (*edges_it != forbidden_node)
			prob = pr.get_weight(node, *edges_it);
		cumulative_prob += prob;
		probs[i] = cumulative_prob;
		i++;
		edges_it++;
	}
	probs[edges_cnt-1]++; // Because of the condition in while loop bellow. Does not influence probability of last element.

	double rnd = ((double)rand() / RAND_MAX) * cumulative_prob;
	edges_it = entry->second.begin();
	i = 0;
	while (probs[i] <= rnd)
	{
		i++;
		edges_it++;
	}

	return *edges_it;
}

template <typename NodeType>
NodeType* Graph<NodeType>::find_node_by_id(int id)
{
	typename graph_map::iterator entry = graph.begin();
	while (entry != graph.end())
	{
		if (((Identifiable*)entry->first)->get_id() == id)
			return entry->first;
		entry++;
	}
	return nullptr;
}

template <typename NodeType>
template <class Compare>
void Graph<NodeType>::sort_edges(NodeType* node, Compare comp)
{
	typename graph_map::iterator entry = graph.find(node);
	if (entry == graph.end())
		throw std::invalid_argument("Given node is not in the graph.");
	entry->second.sort(comp);
}

template <typename NodeType>
template <class Predicate, class Termination>
typename std::list<NodeType*>::iterator Graph<NodeType>::find_edge(NodeType* node, Predicate p, Termination t, int& steps)
{
	std::list<NodeType*>& edges = get_edges_ref(node);
	return find_edge(node, p, t, edges.begin(), steps);
}

template <typename NodeType>
template <class Predicate, class Termination>
typename std::list<NodeType*>::iterator Graph<NodeType>::find_edge(NodeType* node, Predicate p, Termination t, typename std::list<NodeType*>::iterator it_from, int& steps)
{
	std::list<NodeType*>& edges = get_edges_ref(node);
	steps = 1;
	while (it_from != edges.end() && !p(*it_from) && !t(*it_from))
	{
		steps++;
		it_from++;
	}

	if (it_from == edges.end() || p(*it_from)) steps = 0;
	
	return it_from;
}

template <typename NodeType>
template <class Predicate, class Termination>
typename std::list<NodeType*>::iterator Graph<NodeType>::r_find_edge(NodeType* node, Predicate p, Termination t, int& steps)
{
	std::list<NodeType*>& edges = get_edges_ref(node);
	return r_find_edge(node, p, t, edges.rbegin().base(), steps);
}

template <typename NodeType>
template <class Predicate, class Termination>
typename std::list<NodeType*>::iterator Graph<NodeType>::r_find_edge(NodeType* node, Predicate p, Termination t, typename std::list<NodeType*>::iterator it_from, int& steps)
{
	std::list<NodeType*>& edges = get_edges_ref(node);
	typename std::list<NodeType*>::reverse_iterator it(it_from);
	steps = 1;
	while (it != edges.rend() && !p(*it) && !t(*it))
	{
		steps++;
		it++;
	}

	if (it == edges.rend() || !p(*it)) steps = 0;

	return it.base();
}

template <typename NodeType>
std::list<NodeType*>& Graph<NodeType>::get_edges_ref(NodeType* node)
{
	typename graph_map::iterator entry = graph.find(node);
	if (entry == graph.end())
		throw std::invalid_argument("Given node is not in the graph.");
	return entry->second;
}

template <typename NodeType>
std::list<NodeType*>& Graph<NodeType>::get_r_edges_ref(NodeType* node)
{
	typename graph_map::iterator entry = r_graph.find(node);
	if (entry == r_graph.end())
		throw std::invalid_argument("Given node is not in the graph.");
	return entry->second;
}

} // namespace knng
#endif