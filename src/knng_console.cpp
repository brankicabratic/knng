#include <iostream>
#include <fstream>
#include <cctype>
#include <vector>
#include <map>
#include <math.h>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <boost/filesystem.hpp>
#include "common/utils.h"
#include "common/random.h"
#include "data_objects/dist.h"
#include "data_objects/point.h"
#include "datasets/dataset.h"
#include "simulation/simulator.h"
#include "algorithms/naive_knng.h"
#include "algorithms/nn_descent.h"
#include "algorithms/rw_descent.h"
#include "algorithms/nw_descent.h"
#include "algorithms/ha_nn_descent.h"
#include "algorithms/oversized_nn_descent.h"
#include "algorithms/randomized_nn_descent.h"

using namespace knng;

class Action
{
  public:
	virtual bool init(std::vector<std::string> pars) = 0;
	virtual bool execute() = 0;
	virtual std::string get_help(std::string command);
	virtual std::string get_help_synt(std::string command) = 0;
	virtual std::string get_help_pars() = 0;
	virtual std::string get_help_exec() = 0;
};

class HelpAction : public Action
{
	std::string command;

  public:
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class TestAction : public Action
{
  public:
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class CompoundAction : public Action
{
  protected:
	std::vector<Action*> actions;

  public:
	~CompoundAction();
	void add_action(Action* action);

	// Action overrides
	bool init(std::vector<std::string> pars) override;
	virtual bool execute() = 0;

	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
};

class UnconditionalCompoundAction : public CompoundAction
{
  public:
	bool execute() override;
	std::string get_help_exec() override;
};

class SuccessCompoundAction : public CompoundAction
{
  public:
	bool execute() override;
	std::string get_help_exec() override;
};

class FailureCompoundAction : public CompoundAction
{
  public:
	bool execute() override;
	std::string get_help_exec() override;
};

class ScriptAction : public Action
{
	std::string script_path;

  public:
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class DatasetAction : public Action
{
  protected:
	std::string ds_path;
	std::string ds_type;
	std::vector<std::string> ds_type_pars;
	std::string ds_inst_type;

  public:
	virtual bool init(std::vector<std::string> pars) override;
	std::string get_help_pars() override;
};

class CreateDatasetAction : public DatasetAction
{
	int instances;
	int dimensionality;
	double min_val;
	double max_val;

  public:
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class LoadDatasetAction : public DatasetAction
{
  	template <typename InstanceType>
	static Dataset<InstanceType>* last_loaded_dataset;
	template <typename InstanceType>
	static std::string last_loaded_dataset_path;

  public:
	virtual bool init(std::vector<std::string> pars) override;
	virtual bool execute() override;
	std::string get_ds_inst_type();
	template <typename InstanceType>
	Dataset<InstanceType>* get_dataset();
	virtual std::string get_help_synt(std::string command) override;
	virtual std::string get_help_pars() override;
	virtual std::string get_help_exec() override;
};

class LoadBufferedDatasetAction : public LoadDatasetAction
{
	static TimeSeriesBuffer* last_loaded_dataset;
	static std::string last_loaded_dataset_path;
	int sliding_window;

  public:
  	bool init(std::vector<std::string> pars) override;
	TimeSeriesBuffer* get_dataset();
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class RecallAction : public Action
{
	std::string real_knng_path;
	std::string approx_knng_path;
	bool out = false;
	
  public:
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class KNNGAction : public Action
{
  protected:
	std::string out_path;
	int k;
	Dist<Point>* dist;
	LoadDatasetAction load_dataset_action;
	
  public:
	virtual bool init(std::vector<std::string> pars) override;
	virtual bool execute() override;
	virtual std::string get_help_synt(std::string command) override;
	virtual std::string get_help_pars() override;
	virtual std::string get_help_exec() override;
};

class PartialKNNGAction : public KNNGAction
{
  protected:
	int id_from, id_to;

  public:
	virtual bool init(std::vector<std::string> pars) override;
	virtual bool execute() override;
	virtual std::string get_help_synt(std::string command) override;
	virtual std::string get_help_pars() override;
	virtual std::string get_help_exec() override;
};

class NNDescentAction : public KNNGAction
{
	NNDescent<Point>* nndescent = nullptr;
	Dataset<Point>* dataset = nullptr;
	
  public:
	virtual ~NNDescentAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class RWDescentAction : public KNNGAction
{
	RWDescent<Point>* rwdescent = nullptr;
	Dataset<Point>* dataset = nullptr;
	
  public:
	virtual ~RWDescentAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class NWDescentAction : public KNNGAction
{
	NWDescent<Point>* nwdescent = nullptr;
	Dataset<Point>* dataset = nullptr;
	
  public:
	virtual ~NWDescentAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class HANNDescentAction : public KNNGAction
{
	HANNDescent<Point>* ha_nndescent = nullptr;
	Dataset<Point>* dataset = nullptr;
	
  public:
	virtual ~HANNDescentAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class OversizedNNDescentAction : public KNNGAction
{
	OversizedNNDescent<Point>* oversized_nndescent = nullptr;
	Dataset<Point>* dataset = nullptr;
	
  public:
	virtual ~OversizedNNDescentAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class RandomizedNNDescentAction : public KNNGAction
{
	RandomizedNNDescent<Point>* r_nndescent = nullptr;
	Dataset<Point>* dataset = nullptr;
	
  public:
	virtual ~RandomizedNNDescentAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class ReduceKNNGAction : public Action
{
  protected:
	std::string knng_path;
	std::string out_path;
	int k;
	LoadDatasetAction load_dataset_action;
	
  public:
	virtual bool init(std::vector<std::string> pars) override;
	virtual bool execute() override;
	virtual std::string get_help_synt(std::string command) override;
	virtual std::string get_help_pars() override;
	virtual std::string get_help_exec() override;
};

class PrepareSimulationAction : public Action
{
	Simulator* simulator;
	NaiveKNNGraph<TimeSeries>* naive_knng;
	int runs;
	LoadBufferedDatasetAction load_dataset_action;

  public:
	~PrepareSimulationAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class SimulationAction : public Action
{
	Simulator* simulator;
	std::string out;
	int runs;
	LoadBufferedDatasetAction load_dataset_action;
	std::vector<KNNGApproximation<TimeSeries>*> algorithms;
	KNNGApproximation<TimeSeries>* reference_algorithm;

  public:
	~SimulationAction();
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

class ClearSimulationCacheAction : public Action
{
	std::string dataset;

  public:
	bool init(std::vector<std::string> pars) override;
	bool execute() override;
	std::string get_help_synt(std::string command) override;
	std::string get_help_pars() override;
	std::string get_help_exec() override;
};

typedef Action* (*ActionFactory)();

template <typename T>
Action* make() { return new T{}; }

const std::map<std::string, ActionFactory> all_actions = {
	{"help", make<HelpAction>},
	{"test", make<TestAction>},
	{";", make<UnconditionalCompoundAction>},
	{"&&", make<SuccessCompoundAction>},
	{"||", make<FailureCompoundAction>},
	{"script", make<ScriptAction>},
	{"create_ds", make<CreateDatasetAction>},
	{"knng", make<KNNGAction>},
	{"partial_knng", make<PartialKNNGAction>},
	{"nndescent", make<NNDescentAction>},
	{"rwdescent", make<RWDescentAction>},
	{"nwdescent", make<NWDescentAction>},
	{"hanndescent", make<HANNDescentAction>},
	{"osnndescent", make<OversizedNNDescentAction>},
	{"rnndescent", make<RandomizedNNDescentAction>},
	{"reduce_knng", make<ReduceKNNGAction>},
	{"prepare_simulation", make<PrepareSimulationAction>},
	{"simulation", make<SimulationAction>},
	{"clear_sim_cache", make<ClearSimulationCacheAction>},
	{"recall", make<RecallAction>}
};

Action* create_simple_action(std::string command, std::vector<std::string> pars)
{
	std::map<std::string, ActionFactory>::const_iterator all_actions_it = all_actions.find(command);
	if (all_actions_it == all_actions.end())
		throw std::invalid_argument("Command \"" + command + "\" does not exist. See \"help\" for more details.");
	Action* action = all_actions_it->second();
	bool initialized = false;
	try
	{
		initialized = action->init(pars);
	}
	catch (const std::exception& e) { }

	if (!initialized)
	{
		delete action;
		throw std::invalid_argument("Invalid arguments for command \"" + command + "\". See \"help\" for more details.");
	}
	return action;
}

Action* create_action(std::vector<std::string> args)
{
	bool init_current_command = true;
	std::string current_command;
	std::vector<std::string> current_pars;
	Action* current_action = nullptr;
	std::string current_compound_action_command;
	CompoundAction* current_compound_action = nullptr;
	std::vector<std::string>::iterator args_it = args.begin();
	while (args_it != args.end())
	{
		if ((current_command != "help" || current_pars.size() > 0) && (*args_it == ";" || *args_it == "&&" || *args_it == "||"))
		{
			if (current_compound_action == nullptr || current_compound_action != nullptr && current_compound_action_command != *args_it)
			{
				CompoundAction* new_compound_action;
				if (*args_it == ";")
					new_compound_action = new UnconditionalCompoundAction;
				else if (*args_it == "&&")
					new_compound_action = new SuccessCompoundAction;
				else
					new_compound_action = new FailureCompoundAction;
				
				if (current_compound_action == nullptr)
				{
					new_compound_action->add_action(create_simple_action(current_command, current_pars));
				}
				else
				{
					current_compound_action->add_action(create_simple_action(current_command, current_pars));
					new_compound_action->add_action(current_compound_action);
				}
				current_compound_action = new_compound_action;
				current_compound_action_command = *args_it;
			}
			else
			{
				current_compound_action->add_action(create_simple_action(current_command, current_pars));
			}
						
			init_current_command = true;
		}
		else if (init_current_command)
		{
			init_current_command = false;
			current_command = *args_it;
			current_pars.clear();
		}
		else
		{
			current_pars.push_back(*args_it);
		}
		args_it++;
	}

	Action* action = create_simple_action(current_command, current_pars);
	if (current_compound_action != nullptr)
	{
		current_compound_action->add_action(action);
		return current_compound_action;
	}
	return action;
}

// Action class implementation.

std::string Action::get_help(std::string command)
{
	return
		"Syntax:    " + replace_str(get_help_synt(command), "\n", "\n           ") + "\n"
		"Pars:      " + replace_str(get_help_pars(), "\n", "\n           ") + "\n"
		"Execution: " + replace_str(get_help_exec(), "\n", "\n           ") + "\n";
}

// HelpAction class implementation.

bool HelpAction::init(std::vector<std::string> pars)
{
	if (pars.size() > 1) return false;
	if (pars.size() > 0)
		command = pars[0];
	return true;
}

bool HelpAction::execute()
{
	std::vector<std::string> commands;
	if (command != "")
	{
		if (all_actions.find(command) == all_actions.end())
			throw std::invalid_argument("Command \"" + command + "\" is not recognized.");
		commands.push_back(command);
	}
	else
	{
		std::map<std::string, ActionFactory>::const_iterator all_actions_it = all_actions.begin();
		while (all_actions_it != all_actions.end())
		{
			commands.push_back(all_actions_it->first);
			all_actions_it++;
		}
	}
	std::vector<std::string>::iterator commands_it = commands.begin();
	std::cout << "IMPORTANT: If any parameter contains space character, it should be enclosed in double quotes (\"\")." << std::endl;
	std::cout << "---------------------" << std::endl;
	while (commands_it != commands.end())
	{
		Action* action = all_actions.find(*commands_it)->second();
		std::cout << "Command:   " << *commands_it << std::endl;
		std::cout << action->get_help(*commands_it);
		std::cout << "---------------------" << std::endl;
		delete action;
		commands_it++;
	}
	return true;
}

std::string HelpAction::get_help_synt(std::string command)
{
	return command + " [command]";
}

std::string HelpAction::get_help_pars()
{
	return "command - Arbitrary command.";
}

std::string HelpAction::get_help_exec()
{
	return "If command parameter is not present, all available commands will be listed together with their descriptions. Otherwise, if command parameter is present, only description of that command will be shown.";
}

// TestAction class implementation.

bool TestAction::init(std::vector<std::string> pars)
{
	return true;
}

bool TestAction::execute()
{
	srand(5);
	std::cout << rand() << std::endl;
	std::cout << rand() << std::endl;
	srand(5);
	std::cout << rand() << std::endl;
	std::cout << rand() << std::endl;

	return true;
}

std::string TestAction::get_help_synt(std::string command)
{
	return "";
}

std::string TestAction::get_help_pars()
{
	return "-";
}

std::string TestAction::get_help_exec()
{
	return "Executes arbitrary test code that is currently assigned to the command. Used for testing during the development.";
}

// CompoundAction class implementation.

CompoundAction::~CompoundAction()
{
	std::vector<Action*>::iterator actions_it = actions.begin();
	while (actions_it != actions.end())
	{
		Action* action = *actions_it;
		actions_it = actions.erase(actions_it);
		delete action;
	}
}

void CompoundAction::add_action(Action* action)
{
	actions.push_back(action);
}

bool CompoundAction::init(std::vector<std::string> pars)
{
	return true;
}

std::string CompoundAction::get_help_synt(std::string command)
{
	return "command {" + command + " command }";
}

std::string CompoundAction::get_help_pars()
{
	return "command - Arbitrary command with all its belonging parameters.";
}

// UnconditionalCompoundAction class implementation.

bool UnconditionalCompoundAction::execute()
{
	std::vector<Action*>::iterator actions_it = actions.begin();
	while (actions_it != actions.end())
	{
		try
		{
			(*actions_it)->execute();
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
		}
		actions_it++;		
	}
	return true;
}

std::string UnconditionalCompoundAction::get_help_exec()
{
	return "All given commands will be executed one after the other.";
}

// SuccessCompoundAction class implementation.

bool SuccessCompoundAction::execute()
{
	std::vector<Action*>::iterator actions_it = actions.begin();
	while (actions_it != actions.end())
	{
		try
		{
			if (!(*actions_it)->execute())
				break;
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			break;
		}
		actions_it++;		
	}
	return true;
}

std::string SuccessCompoundAction::get_help_exec()
{
	return "Executes commands one by one. The current command is executed only if the preceding command has executed successfully.";
}

// FailureCompoundAction class implementation.

bool FailureCompoundAction::execute()
{
	std::vector<Action*>::iterator actions_it = actions.begin();
	while (actions_it != actions.end())
	{
		try
		{
			if ((*actions_it)->execute())
				break;
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
		}
		actions_it++;		
	}
	return true;
}

std::string FailureCompoundAction::get_help_exec()
{
	return "Executes commands one by one. The current command is executed only if the preceding command has executed unsuccessfully.";
}

// ScriptAction class implementation.

bool ScriptAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 1) return false;
	if (!file_exists(pars[0])) return false;
	script_path = pars[0];
	return true;
}

bool ScriptAction::execute()
{
	std::ifstream fin;
	fin.open(script_path);
	std::vector<std::string> args;
	std::string current_arg = "";
	char sym;
	bool inside_quotes = false;
	bool prev_bs = false;
	while (fin.get(sym))
	{
		if (isspace(sym) && !inside_quotes)
		{
			if (current_arg != "")
			{
				args.push_back(current_arg);
				current_arg = "";
			}
		}
		else if (sym == '"')
		{
			if (inside_quotes && !prev_bs)
			{
				inside_quotes = false;
			}
			else if (!inside_quotes)
			{
				inside_quotes = true;
			}
		}
		else
		{
			current_arg += sym;
		}
		
		prev_bs = sym == '\\';
	}

	if (inside_quotes)
		throw std::invalid_argument("Quotes are not closed");

	if (current_arg != "")
		args.push_back(current_arg);
	
	Action* action = create_action(args);
	bool success;
	try
	{
		success = action->execute();
	}
	catch(const std::exception& e)
	{
		delete action;
		fin.close();
		throw e;
	}
	
	delete action;
	fin.close();

	return success;
}

std::string ScriptAction::get_help_synt(std::string command)
{
	return command + " script_path";
}

std::string ScriptAction::get_help_pars()
{
	return "script_path - Path of the script file.";
}

std::string ScriptAction::get_help_exec()
{
	return "Executes commands from the external script file.";
}

// DatasetAction class implementation.

bool DatasetAction::init(std::vector<std::string> pars)
{
	ds_path = pars[0];
	std::string ds_type_raw = pars[1];
	ds_inst_type = pars[2];
	
	int lbrack_pos = ds_type_raw.find("[");
	int rbrack_pos = ds_type_raw.rfind("]");
	if (lbrack_pos == std::string::npos)
	{
		ds_type = ds_type_raw;
		ds_type_pars.push_back(",");
	}
	else
	{
		ds_type = ds_type_raw.substr(0, lbrack_pos);
		if (rbrack_pos == std::string::npos)
			throw std::invalid_argument("Missing ending square bracket in the dataset type parameter.");
		ds_type_pars.push_back(ds_type_raw.substr(lbrack_pos+1, rbrack_pos-lbrack_pos-1));
	}

	int lbrace_pos = ds_type_raw.find("{", rbrack_pos+1);
	int rbrace_pos = ds_type_raw.rfind("}");
	if (lbrace_pos != std::string::npos)
	{
		if (ds_type == ds_type_raw)
			ds_type = ds_type_raw.substr(0, lbrace_pos);
		if (rbrace_pos == std::string::npos)
			throw std::invalid_argument("Missing ending curly bracket in the dataset type parameter.");
		ds_type_pars.push_back(ds_type_raw.substr(lbrace_pos+1, rbrace_pos-lbrace_pos-1));
	}

	if (ds_type != "csv")
		throw std::invalid_argument("Dataset type value is not valid.");

	if (ds_inst_type != "point" && ds_inst_type != "timeseries")
		throw std::invalid_argument("Dataset instance type value is not valid.");
	
	return true;
}
  
std::string DatasetAction::get_help_pars()
{
	return 	"ds_path - Path of the dataset.\n"
		   	"ds_type - Dataset type. Possible values: csv[delimiter]{label_index} (Represents CSV dataset. Delimiter should be given in square brackets. Square brackets can be ommited, in which case delimiter becomes comma. Label index is given i curly brackets. Curly bracekets can also be ommited in wich case it is assumed that dataset is not labeled.)\n"
			"ds_inst_type - Type of dataset instances. Possible values: point, timeseries.\n";
}

// CreateDatasetAction class implementation.

bool CreateDatasetAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 5 && pars.size() != 7) return false;
	
	DatasetAction::init(pars);
	
	instances = std::stoi(pars[3]);
	dimensionality = std::stoi(pars[4]);
	if (pars.size() == 7)
	{
		min_val = std::stod(pars[5]);
		max_val = std::stod(pars[6]);
		if (min_val > max_val)
			throw std::invalid_argument("Minimum value must be lower than maximum value.");
	}
	else
	{
		min_val = -1;
		max_val = 1;
	}
	return true;
}

bool CreateDatasetAction::execute()
{
	RandomPointGenerator rnd_point(dimensionality, min_val, max_val);
	Dataset<Point>* dataset = Dataset<Point>::create_dataset(instances, rnd_point);
	
	FileParser* parser;
	if (ds_type == "csv")
		parser = new CsvFileParser(ds_path, ds_type_pars[0]);
	
	if (parser != nullptr)
	{
		std::vector<Point*> instances = dataset->get_instances();
		std::vector<RawDataInstance*> raw_instances;
		for (std::vector<Point*>::iterator it = instances.begin(); it != instances.end(); it++)
			raw_instances.push_back((*it)->to_raw_data_instance(typeid(CsvDataInstance<double>)));
		bool success = parser->output(raw_instances);
		std::vector<RawDataInstance*>::iterator it = raw_instances.begin();
		while (it != raw_instances.end())
		{
			RawDataInstance* instance = *it;
			it = raw_instances.erase(it);
			delete instance;
		}
		delete parser;
		delete dataset;
		return success;
	}
	else
	{
		delete dataset;
		return false;
	}
}

std::string CreateDatasetAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type inst_cnt dim [min_val max_val]";
}

std::string CreateDatasetAction::get_help_pars()
{
	return 	DatasetAction::get_help_pars() +
			"inst_cnt - Number of instances.\n"
			"dim - Dataset dimensionality.\n"
			"min_val (default -1) - Instances' values are generated randomly, minimum value being min_val.\n"
			"max_val (default 1) - Instances' values are generated randomly, maximum value being max_val.\n";
}

std::string CreateDatasetAction::get_help_exec()
{
	return "Creates new dataset by generating values randomly (using uniform distribution).";
}

// LoadDatasetAction class implementation.

template <typename InstanceType>
Dataset<InstanceType>* LoadDatasetAction::last_loaded_dataset;

template <typename InstanceType>
std::string LoadDatasetAction::last_loaded_dataset_path;

bool LoadDatasetAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 3) return false;
	
	DatasetAction::init(pars);

	if (ds_type == "csv" && ds_type_pars.size() == 2)
		std::stoi(ds_type_pars[1]);

	return true;
}

bool LoadDatasetAction::execute() { return true; }

std::string LoadDatasetAction::get_ds_inst_type()
{
	return ds_inst_type;
}

template <typename InstanceType>
Dataset<InstanceType>* LoadDatasetAction::get_dataset()
{
	if (ds_inst_type == "point" && !std::is_same<InstanceType, Point>::value || ds_inst_type == "timeseries" && !std::is_same<InstanceType, TimeSeries>::value)
		throw std::invalid_argument("Dataset InstanceType is not compatible with user supplied arguments.");
	
	if (ds_path == last_loaded_dataset_path<InstanceType>)
		return last_loaded_dataset<InstanceType>;

	FileParser* parser = nullptr;
	if (ds_type == "csv")
	{
		int label_index = -1;
		if (ds_type_pars.size() == 2)
			label_index = std::stoi(ds_type_pars[1]);

		if (ds_inst_type == "point" || ds_inst_type == "timeseries")
		{
			parser = new NumericCsvFileParser(ds_path, ds_type_pars[0], label_index);
		}
	}

	Dataset<InstanceType>* dataset = new Dataset<InstanceType>(*parser);

	if (last_loaded_dataset<InstanceType> != nullptr) delete last_loaded_dataset<InstanceType>;
	last_loaded_dataset<InstanceType> = dataset;
	last_loaded_dataset_path<InstanceType> = ds_path;
	delete parser;

	return dataset;
}

std::string LoadDatasetAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type";
}

std::string LoadDatasetAction::get_help_pars()
{
	return DatasetAction::get_help_pars();
}

std::string LoadDatasetAction::get_help_exec()
{
	return "Loads the dataset.";
}

// LoadBufferedDatasetAction class implementation.

TimeSeriesBuffer* LoadBufferedDatasetAction::last_loaded_dataset;
std::string LoadBufferedDatasetAction::last_loaded_dataset_path;

bool LoadBufferedDatasetAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 4) return false;
	
	LoadDatasetAction::init(pars);
	sliding_window = std::stoi(pars[3]);

	return true;
}

TimeSeriesBuffer* LoadBufferedDatasetAction::get_dataset()
{
	if (ds_inst_type != "timeseries")
		throw std::invalid_argument("Dataset InstanceType should be \"timeseries\".");
	
	if (ds_path == last_loaded_dataset_path)
	{
		last_loaded_dataset->reset();
		return last_loaded_dataset;
	}

	FileParser* parser = nullptr;
	if (ds_type == "csv")
	{
		int label_index = -1;
		if (ds_type_pars.size() == 2)
			label_index = std::stoi(ds_type_pars[1]);

		if (ds_inst_type == "timeseries")
		{
			parser = new NumericCsvFileParser(ds_path, ds_type_pars[0], label_index);
		}
	}

	TimeSeriesBuffer* dataset = new TimeSeriesBuffer(*parser, sliding_window);

	if (last_loaded_dataset != nullptr) delete last_loaded_dataset;
	last_loaded_dataset = dataset;
	last_loaded_dataset_path = ds_path;
	delete parser;

	return dataset;
}

std::string LoadBufferedDatasetAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type sliding_window";
}

std::string LoadBufferedDatasetAction::get_help_pars()
{
	return DatasetAction::get_help_pars() +
			"sliding_window - Time series sliding window size\n";
}

std::string LoadBufferedDatasetAction::get_help_exec()
{
	return "Loads the buffered timeseries dataset.";
}

// RecallAction class implementation.

bool RecallAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 2 && pars.size() != 3) return false;
	real_knng_path = pars[0];
	approx_knng_path = pars[1];
	if (pars.size() == 3)
		out = pars[2] == "out";
	return true;
}

bool RecallAction::execute()
{
	std::ifstream file;

	file.open(real_knng_path);
	std::string real_knng_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	file.close();

	file.open(approx_knng_path);
	std::string approx_knng_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	file.close();

	std::vector<int> ids;
	std::stringstream sin(approx_knng_str);
	std::string line;
	while (std::getline(sin, line))
	{
		int id;
		sin >> id;
		ids.push_back(id);
	}

	Dataset<Point> dataset(ids);

	KNNGraph<Point> real_knng(1, ZeroDist<Point>::get_inst(), dataset.get_instances());
	KNNGraph<Point> approx_knng(1, ZeroDist<Point>::get_inst(), dataset.get_instances());

	real_knng.from_string(real_knng_str);
	approx_knng.from_string(approx_knng_str);

	double recall = approx_knng.get_recall(real_knng);

	std::cout << recall << std::endl;
	if (out) output_to_file(approx_knng_path + "_recall", std::to_string(recall));

	return true;
}

std::string RecallAction::get_help_synt(std::string command)
{
	return command + " real_knng_path approx_knng_path [out]";
}

std::string RecallAction::get_help_pars()
{
	return  "real_knng_path - Path to the file where real KNN graph is stored.\n"
			"approx_knng_path - Path to the file where KNN graph approximation is stored.\n"
			"out - If present, the recall will be outputed to the file called the same as file with KNNG approximation, but with suffix \"_recall\". Possible values: \"out\".\n";
}

std::string RecallAction::get_help_exec()
{
	return "Outputs the recall of the KNN graph approximation.";
}

// KNNGAction class implementation.

bool KNNGAction::init(std::vector<std::string> pars)
{
	load_dataset_action.init(pars);

	out_path = pars[3];
	dist = &get_dist<Point>(pars[4]);
	k = std::stoi(pars[5]);

	return true;
}

bool KNNGAction::execute()
{
	Dataset<Point>* dataset = load_dataset_action.get_dataset<Point>();
	auto start = std::chrono::steady_clock::now();
	KNNGraph<Point> knng(k, *dist, dataset->get_instances());
	knng.generate();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	output_to_file(out_path + "_time", std::to_string(duration.count()));
	bool success = output_to_file(out_path, knng.to_string());
	return success;
}

std::string KNNGAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k";
}

std::string KNNGAction::get_help_pars()
{
	return  load_dataset_action.get_help_pars() +
			"out_path - Path to the file where KNN graph approximation will be stored.\n"
			"dist - Distance measure. Possible values: " + l2_dist + ", " + dtw_dist +".\n"
			"k - Number of neighbors in KNN graph.\n";
}

std::string KNNGAction::get_help_exec()
{
	return "Creates KNN graph.";
}

// PartialKNNGAction class implementation.

bool PartialKNNGAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 8)
		return false;

	KNNGAction::init(pars);

	id_from = std::stoi(pars[6]);
	id_to = std::stoi(pars[7]);

	return true;
}

bool PartialKNNGAction::execute()
{
	Dataset<Point>* dataset = load_dataset_action.get_dataset<Point>();
	std::vector<Point*>& instances = dataset->get_instances();
	std::unordered_set<Point*> subset;
	for (auto instance: instances)
		if (instance->get_id() >= id_from && instance->get_id() <= id_to)
			subset.insert(instance);
	auto start = std::chrono::steady_clock::now();
	KNNGraph<Point> knng(k, *dist, instances);
	
	boost::filesystem::path p(out_path);
	if (p.parent_path() != "")
		boost::filesystem::create_directories(p.parent_path());
	std::fstream stream(out_path, std::fstream::out);
	if (!stream.is_open()) 
		return false;	
	knng.generate_for_subset(subset, stream);
	stream.close();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	return true;
}

std::string PartialKNNGAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k min_id max_id";
}

std::string PartialKNNGAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"min_id - \n"
			"max_id - \n";
}

std::string PartialKNNGAction::get_help_exec()
{
	return "Creates NN lists for all nodes whose id is a value in range [min_id,max_id].";
}

// NNDescentAction class implementation.

NNDescentAction::~NNDescentAction()
{
	if (nndescent != nullptr)
		delete nndescent;

	if (dataset != nullptr)
		delete dataset;
}

bool NNDescentAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 8 && pars.size() != 9) return false;

	KNNGAction::init(pars);
	dataset = load_dataset_action.get_dataset<Point>();

	double sampling = 1;
	if (pars.size() == 9)
		sampling = std::stod(pars[8]);

	if (pars[6] == "it")
	{
		nndescent = new NNDescent<Point>(k, *dist, dataset->get_instances(), std::stoi(pars[7]), sampling);
	}
	else if (pars[6] == "conv")
	{
		nndescent = new NNDescent<Point>(k, *dist, dataset->get_instances(), std::stod(pars[7]), sampling);
	}
	else
	{
		throw std::invalid_argument("nndes_type value is not valid. Read help for more details.");
	}
	return true;
}

bool NNDescentAction::execute()
{
	if (nndescent == nullptr) return false;

	auto start = std::chrono::steady_clock::now();
	
	nndescent->generate_knng();
	KNNGraph<Point>& knng = nndescent->get_knng();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    output_to_file(out_path + "_time", std::to_string(duration.count()));

	return output_to_file(out_path, knng.to_string(false));
}

std::string NNDescentAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k nndes_type (it_count | conv_ratio) [sampling]";
}

std::string NNDescentAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"nndes_type - NNDescent type. Possible values: \"it\" (NNDescent terminates when given number of iterations is reached), \"conv\" (NNDescent terminates when number of updates is less than threshold).\n"
			"it_count - This value is present when nndes_type is \"it\". It represents number of algorithm iterations.\n"
			"conv_ratio - This value is present when nndes_type is \"conv\". If there are less than conv_ratio*N*K updates of NN lists in current iteration, the algorithm terminates.\n"
			"sampling - The portion of neighbors to be used in local joins.";
}

std::string NNDescentAction::get_help_exec()
{
	return "Creates KNN graph approximation by using NNDescent algorithm.";
}

// RWDescentAction class implementation.

RWDescentAction::~RWDescentAction()
{
	if (rwdescent != nullptr)
		delete rwdescent;

	if (dataset != nullptr)
		delete dataset;
}

bool RWDescentAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 10) return false;

	bool conv;
	if (pars[8] == "it")
		conv = false;
	else if (pars[8] == "conv")
		conv = true;
	else
		throw std::invalid_argument("rwdes_type value is not valid. Read help for more details.");

	if ((!conv && pars.size() != 10 && pars.size() != 11) ||
		 (conv && pars.size() != 11 && pars.size() != 12))
		return false;

	KNNGAction::init(pars);
	dataset = load_dataset_action.get_dataset<Point>();

	int rws_count = std::stoi(pars[6]);
	int rand_count = std::stoi(pars[7]);
	int its_count = std::stoi(pars[9]);
	RWDescent<Point>::RWPr rw_pr = RWDescent<Point>::RWPr::Uniform;
	if (!conv && pars.size() == 11 || conv && pars.size() == 12)
	{
		std::string rw_pr_str = conv ? pars[11] : pars[10];
		if (rw_pr_str == "uniform")
			rw_pr = RWDescent<Point>::RWPr::Uniform;
		else if (rw_pr_str == "edge_maturity")
			rw_pr = RWDescent<Point>::RWPr::EdgeMaturity;
		else
			throw std::invalid_argument("rw_pr value is not valid. Read help for more details.");
	}

	std::vector<Point*>& points = dataset->get_instances();

	if (conv)
	{
		double conv_ratio = std::stod(pars[10]);
		rwdescent = new RWDescent<Point>(k, *dist, points, rws_count, rand_count, its_count, conv_ratio, rw_pr);
	}
	else
	{
		rwdescent = new RWDescent<Point>(k, *dist, points, rws_count, rand_count, its_count, rw_pr);
	}

	return true;
}

bool RWDescentAction::execute()
{
	if (rwdescent == nullptr) return false;

	auto start = std::chrono::steady_clock::now();
	
	rwdescent->generate_knng();
	KNNGraph<Point>& knng = rwdescent->get_knng();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    output_to_file(out_path + "_time", std::to_string(duration.count()));

	return output_to_file(out_path, knng.to_string(false));
}

std::string RWDescentAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k rws_count rand_count rwdes_type (it_count | max_it_count) [conv_ratio] [rw_pr]";
}

std::string RWDescentAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"rws_count - Random walks count. It is a number that represents count of random walks that starts from each node (it is usually some factor of k value).\n"
			"rand_count - Number of random comparissons per point in randomize phase.\n"
			"rwdes_type - RWDescent type. Possible values: \"it\" (RWDescent terminates when given number of iterations is reached), \"conv\" (RWDescent terminates when number of updates of each node's NN list is less than threshold).\n"
			"it_count - This value is present when rwdes_type is \"it\". It represents number of algorithm iterations.\n"
			"max_it_count - This value is present when rwdes_type is \"conv\". It represents maximal number of algorithm iterations. Namely, if alogrithm converges before max_it_count iterations, it will end when it converges, otherwise it will end after max_it_count iterations.\n"
			"conv_ratio - This value is present when rwdes_type is \"conv\". Value obtained by multiplying rws_count and conv_ratio represents the treshold of updates obtained by random walks from a given node. If number of updates is above the treshold, the node did not converge, otherwise it did converge. The whole algorithm is converged when all nodes converged.\n"
			"rw_pr - Algorithm for edges' traversal probabilities assignments during the random walks. Possible values: uniform, edge_maturity.";
}

std::string RWDescentAction::get_help_exec()
{
	return "Creates KNN graph approximation by using RWDescent algorithm.";
}


// NWDescentAction class implementation.

NWDescentAction::~NWDescentAction()
{
	if (nwdescent != nullptr)
		delete nwdescent;

	if (dataset != nullptr)
		delete dataset;
}

bool NWDescentAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 10) return false;

	bool conv;
	if (pars[8] == "it")
		conv = false;
	else if (pars[8] == "conv")
		conv = true;
	else
		throw std::invalid_argument("rwdes_type value is not valid. Read help for more details.");

	if ((!conv && pars.size() != 10 && pars.size() != 11) ||
		 (conv && pars.size() != 11 && pars.size() != 12))
		return false;

	KNNGAction::init(pars);
	dataset = load_dataset_action.get_dataset<Point>();

	int rws_count = std::stoi(pars[6]);
	int rand_count = std::stoi(pars[7]);
	int its_count = std::stoi(pars[9]);

	std::vector<Point*>& points = dataset->get_instances();

	if (conv)
	{
		double conv_ratio = std::stod(pars[10]);
		nwdescent = new NWDescent<Point>(k, *dist, points, rws_count, rand_count, its_count, conv_ratio);
	}
	else
	{
		nwdescent = new NWDescent<Point>(k, *dist, points, rws_count, rand_count, its_count);
	}

	return true;
}

bool NWDescentAction::execute()
{
	if (nwdescent == nullptr) return false;

	auto start = std::chrono::steady_clock::now();
	
	nwdescent->generate_knng();
	KNNGraph<Point>& knng = nwdescent->get_knng();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    output_to_file(out_path + "_time", std::to_string(duration.count()));

	return output_to_file(out_path, knng.to_string(false));
}

std::string NWDescentAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k rws_count rand_count rwdes_type (it_count | max_it_count) [conv_ratio]";
}

std::string NWDescentAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"rws_count - Random walks count. It is a number that represents count of random walks that starts from each node (it is usually some factor of k value).\n"
			"rand_count - Number of random comparissons per point in randomize phase.\n"
			"rwdes_type - RWDescent type. Possible values: \"it\" (RWDescent terminates when given number of iterations is reached), \"conv\" (RWDescent terminates when number of updates of each node's NN list is less than threshold).\n"
			"it_count - This value is present when rwdes_type is \"it\". It represents number of algorithm iterations.\n"
			"max_it_count - This value is present when rwdes_type is \"conv\". It represents maximal number of algorithm iterations. Namely, if alogrithm converges before max_it_count iterations, it will end when it converges, otherwise it will end after max_it_count iterations.\n"
			"conv_ratio - This value is present when rwdes_type is \"conv\". Value obtained by multiplying rws_count and conv_ratio represents the treshold of updates obtained by random walks from a given node. If number of updates is above the treshold, the node did not converge, otherwise it did converge. The whole algorithm is converged when all nodes converged.";
}

std::string NWDescentAction::get_help_exec()
{
	return "Creates KNN graph approximation by using NWDescent algorithm.";
}


// HANNDescentAction class implementation.

HANNDescentAction::~HANNDescentAction()
{
	if (ha_nndescent != nullptr)
		delete ha_nndescent;

	if (dataset != nullptr)
		delete dataset;
}

bool HANNDescentAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 10 && pars.size() != 11) return false;

	KNNGAction::init(pars);
	dataset = load_dataset_action.get_dataset<Point>();

	int h_min = std::stoi(pars[8]);
	int h_max = std::stoi(pars[9]);

	double sampling = 1;
	if (pars.size() == 11)
		sampling = std::stod(pars[10]);

	if (pars[6] == "it")
	{
		ha_nndescent = new HANNDescent<Point>(k, *dist, dataset->get_instances(), std::stoi(pars[7]), sampling, h_min, h_max);
	}
	else if (pars[6] == "conv")
	{
		ha_nndescent = new HANNDescent<Point>(k, *dist, dataset->get_instances(), std::stod(pars[7]), sampling, h_min, h_max);
	}
	else
	{
		throw std::invalid_argument("ha_nndes_type value is not valid. Read help for more details.");
	}
	return true;
}

bool HANNDescentAction::execute()
{
	if (ha_nndescent == nullptr) return false;

	auto start = std::chrono::steady_clock::now();
	
	ha_nndescent->generate_knng();
	KNNGraph<Point>& knng = ha_nndescent->get_knng();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    output_to_file(out_path + "_time", std::to_string(duration.count()));

	return output_to_file(out_path, knng.to_string(false));
}

std::string HANNDescentAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k ha_nndes_type (it_count | conv_ratio) h_min h_max [sampling]";
}

std::string HANNDescentAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"ha_nndes_type - Hubness Aware NNDescent type. Possible values: \"it\" (HANNDescent terminates when given number of iterations is reached), \"conv\" (HANNDescent terminates when number of updates is less than threshold).\n"
			"it_count - This value is present when nndes_type is \"it\". It represents number of algorithm iterations.\n"
			"conv_ratio - This value is present when nndes_type is \"conv\". If there are less than conv_ratio*N*K updates of NN lists in current iteration, the algorithm terminates.\n"
			"h_min - \n"
			"h_max - \n"
			"sampling - The portion of neighbors to be used in local joins.\n";
}

std::string HANNDescentAction::get_help_exec()
{
	return "Creates KNN graph approximation by using Hubness Aware NNDescent algorithm.";
}

// OversizedNNDescentAction class implementation.

OversizedNNDescentAction::~OversizedNNDescentAction()
{
	if (oversized_nndescent != nullptr)
		delete oversized_nndescent;

	if (dataset != nullptr)
		delete dataset;
}

bool OversizedNNDescentAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 9 && pars.size() != 10) return false;

	KNNGAction::init(pars);
	dataset = load_dataset_action.get_dataset<Point>();

	int k2 = std::stoi(pars[8]);

	double sampling = 1;
	if (pars.size() == 10)
		sampling = std::stod(pars[9]);

	if (pars[6] == "it")
	{
		oversized_nndescent = new OversizedNNDescent<Point>(k, *dist, dataset->get_instances(), std::stoi(pars[7]), sampling, k2);
	}
	else if (pars[6] == "conv")
	{
		oversized_nndescent = new OversizedNNDescent<Point>(k, *dist, dataset->get_instances(), std::stod(pars[7]), sampling, k2);
	}
	else
	{
		throw std::invalid_argument("os_nndes_type value is not valid. Read help for more details.");
	}
	return true;
}

bool OversizedNNDescentAction::execute()
{
	if (oversized_nndescent == nullptr) return false;

	auto start = std::chrono::steady_clock::now();
	
	oversized_nndescent->generate_knng();
	KNNGraph<Point>& knng = oversized_nndescent->get_knng();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    output_to_file(out_path + "_time", std::to_string(duration.count()));

	return output_to_file(out_path, knng.to_string(false));
}

std::string OversizedNNDescentAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k os_nndes_type (it_count | conv_ratio) k2 [sampling]";
}

std::string OversizedNNDescentAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"os_nndes_type - Oversized NNDescent type. Possible values: \"it\" (Oversized NNDescent terminates when given number of iterations is reached), \"conv\" (Oversized NNDescent terminates when number of updates is less than threshold).\n"
			"it_count - This value is present when os_nndes_type is \"it\". It represents number of algorithm iterations.\n"
			"conv_ratio - This value is present when os_nndes_type is \"conv\". If there are less than conv_ratio*N*K updates of NN lists in current iteration, the algorithm terminates.\n"
			"k2 - \n"
			"sampling - The portion of neighbors to be used in local joins.\n";
}

std::string OversizedNNDescentAction::get_help_exec()
{
	return "Creates KNN graph approximation by using Oversized NNDescent algorithm.";
}

// RandomizedNNDescentAction class implementation.

RandomizedNNDescentAction::~RandomizedNNDescentAction()
{
	if (r_nndescent != nullptr)
		delete r_nndescent;

	if (dataset != nullptr)
		delete dataset;
}

bool RandomizedNNDescentAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 9 && pars.size() != 10) return false;

	KNNGAction::init(pars);
	dataset = load_dataset_action.get_dataset<Point>();

	int r = std::stoi(pars[8]);

	double sampling = 1;
	if (pars.size() == 10)
		sampling = std::stod(pars[9]);

	if (pars[6] == "it")
	{
		r_nndescent = new RandomizedNNDescent<Point>(k, *dist, dataset->get_instances(), std::stoi(pars[7]), sampling, r);
	}
	else if (pars[6] == "conv")
	{
		r_nndescent = new RandomizedNNDescent<Point>(k, *dist, dataset->get_instances(), std::stod(pars[7]), sampling, r);
	}
	else
	{
		throw std::invalid_argument("r_nndes_type value is not valid. Read help for more details.");
	}
	return true;
}

bool RandomizedNNDescentAction::execute()
{
	if (r_nndescent == nullptr) return false;

	auto start = std::chrono::steady_clock::now();
	
	r_nndescent->generate_knng();
	KNNGraph<Point>& knng = r_nndescent->get_knng();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    output_to_file(out_path + "_time", std::to_string(duration.count()));

	return output_to_file(out_path, knng.to_string(false));
}

std::string RandomizedNNDescentAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type out_path dist k os_nndes_type (it_count | conv_ratio) r [sampling]";
}

std::string RandomizedNNDescentAction::get_help_pars()
{
	return  KNNGAction::get_help_pars() +
			"os_nndes_type - Oversized NNDescent type. Possible values: \"it\" (Oversized NNDescent terminates when given number of iterations is reached), \"conv\" (Oversized NNDescent terminates when number of updates is less than threshold).\n"
			"it_count - This value is present when os_nndes_type is \"it\". It represents number of algorithm iterations.\n"
			"conv_ratio - This value is present when os_nndes_type is \"conv\". If there are less than conv_ratio*N*K updates of NN lists in current iteration, the algorithm terminates.\n"
			"r - \n"
			"sampling - The portion of neighbors to be used in local joins.\n";
}

std::string RandomizedNNDescentAction::get_help_exec()
{
	return "Creates KNN graph approximation by using Randomized NNDescent algorithm.";
}

// ReduceKNNGAction class implementation.

bool ReduceKNNGAction::init(std::vector<std::string> pars)
{
	load_dataset_action.init(pars);

	knng_path = pars[3];
	out_path = pars[4];
	k = std::stoi(pars[5]);

	return true;
}

bool ReduceKNNGAction::execute()
{
	Dataset<Point>* dataset = load_dataset_action.get_dataset<Point>();
	
	KNNGraph<Point> knng(k, ZeroDist<Point>::get_inst(), dataset->get_instances());

	std::ifstream file(knng_path);
	std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	file.close();
	knng.from_string(str);
	file.close();
	
	KNNGraph<Point>* reduced_knng = knng.get_reduced_to_k(k);
	bool success = output_to_file(out_path, reduced_knng->to_string());

	delete reduced_knng;
	return success;
}

std::string ReduceKNNGAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type knng_path out_path k";
}

std::string ReduceKNNGAction::get_help_pars()
{
	return  load_dataset_action.get_help_pars() +
			"knng_path - Path to the file where KNN graph is stored.\n"
			"out_path - Path to the file where reduced KNN graph will be stored.\n"
			"k - Number of neighbors in reduced KNN graph.\n";
}

std::string ReduceKNNGAction::get_help_exec()
{
	return "Reduces KNN graph to given k.";
}

// PrepareSimulationAction class implementation.

PrepareSimulationAction::~PrepareSimulationAction()
{
	if (simulator != nullptr)
		delete simulator;
	if (naive_knng != nullptr)
		delete naive_knng;
}

bool PrepareSimulationAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 8) return false;

	load_dataset_action.init(pars);

	int i = 4;
	runs = std::stoi(pars[i++]);
	int k = std::stoi(pars[i++]);
	std::string dist_str = pars[i++];
	if (dist_str != l2_dist && dist_str != dtw_dist)
		throw std::invalid_argument("Unsupported distance.");
	Dist<TimeSeries>& dist = dist_str == l2_dist ? (Dist<TimeSeries>&)TimeSeriesL2Dist::get_inst() : (Dist<TimeSeries>&)TimeSeriesDTWDist::get_inst();
	int batch_size_min = std::stoi(pars[i++]);
	int batch_size_max = batch_size_min;
	if (pars.size() - i == 3)
		batch_size_max = std::stoi(pars[i++]);
	int points_min = -1;
	int points_max = -1;
	if (pars.size() - i >= 1) 
	{
		points_min = std::stoi(pars[i++]);
		points_max = points_min;
		if (pars.size() - i == 1)
			points_max = std::stoi(pars[i++]);
	}
	TimeSeriesBuffer *dataset = load_dataset_action.get_dataset();
	naive_knng = new NaiveKNNGraph<TimeSeries>(k, dist, dataset->get_instances());
	simulator = new Simulator(*dataset, batch_size_min, batch_size_max, points_min, points_max);
	return true;
}

bool PrepareSimulationAction::execute()
{
	NullOutStream null_out;
	for (int i = 0; i < runs; i++)
		simulator->generate_reference_knngs(naive_knng, null_out, i + 1);
	return true;
}

std::string PrepareSimulationAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type sliding_window runs k dist (batch_size | (batch_size_min batch_size_max [points_min points_max]))";
}

std::string PrepareSimulationAction::get_help_pars()
{
	return  load_dataset_action.get_help_pars() +
			"runs - Number of simulation runs.\n"
			"k - K value of the KNN Graph.\n"
			"dist - Distance function that will be used. Possible values are: \"" + l2_dist + "\" (Euclidean distance), \"" + dtw_dist + "\" (Dynamic Time Warping).\n"
			"batch_size - The size of the batch that will be loaded into time series during the updates.\n"
			"batch_size_min - When this value is given, the size of the batch that will be loaded into time series during the updates is randomly chosen. This value is the lower bound of randomly chosen batch value.\n"
			"batch_size_max - When this value is given, the size of the batch that will be loaded into time series during the updates is randomly chosen. This value is the upper bound of randomly chosen batch value.\n"
			"points_min - When this value is given, the number of points that will be updated during one update is randomly chosen. This value is then lower bound of number of updated points.\n"
			"points_max - When this value is given, the number of points that will be updated during one update is randomly chosen. This value is then upper bound of number of updated points.\n";
}

std::string PrepareSimulationAction::get_help_exec()
{
	return "Does a preparation for simulation by creating all KNN-Graphs later needed for calculation of recall values.";
}

// SimulationAction class implementation.

SimulationAction::~SimulationAction()
{
	if (simulator != nullptr)
		delete simulator;

	for (KNNGApproximation<TimeSeries>* algorithm: algorithms)
		delete algorithm;
	
	if (reference_algorithm != nullptr)
		delete reference_algorithm;
}

bool SimulationAction::init(std::vector<std::string> pars)
{
	if (pars.size() < 9) return false;

	load_dataset_action.init(pars);

	int i = 4;
	runs = std::stoi(pars[i++]);
	out = pars[i++];
	int k = std::stoi(pars[i++]);
	std::string dist_str = pars[i++];
	if (dist_str != l2_dist && dist_str != dtw_dist)
		throw std::invalid_argument("Unsupported distance.");
	Dist<TimeSeries>& dist = dist_str == l2_dist ? (Dist<TimeSeries>&)TimeSeriesL2Dist::get_inst() : (Dist<TimeSeries>&)TimeSeriesDTWDist::get_inst();
	int batch_size_min = std::stoi(pars[i++]);
	int batch_size_max = batch_size_min;
	if (pars.size() - i == 3)
		batch_size_max = std::stoi(pars[i++]);
	int points_min = -1;
	int points_max = -1;
	if (pars.size() - i >= 1) 
	{
		points_min = std::stoi(pars[i++]);
		points_max = points_min;
		if (pars.size() - i == 1)
			points_max = std::stoi(pars[i++]);
	}
	TimeSeriesBuffer *dataset = load_dataset_action.get_dataset();

	algorithms.push_back(new NNDescent<TimeSeries>(k, dist, dataset->get_instances(), 0.01, 1));
	int dist_cnt =  (int)std::round((float)dataset->get_instances().size() / (4*k*k));
	// algorithms.push_back(new RWDescent<TimeSeries>(k, dist, dataset->get_instances(), 5, dist_cnt, 10000000, 0.001));
	// algorithms.push_back(new RWDescent<TimeSeries>(k, dist, dataset->get_instances(), 10, dist_cnt, 10000000, 0.001));
	// algorithms.push_back(new NWDescent<TimeSeries>(k, dist, dataset->get_instances(), 5, dist_cnt, 10000000, 0.001));
	// algorithms.push_back(new NWDescent<TimeSeries>(k, dist, dataset->get_instances(), 10, dist_cnt, 10000000, 0.001));

	reference_algorithm = new NaiveKNNGraph<TimeSeries>(k, dist, dataset->get_instances());
	simulator = new Simulator(*dataset, batch_size_min, batch_size_max, points_min, points_max);
	return true;
}

bool SimulationAction::execute()
{
	if (out == "cout")
	{
		for (int i = 0; i < runs; i++)
			simulator->start_simulation(algorithms, reference_algorithm, std::cout, i);
	}
	else
	{
		for (int i = 0; i < runs; i++)
		{
			if (file_exists(out + "_" + std::to_string(i+1))) continue; 
			std::ofstream fout;
			fout.open(out + "_" + std::to_string(i+1));
			simulator->start_simulation(algorithms, reference_algorithm, fout, i+1);
			fout.close();
		}
	}

	return true;
}

std::string SimulationAction::get_help_synt(std::string command)
{
	return command + " ds_path ds_type ds_inst_type sliding_window runs out k dist (batch_size | (batch_size_min batch_size_max [points_min points_max]))";
}

std::string SimulationAction::get_help_pars()
{
	return  load_dataset_action.get_help_pars() +
			"runs - Number of simulation runs.\n"
			"out - If value \"cout\" is given, the simulation will output results to the standard output. Otherwise, the file path should be given, in which case the results will be written in that file. The results after each update come in following format: \"execution_time,dist_calcs_recall{;execution_time,dist_calcs_recall}\", where curly braces are not part of output, but mean that part inside the braces can repeat zero or more times (depending on the number of examined algorithms). Results of each update are then placed in their own line.\n"
			"k - K value of the KNN Graph.\n"
			"dist - Distance function that will be used. Possible values are: \"" + l2_dist + "\" (Euclidean distance), \"" + dtw_dist + "\" (Dynamic Time Warping).\n"
			"batch_size - The size of the batch that will be loaded into time series during the updates.\n"
			"batch_size_min - When this value is given, the size of the batch that will be loaded into time series during the updates is randomly chosen. This value is the lower bound of randomly chosen batch value.\n"
			"batch_size_max - When this value is given, the size of the batch that will be loaded into time series during the updates is randomly chosen. This value is the upper bound of randomly chosen batch value.\n"
			"points_min - When this value is given, the number of points that will be updated during one update is randomly chosen. This value is then lower bound of number of updated points.\n"
			"points_max - When this value is given, the number of points that will be updated during one update is randomly chosen. This value is then upper bound of number of updated points.\n";
}

std::string SimulationAction::get_help_exec()
{
	return "Executes simulation whose purpose is to evaluate performance of rw-descent algorithm used for updating KNN-Graph whose nodes are time series that change over time. Time series are always of the fixed size (that is equal to sliding window) but during the time they obtain new values, and dispose the old ones. Each time a subset of time series update, rw-descent should update KNN-Graph approximation. Newly updated approximation is then evaluated.";
}

// ClearSimulationCacheAction class implementation.

bool ClearSimulationCacheAction::init(std::vector<std::string> pars)
{
	if (pars.size() != 1) return false;
	dataset = pars[0];
	return true;
}

bool ClearSimulationCacheAction::execute()
{
	Simulator::clear_ds_cache(dataset, true);
	Simulator::clear_ds_cache(dataset, false);

	return true;
}

std::string ClearSimulationCacheAction::get_help_synt(std::string command)
{
	return command + " ds_name";
}

std::string ClearSimulationCacheAction::get_help_pars()
{
	return  "ds_name - Name of the data set. It is actally the file name of the data set without extension.\n";
}

std::string ClearSimulationCacheAction::get_help_exec()
{
	return "Removes all simulation cache for the given data set.";
}


int main(int argc, char **argv)
{
	std::vector<std::string> args;
	for (int i = 1; i < argc; i++)
		args.push_back(argv[i]);
	
	try
	{
		bool success;
		Action* action = create_action(args);
		try
		{
			success = action->execute();
			delete action;
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			delete action;
		}		

		if (!success)
			std::cerr << "Action was not executed successfully." << '\n';
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
	return 0;
}