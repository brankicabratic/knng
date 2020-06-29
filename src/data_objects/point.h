#ifndef POINT_H
#define POINT_H

#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <math.h>
#include <limits>
#include "../common/string_converter.h"
#include "../common/identifiable.h"
#include "../common/random.h"
#include "../exceptions/not_implemented_exception.h"
#include "../datasets/file_parser.h"
#include "../data_objects/dist.h"

namespace knng
{

// Model class for d-dimensional point in space.
class Point : public StringConverter, public RawDataInstanceConverter, public Identifiable
{
  protected:
	std::string label;
	std::vector<double> data;

  public:
	Point();
	Point (const Point &p);                                                 // Copy constructor.
	Point(int id, int d, std::string label = "");                           // Initializes a d-dimensional point by putting zero as a value of each dimension.
	Point(int id, const std::vector<double>& data, std::string label = ""); // Initializes a point with given data.
	int get_dim_cnt() const;                                                // Returns the dimensionality of the point.
	double get_dim_val(int dim) const;                                      // Returns the value of dimension dim.
	void set_dim_val(int dim, double val);                                  // Sets the value of the dimension dim.

	// Operators override
	bool operator<(const Point& p);

	// StringConverter override
	std::string to_string(bool brief = true) override;
	void from_string(std::string str, bool brief = true) override;

	// RawDataInstanceConverter override
	virtual RawDataInstance* to_raw_data_instance(const std::type_info& raw_instance_type) override;
	virtual void from_raw_data_instance(RawDataInstance* instance) override;
};

// L2 (Euclidean) distance defined on class Point.
class PointL2Dist : public Dist<Point>
{
	static PointL2Dist *instance;
  
  protected:
	PointL2Dist();

  public:
	static PointL2Dist& get_inst(); // Returns an instance of PointL2Dist.

	// Dist<Point> override
	double calc(const Point&, const Point&) override;
};

// DTW (Dynamic Time Warping) distance defined on class Point.
class PointDTWDist : public Dist<Point>
{
	static PointDTWDist *instance;
  
  protected:
	PointDTWDist();

  public:
	static PointDTWDist& get_inst(); // Returns an instance of PointDTWDist.

	// Dist<Point> override
	double calc(const Point&, const Point&) override;
};

class RandomPointGenerator : public RandomObjectGenerator<Point>
{
  public:
	enum Distribution { Uniform };

  protected:
	int d, id;
	double min_val, max_val;
	Distribution distribution;

  public:
	RandomPointGenerator(int d, double min_val, double max_val, Distribution distribution = Uniform);
	Point* create_random_object() override;
};

// Point class implementation.

Point::Point() { }

Point::Point(const Point &p)
{
	this->id = p.id;
	this->label = p.label;
	for (std::vector<double>::const_iterator it = p.data.begin(); it != p.data.end(); ++it)
	{
		this->data.push_back(*it);
	}
}

Point::Point(int id, int d, std::string label)
{
	this->id = id;
	this->label = label;
	for (int i = 0; i < d; i++)
		data.push_back(0);
}

Point::Point(int id, const std::vector<double>& data, std::string label)
{
	this->id = id;
	this->label = label;
	for (std::vector<double>::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		this->data.push_back(*it);
	}
}

int Point::get_dim_cnt() const
{
	return data.size();
}

double Point::get_dim_val(int dim) const
{
	if (dim < 0 || dim >= data.size())
		throw std::out_of_range("Point::get_dim_val - Invalid dimension.");
	
	return data[dim];
}

void Point::set_dim_val(int dim, double val)
{
	if (dim < 0 || dim >= data.size())
		throw std::out_of_range("Point::set_dim_val - Invalid dimension.");
	
	data[dim] = val;
}

bool Point::operator<(const Point& p)
{
	return id < p.id;
}

std::string Point::to_string(bool brief)
{
	if (brief)
		return std::to_string(id);
	
	std::stringstream sout;
	sout << label;
	std::vector<double>::iterator data_it = data.begin();
	while (data_it != data.end())
	{
		sout << " " << *data_it;
		data_it++;
	}
	return sout.str();
}

void Point::from_string(std::string str, bool brief)
{
	if (brief) throw NotImplementedException();
	
	std::istringstream sin(str);
	double value;
	char delimiter;
	while (sin >> value)
	{
		sin >> delimiter;
		data.push_back(value);
	}
}

RawDataInstance* Point::to_raw_data_instance(const std::type_info& raw_instance_type)
{
	if (raw_instance_type == typeid(CsvDataInstance<double>))
		return new CsvDataInstance<double>(data, label);
	
	throw std::invalid_argument("Point::to_raw_data_instance - RawDataInstance type incompatible with Point.");
}

void Point::from_raw_data_instance(RawDataInstance* instance)
{
	CsvDataInstance<double>* casted_instance_dbl = dynamic_cast<CsvDataInstance<double>*>(instance);
	if (casted_instance_dbl != nullptr)
	{
		label = casted_instance_dbl->get_label();
		data.clear();
		for (int i = 0; i < casted_instance_dbl->size(); i++)
			data.push_back((*casted_instance_dbl)[i]);
		return;
	}

	CsvDataInstance<int>* casted_instance_int = dynamic_cast<CsvDataInstance<int>*>(instance);
	if (casted_instance_int != nullptr)
	{
		label = casted_instance_int->get_label();
		data.clear();
		for (int i = 0; i < casted_instance_int->size(); i++)
			data.push_back((*casted_instance_int)[i]);
		return;
	}

	throw std::invalid_argument("Point::from_raw_data_instance - RawDataInstance type incompatible with Point.");
}

// PointL2Dist class implementation.

PointL2Dist *PointL2Dist::instance;

PointL2Dist::PointL2Dist()
{
}

PointL2Dist& PointL2Dist::get_inst()
{
	if (instance == nullptr)
		instance = new PointL2Dist;
	return *instance;
}

double PointL2Dist::calc(const Point &p1, const Point &p2)
{
	if (p1.get_dim_cnt() != p2.get_dim_cnt())
		throw new IncompatibleException;
	
	double sum = 0;
	for (int i = 0; i < p1.get_dim_cnt(); i++)
	{
		double diff = p1.get_dim_val(i) - p2.get_dim_val(i);
		sum += diff * diff;
	}
	return sqrt(sum);
}

// PointDTWDist class implementation.

PointDTWDist *PointDTWDist::instance;

PointDTWDist::PointDTWDist()
{
}

PointDTWDist& PointDTWDist::get_inst()
{
	if (instance == nullptr)
		instance = new PointDTWDist;
	return *instance;
}

double PointDTWDist::calc(const Point &p1, const Point &p2)
{
	int n = p1.get_dim_cnt();
	int m = p2.get_dim_cnt();
	double** matrix = new double*[n+1];
	for (int i = 0; i <= n; i++) {
		matrix[i] = new double[m+1];
		for (int j = 0; j <= m; j++)
			matrix[i][j] = std::numeric_limits<double>::infinity();
	}    
	matrix[0][0] = 0;
    
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++) {
			double cost = std::abs(p1.get_dim_val(i-1) - p2.get_dim_val(j-1));
			matrix[i][j] = cost + std::min(std::min(matrix[i-1][j], matrix[i][j-1]),matrix[i-1][j-1]);
		}

	double dtw = matrix[n][m];

	for (int i = 0; i <= n; ++i)
		delete [] matrix[i];
	delete [] matrix;

	return dtw;
}

// RandomPointGenerator class implementation.

RandomPointGenerator::RandomPointGenerator(int d, double min_val, double max_val, Distribution distribution) : id(0), d(d), min_val(min_val), max_val(max_val), distribution(distribution) { }

Point* RandomPointGenerator::create_random_object()
{
	std::vector<double> point_data;
	if (distribution == Distribution::Uniform)
		for (int i = 0; i < d; i++)
		{
			double rand_val = min_val + ((double)rand() / RAND_MAX) * (max_val - min_val);
			point_data.push_back(rand_val);
		}
	Point* p = new Point(0, point_data);
	p->set_id(id++);
	return p;
}

} // namespace knng
#endif