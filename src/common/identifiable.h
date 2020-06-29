#ifndef IDENTIFIABLE_H
#define IDENTIFIABLE_H

#include <string>

namespace knng
{

// Abstract class that is extended by each class whose objects need to have IDs.
class Identifiable
{
  protected:
	int id;

  public:
	virtual int get_id()
	{
		return id;
	}

	virtual void set_id(int id)
	{
		this->id = id;
	} 
};

} // namespace knng
#endif