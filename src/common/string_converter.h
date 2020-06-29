#ifndef STRING_CONVERTER_H
#define STRING_CONVERTER_H

#include <string>

namespace knng
{

// Abstract class that is extended by each class that supports conversion to string.
class StringConverter
{
  public:
	virtual std::string to_string(bool brief = true) { return ""; }  // Returns string representation of the object.
	virtual void from_string(std::string str, bool brief = true) { } // Initializes object from the string.
};

} // namespace knng
#endif