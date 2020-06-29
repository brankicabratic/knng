#ifndef NOT_IMPLEMENTED_EXCEPTION_H
#define NOT_IMPLEMENTED_EXCEPTION_H

#include <stdexcept>

namespace knng
{

class NotImplementedException : public std::logic_error
{
  public:
        NotImplementedException () : std::logic_error{"Not yet implemented."} {}
};

} // namespace knng
#endif