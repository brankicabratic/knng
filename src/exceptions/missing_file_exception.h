#ifndef MISSING_FILE_EXCEPTION_H
#define MISSING_FILE_EXCEPTION_H

#include <stdexcept>

namespace knng
{

class MissingFileException : public std::logic_error
{
  public:
        MissingFileException () : std::logic_error{"Missing file."} {}
};

} // namespace knng
#endif