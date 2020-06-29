#ifndef UTILS_H
#define UTILS_H

#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_set>
#include <exception>

namespace knng
{

// PointersPairHash implements a hash function for the pair of pointers to objects of type T.
template <class T>
struct PointersPairHash
{
	inline std::size_t operator() (const std::pair<T*, T*> &p)  const throw()
	{
		T* first = p.first;
		T* second = p.second;
		if (first > second)
			std::swap(first, second);
		auto h1 = std::hash<T*>{}(first);
		auto h2 = std::hash<T*>{}(second);
		return h1 ^ (h2 + 0x9e3779b9 + (h1<<6) + (h1>>2));  
	}
};

// PointersPairEqual implements a equals function for the pair of pointers to objects of type T.
template <class T>
struct PointersPairEqual
{
	bool operator()(const std::pair<T*, T*> &a1, const std::pair<T*, T*> &a2) const
	{
		return (a1.first == a2.first && a1.second == a2.second) ||
			(a1.first == a2.second && a1.second == a2.first);
	}
};

struct PathDescriptor
{
	std::string folder_path;
	std::string filename;
	std::string filename_without_ext;
	std::string ext;
};

PathDescriptor splitpath(const std::string& str)
{
	PathDescriptor path_desc;
	std::unordered_set<char> delims{'\\', '/'};
	
	int last_delim_pos = -1;
	std::unordered_set<char>::iterator delims_it = delims.begin();
	while (delims_it != delims.end())
	{
		int delim_pos = str.rfind(*delims_it);
		if (delim_pos > last_delim_pos)
			last_delim_pos = delim_pos;
		delims_it++;
	}

	path_desc.folder_path = last_delim_pos >= 0 ? str.substr(0, last_delim_pos + 1) : ".";
	path_desc.filename = str.substr(last_delim_pos + 1);
	int dot_pos = path_desc.filename.rfind('.');
	path_desc.filename_without_ext = path_desc.filename.substr(0, dot_pos);
	path_desc.ext = path_desc.filename.substr(dot_pos + 1);

	return path_desc;
}

class NullOutBuf : public std::streambuf {
  public:
	virtual std::streamsize xsputn(const char * s, std::streamsize n) {
		return n;
	}

	virtual int overflow(int c) {
		return 1;
	}
};

class NullOutStream : public std::ostream
{
  public:
	NullOutStream() : std::ostream(&buf) {}
  private:
	NullOutBuf buf;
};

bool file_exists (const std::string& name)
{
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}

bool output_to_file(std::string file_path, std::string content)
{
	boost::filesystem::path p(file_path);
	if (p.parent_path() != "")
		boost::filesystem::create_directories(p.parent_path());
	std::fstream stream(file_path, std::fstream::out);
	if (!stream.is_open()) 
		return false;
	stream << content;
	stream.close();
	return true;
}

std::string replace_str(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = -1;
	while ((start_pos = str.find(from, start_pos + 1)) != std::string::npos)
    	str = str.replace(start_pos, from.length(), to);
    return str;
}

double get_epsilon()
{
	double third= 1.0/3.0;
	std::stringstream s;
	s << third;
	return (1 - std::fabs(3*std::stod(s.str()))) * 10;
}

} // namespace knng
#endif