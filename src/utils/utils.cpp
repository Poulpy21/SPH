#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

using namespace log4cpp;

namespace utils {

	const std::string toStringMemory(unsigned long bytes) {

		std::stringstream ss;

		const char prefix[] = {' ', 'K', 'M', 'G', 'T', 'P'};
		unsigned long val = 1;
		for (int i = 0; i < 6; i++) {
			if(bytes < 1024*val) {
				ss << round(100*static_cast<float>(bytes)/val)/100.0 << prefix[i] << 'B';
				break;
			}
			val *= 1024;
		}

		const std::string str(ss.str());
		return str;
	 }
	
}

