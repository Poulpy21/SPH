
#ifndef GLUTILS_H
#define GLUTILS_H

#include "headers.hpp"

namespace utils {
		void glAssert(const std::string &file, int line, bool abort = true);
		bool isExtensionSupported(const char *extList, const char *extension);
}

#endif /* end of include guard: GLUTILS_H */
