
#ifndef GLUTILS_H
#define GLUTILS_H

#include "headers.hpp"

namespace utils {
		void glAssert(const std::string &file, int line, bool abort = true);
		bool isExtensionSupported(const char *extList, const char *extension);

        //texture initialization helpers
        GLenum internalFormatToValidExternalFormat(unsigned int internalFormat);
        GLenum internalFormatToValidExternalType(unsigned int internalFormat);

        size_t externalTypeToBytes(GLenum externalType);
        unsigned int externalFormatToChannelNumber(GLenum externalFormat);

        const std::string toStringInternalFormat(unsigned int internalFormat);
        const std::string toStringExternalFormat(GLenum externalFormat);
        const std::string toStringExternalType(GLenum externalType);
        
        const std::string toStringTextureTarget(GLenum textureTarget);
}

#endif /* end of include guard: GLUTILS_H */
