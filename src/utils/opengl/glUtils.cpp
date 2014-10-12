
#include "log.hpp"
#include "glUtils.hpp"

using namespace log4cpp;

namespace utils {

	bool contextError = false;

	void glAssert(const std::string &file, int line, bool abort) {

		GLenum error = glGetError();

		switch(error) {
			case GL_NO_ERROR:
				break; 
			case GL_INVALID_ENUM:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_ENUM\n\t\t"
					"An unacceptable value is specified for an enumerated argument. "
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;

			case GL_INVALID_VALUE:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_VALUE\n\t\t"
					"A numeric argument is out of range."
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;
			case GL_INVALID_OPERATION:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_OPERATION\n\t\t"
					"The specified operation is not allowed in the current state."
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_FRAMEBUFFER_OPERATION\n\t\t"
					"The framebuffer object is not complete. "
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;
			case GL_OUT_OF_MEMORY:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_OUT_OF_MEMORY\n\t\t"
					"There is not enough memory left to execute the command. The state of the case GL"
					"The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
					break;
			case GL_STACK_UNDERFLOW:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_STACK_UNDERFLOW\n\t\t"
					"An attempt has been made to perform an operation that would cause an internal stack to underflow.";
				break;
			case GL_STACK_OVERFLOW:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_STACK_OVERFLOW\n\t\t"
					"An attempt has been made to perform an operation that would cause an internal stack to overflow.";
				break;
			default:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << "\n\t\t"
					"Unknown error !";
		}

		if(error != GL_NO_ERROR && abort)
			exit(EXIT_FAILURE);
	}

	bool isExtensionSupported(const char *extList, const char *extension)
	{
		const char *start;
		const char *where, *terminator;

		where = strchr(extension, ' ');
		if (where || *extension == '\0')
			return false;

		for (start=extList;;) {
			where = strstr(start, extension);

			if (!where)
				break;

			terminator = where + strlen(extension);

			if ( where == start || *(where - 1) == ' ' )
				if ( *terminator == ' ' || *terminator == '\0' )
					return true;

			start = terminator;
		}

		return false;
	}

}
