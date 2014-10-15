
#ifndef GLOBALS_H
#define GLOBALS_H

#include "headers.hpp"
#include "viewer.hpp"
#include <ostream>
#include <string>

#include <QGLViewer/vec.h>
using namespace qglviewer;  // to use class Vec of the qglviewer lib

struct modelViewUniformBlock {
	GLfloat projectionMatrix[16];
	GLfloat viewMatrix[16];
	GLfloat cameraPosition[4];
	GLfloat cameraDirection[4];
	GLfloat cameraUp[4];
	GLfloat cameraRight[4];
};

class Globals {

	public:
		static void init();
		static void check();
		static void print(std::ostream &out);
	
		static const unsigned char *glVersion;
		static const unsigned char *glShadingLanguageVersion;
			
		static int glMax3DTextureSize;
		static int glMaxTextureSize;
		static int glMaxVertexAttribs;
		static int glMaxDrawBuffers;
		static int glMaxCombinedTextureImageUnits;

		static int glMaxVertexUniformBlocks;
		static int glMaxGeometryUniformBlocks;
		static int glMaxFragmentUniformBlocks;
		static int glMaxUniformBlockSize;
		
		static float *glPointSizeRange;
		static float glPointSizeGranularity;
		static float glPointSize;
		
		static Viewer *viewer;
		static unsigned int projectionViewUniformBlock;

        // Current position
        static float dt;
        static Vec pos;
        static Vec offset;
};
	
#endif /* end of include guard: GLOBALS_H */
