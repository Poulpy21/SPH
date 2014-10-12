#ifndef _CUSTOM_HEADERS_H
#define _CUSTOM_HEADERS_H

//always first !
#include <GL/glew.h>

#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cudaUtils.hpp"

#include <QWidget>
#include <QImage>
#include <QApplication>

#include <QGLViewer/qglviewer.h>
#include <QGLViewer/vec.h>
#include <QGLViewer/camera.h>
#include <QGLViewer/quaternion.h>

#endif /* end of include guard: HEADERS_H */

