
#include "headers.hpp"
#include "defines.hpp"

#include <cstdlib>
#include <cassert>
#include <ostream>
#include <iostream>
#include <qapplication.h>
#include <omp.h>

#include "globals.hpp"
#include "log.hpp"
#include "rand.hpp"
#include "texture.hpp"
#include "renderRoot.hpp"
#include "viewer.hpp"

#include "implicitTree.h"
#include "implicitFunc.h"

#include "renderRoot.hpp"

#include "simpleParticleSystem2D.hpp"
#include "marchingCubes.hpp"

#include "sharedSurfaceResource.hpp"

#include "vec.hpp"
#include "simplyLinkedList.hpp"

#include <atomic>

using namespace log4cpp;
using namespace cuda_gl_interop;
    
void printList(List<char>* list) {
    Node<char>* node = list->first; 
    unsigned int count = 0;
    while(node != 0 && node->data != 0) {
        std::cout << *(node->data);
        count++;
        node = node->next;
    }
    std::cout << " ("<< count << " caracters)" << std::endl;
}

int main(int argc, char** argv) {

    //logs
    log4cpp::initLogs();
    log_console->infoStream() << "[Logs Init] ";

    //random
    srand(static_cast<unsigned int>(time(NULL)));
    log_console->infoStream() << "[Rand Init] ";

    //cuda
    //log_console->infoStream() << "[CUDA Init] ";
    //CudaUtils::logCudaDevices(*log_console);

    // glut initialisation (mandatory) 
    glutInit(&argc, argv);
    log_console->infoStream() << "[Glut Init] ";

    // Read command lines arguments.
    QApplication application(argc,argv);
    log_console->infoStream() << "[Qt Init] ";

    // Instantiate the viewer (mandatory)
    log_console->infoStream() << "[Viewer Init] ";
    Viewer *viewer = new Viewer();
    viewer->setWindowTitle("SPH");
    viewer->show();

    //glew initialisation (mandatory)
    log_console->infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

    //global vars
    Globals::init();
#ifdef _DEBUG
    Globals::print(std::cout);
#endif
    Globals::check();
    Globals::viewer = viewer;

    //texture manager
    Texture::init();

    //FIN INIT//
    log_console->infoStream() << "";
    log_console->infoStream() << "Data size check :"; // !! GL_BYTE != Glbyte !!
    log_console->infoStream() << "\tSizeOf(GLboolean) = " << sizeof(GLboolean);
    log_console->infoStream() << "\tSizeOf(GLbyte) = " << sizeof(GLbyte);
    log_console->infoStream() << "\tSizeOf(GLshort) = " << sizeof(GLshort);
    log_console->infoStream() << "\tSizeOf(GLint) = " << sizeof(GLint);
    log_console->infoStream() << "\tSizeOf(GLfloat) = " << sizeof(GLfloat);
    log_console->infoStream() << "\tSizeOf(GLdouble) = " << sizeof(GLdouble);

    log_console->infoStream() << "";
    log_console->infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";
    log_console->infoStream() << "";

    List<char> *list = new List<char>();
    std::string str("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

#pragma omp parallel shared(list, str)
{
    #pragma omp for
    for(unsigned int i=0; i<100*str.length();i++) {
        list->push_back(&str[i%str.length()]);
    }
    
    #pragma omp master 
    {
        printList(list);
    }
}

    std::cout << sizeof(std::atomic<int>) << std::endl;

    //RenderRoot *root = new RenderRoot(); 
    //viewer->addRenderable(root);

    //Run main loop.
    //application.exec();

    //Exit
    //alutExit();

    return EXIT_SUCCESS;
}

