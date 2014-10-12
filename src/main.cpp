
#include "headers.hpp"

#include <cstdlib>
#include <cassert>
#include <ostream>
#include <iostream>
#include <qapplication.h>

#include "globals.hpp"
#include "log.hpp"
#include "rand.hpp"
#include "texture.hpp"
#include "renderRoot.hpp"
#include "viewer.hpp"


#include "simpleParticleSystem2D.hpp"

using namespace log4cpp;

int main(int argc, char** argv) {
    
        //random
        srand(time(NULL));

        //logs
        log4cpp::initLogs();

        //cuda
        CudaUtils::logCudaDevices(*log_console);

        log_console->infoStream() << "[Rand Init] ";
        log_console->infoStream() << "[Logs Init] ";

        // glut initialisation (mandatory) 
        glutInit(&argc, argv);
        log_console->infoStream() << "[Glut Init] ";

        // Read command lines arguments.
        QApplication application(argc,argv);
        log_console->infoStream() << "[Qt Init] ";

        // Instantiate the viewer (mandatory)
        Viewer *viewer = new Viewer();
        viewer->setWindowTitle("SPH");
        viewer->show();

        //glew initialisation (mandatory)
        log_console->infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

        //global vars
        Globals::init();
        Globals::print(std::cout);
        Globals::check();
        Globals::viewer = viewer;

        //texture manager
        Texture::init();

        log_console->infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";
        //FIN INIT//

        RenderRoot *root = new RenderRoot(); 

        ParticleSystem<2> *system = new SimpleParticleSystem2D(1000u, 0.0001f);
        root->addChild("SPH_ParticleSystem", system);
        
        //Configure viewer
        viewer->addRenderable(root);

        //Run main loop.
        application.exec();

        //Exit
        alutExit();

        return EXIT_SUCCESS;
}

