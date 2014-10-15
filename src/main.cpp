
#include "headers.hpp"
#include "defines.hpp"

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

#include "implicitTree.h"
#include "implicitFunc.h"


#include "simpleParticleSystem2D.hpp"

using namespace log4cpp;

int main(int argc, char** argv) {

        CHECK_CUDA_ERRORS(cudaFree(0));
        initPointers();
        node_s *n1 = makeDensityNode(d1); 
        node_s *n2 = makeDensityNode(d2); 
        node_s *n3 = makeDensityNode(d3); 
        node_s *n4 = makeOperatorNode(n1,n2,op1);
        node_s *n5 = makeOperatorNode(n1,n3,op2);
        node_s *n6 = makeOperatorNode(n2,n3,op1);
        node_s *n7 = makeOperatorNode(n4,n5,op1);
        node_s *n8 = makeOperatorNode(n4,n6,op2);
        node_s *n9 = makeOperatorNode(n7,n8,op1);

        node_s *n_d = makeDeviceTreeFromHost(n9);
        node_s *n_h = n9;

        float* X_h[3];
        float* X_d[3];
        float *res_h, *res_d;

        for (unsigned int i = 0u; i < 3u; i++) {
            X_h[i] = new float[10];
            CHECK_CUDA_ERRORS(cudaMalloc(X_d+i, 10*sizeof(float)));
            for (unsigned j = 0; j < 10; j++) {
                X_h[i][j] = j/10.0;
            }
            CHECK_CUDA_ERRORS(cudaMemcpy(X_d[i],X_h[i],10*sizeof(float),cudaMemcpyHostToDevice));
        }
    
        res_h = new float[1000];
        CHECK_CUDA_ERRORS(cudaMalloc(&res_d, 1000*sizeof(float)));
       
        //cpu side
        for (unsigned int i = 0; i < 1000; i++) {
            if(i%10==0) std::cout << std::endl;            
            if(i>=100) break;
            std::cout << "\t" << evalNode(n_h,(i%10)/10.0f, ((i/10)%10)/10.0f, (i/100)/10.0f);
        }

        //kernel
        computeTestKernel(X_d[0],X_d[1],X_d[2],res_d,n_d);
        checkKernelExecution();
        
        //copy back
        CHECK_CUDA_ERRORS(cudaMemcpy(res_h,res_d,1000*sizeof(float),cudaMemcpyDeviceToHost));

        //print
        for (unsigned int i = 0; i < 1000; i++) {
            if(i%10==0) std::cout << std::endl;            
            if(i>=100) break;
            std::cout << "\t" << res_h[i];
        }



        //logs
        log4cpp::initLogs();
        log_console->infoStream() << "[Logs Init] ";
        
        //random
        srand(time(NULL));
        log_console->infoStream() << "[Rand Init] ";

        //cuda
        log_console->infoStream() << "[CUDA Init] ";
        CudaUtils::logCudaDevices(*log_console);

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
#ifdef __DEBUG
        Globals::print(std::cout);
#endif
        Globals::check();
        Globals::viewer = viewer;

        //texture manager
        Texture::init();

        log_console->infoStream() << "";
        log_console->infoStream() << "";
        log_console->infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";
        
        exit(0);
        //FIN INIT//

        RenderRoot *root = new RenderRoot(); 

        ParticleSystem<2> *system = new SimpleParticleSystem2D(1000u, 0.001f);

        root->addChild("SPH_ParticleSystem", system);
        
        //Configure viewer
        viewer->addRenderable(root);

        //Run main loop.
        application.exec();

        //Exit
        alutExit();

        return EXIT_SUCCESS;
}

