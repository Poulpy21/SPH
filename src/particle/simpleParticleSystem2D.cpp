
#include "simpleParticleSystem2D.hpp"
#include "defines.hpp"
#include "rand.hpp"
#include "utils.hpp"

#define __PRINT 0

SimpleParticleSystem2D::SimpleParticleSystem2D(unsigned int nParticles_, float dt_) :
    ParticleSystem(nParticles_, dt_),
    _particleProgram("particle"),
    _domainProgram("domain"),
    _particleVBO(0),
    _domainVBO(0),
    _particlePositions(0)
{
    log4cpp::log_console->infoStream() << "[ParticleSystem] Creating a new simple particle system 2D (" 
        << _nParticles << " particles, dt = " << _dt << "s).";
    initParticles();
    computeInitialEulerStep();

    //Particle poisitions VBO and array
    glGenBuffers(1, &_particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _particleVBO);	
    glBufferData(GL_ARRAY_BUFFER, _nParticles*3*sizeof(float), NULL, GL_DYNAMIC_DRAW);

    //Domain Quad
    const GLfloat quadVertex[] = { 0.0f, 0.0f  , 0.0f ,
        0.0f, _h  , 0.0f ,
        _w  , _h  , 0.0f ,
        _w  , 0.0f  , 0.0f };

    glGenBuffers(1, &_domainVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _domainVBO);	
    glBufferData(GL_ARRAY_BUFFER, 4*3*sizeof(float), quadVertex, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);	

    _particlePositions = new GLfloat[3*_nParticles];

    makeParticleProgram();
    makeDomainProgram();

    makeGrid();
}

SimpleParticleSystem2D::~SimpleParticleSystem2D() {
    delete [] _particlePositions;

    for (unsigned int i = 0; i < _gridHeight; i++) {
        delete [] _neighborGrid[i];
    }
    delete [] _neighborGrid;
}

void SimpleParticleSystem2D::initParticles() 
{
    log4cpp::log_console->infoStream() << "[ParticleSystem] Creating particles...";

    float x[2],v[2],a[2];
    for (unsigned int i = 0; i < _nParticles; i++) {
        x[0] = Random::randf(0,_w/2.0f);
        x[1] = Random::randf(_h/5.0f,_h);

        v[0] = 0.0f; 
        v[1] = 0.0f;

        a[0] = 0.0f;
        a[1] = 0.0f;

        this->_particles.push_back(Particle<2>(x,v,a,_m,_rho_0,_dh)); 
    }
}

void SimpleParticleSystem2D::updateGrid() 
{
    log4cpp::log_console->debugStream() << "[ParticleSystem] Updating Neighbor Grid...";
    for (unsigned i = 0; i < _gridHeight; i++) {
        for (unsigned j = 0; j < _gridWidth; j++) {
            _neighborGrid[i][j].clear();
        }
    }

    for(Particle<2u> &p : _particles) {
        _neighborGrid[p.i()][p.j()].push_back(&p);
    }
}

void SimpleParticleSystem2D::computeAccelerations() 
{
    log4cpp::log_console->debugStream() << "[ParticleSystem] Computing accelerations...";

    int i, j;
    bool print = true;

    for(Particle<2u> &p : _particles) {
        i = p.i();
        j = p.j();

        assert(i < int(_gridHeight));
        assert(j < int(_gridWidth));

        p.rho() = 0.0f;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if(i+di >= int(_gridHeight) || i+di < 0 || j+dj >= int(_gridWidth) || j+dj < 0)
                    continue;


                for(Particle<2u> *n : _neighborGrid[i+di][j+dj]) {
                    if(p.id() ==  n->id())
                        continue;
                    p.rho() += n->m()*W(p.x(), n->x());
                }
            }
        }

        p.P() = _P_0*(powf(p.rho()/_rho_0, 7) - 1); 

#ifdef __PRINT
        if(print) {
            std::cout << "RHO="<<p.rho() << "\tRHO/RHO0=" << p.rho()/_rho_0  << "\t" << "P=" << p.P() << std::endl;
            print = false;
        }
#endif
    }

    float X0,X1,V0,V1;
    float dP, dPx, dPy;
    float d2V, d2Vx, d2Vy;

    print = true;

    for(Particle<2u> &p : _particles) {

        i = p.i();
        j = p.j();

        dPx = 0.0f;
        dPy = 0.0f;
        d2Vx = 0.0f;
        d2Vy = 0.0f;

        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if(i+di >= int(_gridHeight) || i+di < 0 || j+dj >= int(_gridWidth) || j+dj < 0)
                    continue;

                for(Particle<2u> *n : _neighborGrid[i+di][j+dj]) {
                    if(p.id() ==  n->id())
                        continue;
                    X0 = p.x()[0] - n->x()[0];
                    X1 = p.x()[1] - n->x()[1];

                    V0 = p.x()[0] - n->x()[0];
                    V1 = p.x()[1] - n->x()[1];

                    dP = n->m()*(p.P()/(p.rho()*p.rho()) + n->P()/(n->rho()*n->rho()));

                    dPx += dP*D_Wx(p.x(),n->x());
                    dPy += dP*D_Wy(p.x(),n->x());

                    d2V = 2*n->m()/n->rho() 
                        *(X0*D_Wx(p.x(),n->x()) + X1*D_Wy(p.x(),n->x())) 
                        /(X0*X0 + X1*X1 + 0.01f*_dh*_dh);

                    d2Vx += V0*d2V;
                    d2Vy += V1*d2V;

#ifdef __PRINT
                    if(print) {
                        std::cout << "Particle " << n << " is a friend !" << n->x()[0] << "," << n->x()[1] << std::endl;
                        printf("W = %f\n",W(p.x(),n->x()));
                        printf("(D_Wx,D_Wy)=(%f,%f)\n",D_Wx(p.x(),n->x()),D_Wy(p.x(),n->x()));
                        printf("(dP,d2V)=(%f,%f)\n",dP,d2V);
                        printf("(dPx,dPy)=(%f,%f)\n",dPx,dPy);
                        printf("(d2Vx,d2Vz)=(%f,%f)\n",d2Vx,d2Vy);
                    }
#endif
                }
            }
        }

        //reset
        p.a()[0] = 0.0f;
        p.a()[1] = 0.0f;

        //acceleration due to pressure
        //p.a()[0] += -dPx;
        //p.a()[1] += -dPy;

        //acceleration due to viscosity
        //p.a()[0] += _nu*d2Vx;
        //p.a()[1] += _nu*d2Vy;

        //acceleration due to gravity
        p.a()[0] += 0;
        p.a()[1] += -_g;

        //damping
        p.a()[0] += -_k*p.v()[0];
        p.a()[1] += -_k*p.v()[1];

#ifdef __PRINT
        if(print) {
            printf("Pressure Gradient \t(dPx,dPy)=(%f,%f)\n",dPx,dPy);
            printf("Velocity Laplacian \t(d2Vx,d2Vy)=(%f,%f)\n",d2Vx,d2Vy);
            printf("Pressure Acceleration \t(Apx,Apy)=(%f,%f)\n", 1/p.rho()*dPx, 1/p.rho()*dPy);
            printf("Viscosity Acceleration \t(Avx,Avy)=(%f,%f)\n", _nu*d2Vx, _nu*d2Vy);
            printf("Gravity Acceleration \t(Agx,Agy)=(%f,%f)\n", 0.0f, -_g);
            print=false;
        }
#endif
    }
}

void SimpleParticleSystem2D::computeSpatialInteractions() 
{
    log4cpp::log_console->debugStream() << "[ParticleSystem] Computing spatial interactions...";
}

void SimpleParticleSystem2D::integrateScheme() 
{
    log4cpp::log_console->debugStream() << "[ParticleSystem] Integrating Scheme...";
    float v_halfStepX, v_halfStepY;

    bool print = true;
    for(Particle<2u> &p : _particles) {
        v_halfStepX = p.v_old()[0] + _dt*p.a()[0];        
        v_halfStepY = p.v_old()[1] + _dt*p.a()[1];        

        p.x()[0] = std::max(0.0f,std::min(_w,p.x()[0] + _dt*v_halfStepX));
        p.x()[1] = std::max(0.0f,std::min(_h,p.x()[1] + _dt*v_halfStepY));

        p.i() = ceil(p.x()[1])/(2*_dh);
        p.j() = ceil(p.x()[0])/(2*_dh);

        p.v()[0] = 1/2.0f*(p.v_old()[0] + v_halfStepX);
        p.v()[1] = 1/2.0f*(p.v_old()[1] + v_halfStepY);

#ifdef __PRINT
        if(print) {
            printf("(X',Y')=(%f,%f)\n", p.x()[0], p.x()[1]);
            printf("(Vx',Vy')=(%f,%f)\n", p.v()[0], p.v()[1]);
            printf("(Ax',Ay')=(%f,%f)\n", p.a()[0], p.a()[1]);
            print = false;
        }
#endif

        p.v_old()[0] = v_halfStepX;
        p.v_old()[1] = v_halfStepY;
    }
}

void SimpleParticleSystem2D::drawDownwards(const float *currentTransformationMatrix) 
{
    log4cpp::log_console->debugStream() << "[ParticleSystem] Drawing...";
    drawDomain();
    drawParticles();
}

void SimpleParticleSystem2D::drawDomain(const float *currentTransformationMatrix) 
{
}

void SimpleParticleSystem2D::drawParticles(const float *currentTransformationMatrix) 
{
    unsigned int i = 0u;
    for(Particle<2> &p : this->_particles) {
        _particlePositions[i++] = p.x()[0];
        _particlePositions[i++] = p.x()[1];
        _particlePositions[i++] = 0.0f;
    }

    _domainProgram.use();
    glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);
    glUniformMatrix4fv(_domainUniformLocations["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
    glUniform1f(_domainUniformLocations["xmin"], 0.0f);
    glUniform1f(_domainUniformLocations["xmax"], _w/2);
    glUniform1f(_domainUniformLocations["ymin"], _h/5);
    glUniform1f(_domainUniformLocations["ymax"], _h);
    glUniform1f(_domainUniformLocations["ww"], _w);
    glUniform1f(_domainUniformLocations["hh"], _h);

    glBindBuffer(GL_ARRAY_BUFFER, _domainVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_QUADS, 0, 4);

    _particleProgram.use();
    glEnable(GL_POINT_SMOOTH);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);
    glUniformMatrix4fv(_particleUniformLocations["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
    glPointSize(10.0);
    glBindBuffer(GL_ARRAY_BUFFER, _particleVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _nParticles*3*sizeof(GLfloat), _particlePositions);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, _nParticles);
    glDisable(GL_POINT_SMOOTH);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);

}

void SimpleParticleSystem2D::makeParticleProgram() {
    _particleProgram.bindAttribLocations("0", "particlePos");
    _particleProgram.bindFragDataLocation(0, "out_colour");
    _particleProgram.bindUniformBufferLocations("0", "projectionView");

    _particleProgram.attachShader(Shader("shaders/particle/particle_vs.glsl", GL_VERTEX_SHADER));
    _particleProgram.attachShader(Shader("shaders/particle/particle_fs.glsl", GL_FRAGMENT_SHADER));

    _particleProgram.link();

    _particleUniformLocations = _particleProgram.getUniformLocationsMap("modelMatrix", true);
}

void SimpleParticleSystem2D::makeDomainProgram() {
    _domainProgram.bindAttribLocations("0", "vertexPos");
    _domainProgram.bindFragDataLocation(0, "out_colour");
    _domainProgram.bindUniformBufferLocations("0", "projectionView");

    _domainProgram.attachShader(Shader("shaders/domain/domain_vs.glsl", GL_VERTEX_SHADER));
    _domainProgram.attachShader(Shader("shaders/domain/domain_fs.glsl", GL_FRAGMENT_SHADER));

    _domainProgram.link();

    _domainUniformLocations = _domainProgram.getUniformLocationsMap("modelMatrix xmin xmax ymin ymax ww hh", false);
}

void SimpleParticleSystem2D::makeGrid() {
    _gridWidth = ceil(_w/(2*_dh)) + 1;
    _gridHeight = ceil(_h/(2*_dh)) + 1;

    log4cpp::log_console->infoStream() << "[ParticleSystem] Grid Dimension is " << utils::toStringDimension(_gridWidth, _gridHeight, 0u) << ".";

    _neighborGrid = new std::vector<Particle<2u>*>*[_gridHeight];
    for (unsigned int i = 0; i < _gridHeight; i++) {
        _neighborGrid[i] = new std::vector<Particle<2u>*>[_gridWidth];
    }
}

float SimpleParticleSystem2D::cubicSpline(float q) {
    float r;

    if(q >= 0.0f && q < 1.0f) 
        r = 2.0f/3.0f - q*q + 1/2.0f*q*q*q;
    else if (q >= 1.0f && q < 2.0f)
        r = 1/6.0f*(2.0f-q)*(2.0f-q)*(2.0f-q);
    else 
        r = 0.0f;

    return r;
}

float SimpleParticleSystem2D::D_cubicSpline(float q) {
    float r;

    if(q >= 0.0f && q < 1.0f) 
        r = -2.0f*q + 3.0/2.0f*q*q;
    else if (q >= 1.0f && q < 2.0f)
        r = -1/2.0f*(2.0f-q)*(2.0f-q);
    else 
        r = 0.0f;

    return r;
}

float SimpleParticleSystem2D::D2_cubicSpline(float q) {
    float r;

    if(q >= 0.0f && q < 1.0f) 
        r = 3.0f*q - 2.0f;
    else if (q >= 1.0f && q < 2.0f)
        r = 2.0f-q;
    else 
        r = 0.0f;

    return r;
}

float SimpleParticleSystem2D::W(float* xi, float* xj) {
    float n2 = (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]);
    float q = sqrt(n2)/_dh;
    return 15.0f/(7.0f*M_PI*_dh*_dh)*cubicSpline(q);
}

float SimpleParticleSystem2D::D_Wx(float* xi, float* xj) {
    float n2 = (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]);
    float q = sqrt(n2)/_dh;

    return 15.0f/(7.0f*M_PI*_dh*_dh) * D_cubicSpline(q) * (xi[0] - xj[0]) / (_dh*sqrt(n2));
}

float SimpleParticleSystem2D::D_Wy(float* xi, float* xj) {
    float n2 = (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]);
    float q = sqrt(n2)/_dh;

    return 15.0f/(7.0f*M_PI*_dh*_dh) * D_cubicSpline(q) * (xi[1] - xj[1]) / (_dh*sqrt(n2));
}

float SimpleParticleSystem2D::D2_W(float* xi, float* xj) {
    float n2 = (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]);
    float q = sqrt(n2)/_dh;
    return 15.0f/(7.0f*M_PI*_dh*_dh) * ( D2_cubicSpline(q)/(_dh*_dh) + D_cubicSpline(q)*2.0f/(_dh*sqrt(n2)) );
}



