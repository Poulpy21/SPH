
#ifndef PARTICLE_H
#define PARTICLE_H

#include "utils.hpp"

template <unsigned int N>
class Particle {
    public:
        Particle(float x0[N], float v0[N], float a0[N],
                float m, float rho, float dh);

        Particle(const Particle &P);
        virtual ~Particle ();

        const float* x() const;
        const float* v() const;
        const float* a() const;
        const float* v_old() const;
        float m() const;
        float rho() const;
        float P() const;
        unsigned int i() const;
        unsigned int j() const;
        unsigned int id() const;
        
        float* x();
        float* v();
        float* a();
        float* v_old();
        float& m();
        float& rho();
        float& P();
        unsigned int & i();
        unsigned int & j();

    private:
        float _x[N], _v[N], _a[N];
        float _m, _rho, _P;

        float _v_old[N];
        unsigned int _i, _j;
    
        unsigned int _id;
        static unsigned int __id__;
};
        

template <unsigned int N>
unsigned int Particle<N>::__id__ = 0u;

template <unsigned int N>
std::ostream & operator << (std::ostream &os, Particle<N> &P) {
    os << "Particule" << std::endl;
    os << " X" << utils::toStringVec3(P.x()[0],N>1?P.x()[1]:0,N>2?P.x()[2]:0)<< std::endl;
    os << " V" << utils::toStringVec3(P.v()[0],N>1?P.v()[1]:0,N>2?P.v()[2]:0)<< std::endl;
    os << " A" << utils::toStringVec3(P.a()[0],N>1?P.a()[1]:0,N>2?P.a()[2]:0)<< std::endl;
    os << " V-1/2" << utils::toStringVec3(P.v_old()[0],N>1?P.v_old()[1]:0,N>2?P.v_old()[2]:0)<< std::endl;
    os << " N(" << P.i() << "," << P.j() << ")" << std::endl;
    os << " P="<<P.P() << "  rho=" << P.rho();
    return os;
}

template <unsigned int N>
Particle<N>::Particle(float x_[N], float v_[N], float a_[N],
        float m_, float rho_, float dh_) :
    _m(m_), _rho(rho_),
    _P(0), 
    _i(ceil(x_[1]/(2*dh_))), _j(ceil(x_[0]/(2*dh_))),
    _id(__id__++)
{
    for (unsigned int i = 0; i < N; i++) {
        _x[i] = x_[i];
        _v[i] = v_[i];
        _a[i] = a_[i];
        _v_old[i] = 0;
    }
}
        
template <unsigned int N>
Particle<N>::Particle(const Particle &P) {
    this->_m = P.m();
    this->_rho = P.rho();
    this->_i = P.i();
    this->_j = P.j();
    this->_id = P.id();
    for (unsigned int i = 0; i < N; i++) {
        this->_x[i] = P.x()[i];
        this->_v[i] = P.v()[i];
        this->_a[i] = P.a()[i];
        this->_v_old[i] = P.v_old()[i];
    }
}

template <unsigned int N>
Particle<N>::~Particle() {
}

template <unsigned int N>
const float* Particle<N>::x() const { return _x; }

template <unsigned int N>
const float* Particle<N>::v() const { return _v; }

template <unsigned int N>
const float* Particle<N>::a() const { return _a; }

template <unsigned int N>
const float* Particle<N>::v_old() const { return _v_old; }

template <unsigned int N>
float* Particle<N>::x() { return _x; }

template <unsigned int N>
float* Particle<N>::v() { return _v; }

template <unsigned int N>
float* Particle<N>::a() { return _a; }

template <unsigned int N>
float* Particle<N>::v_old() { return _v_old; }

template <unsigned int N>
float Particle<N>::m() const { return _m; }

template <unsigned int N>
float Particle<N>::rho() const { return _rho; }

template <unsigned int N>
float Particle<N>::P() const { return _P; }

template <unsigned int N>
float& Particle<N>::m() { return _m; }

template <unsigned int N>
float& Particle<N>::rho() { return _rho; }

template <unsigned int N>
float& Particle<N>::P() { return _P; }

template <unsigned int N>
unsigned int Particle<N>::i() const { return _i; }

template <unsigned int N>
unsigned int Particle<N>::j() const { return _j; }

template <unsigned int N>
unsigned int& Particle<N>::i() { return _i; }

template <unsigned int N>
unsigned int& Particle<N>::j() { return _j; }

template <unsigned int N>
unsigned int Particle<N>::id() const { return _id; }

#endif /* end of include guard: PARTICLE_H */
