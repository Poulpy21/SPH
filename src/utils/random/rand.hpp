
#ifndef MY_RAND_H
#define MY_RAND_H

#include <cstdlib>
#include <cassert>
#include <QGLViewer/vec.h>

class Random {
        public :
                static float randf() {
                        return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                }

                static float randf(float LO, float HI) {
                        return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                }
                
                static int randi(int LO, int HI) {
                        return LO + int(static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
                }

                static qglviewer::Vec randBoxPos(qglviewer::Vec xmin, qglviewer::Vec xmax) {
                    assert(xmin.x <= xmax.x && xmin.y <= xmax.y && xmin.z <= xmax.z);
                    qglviewer::Vec dX = xmax - xmin;
                    qglviewer::Vec x0 = xmin + qglviewer::Vec(dX.x*Random::randf(), dX.y*Random::randf(), dX.z*Random::randf());

                    return x0;
                }
};

#endif /* end of include guard: MY_RAND_H */

