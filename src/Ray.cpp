#include "Ray.hpp"

point3 Ray::at(double t) const {
    return orig + t*dir;
}