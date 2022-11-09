#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class Ray {
    point3 orig;
    vec3 dir;
    double tm;

public:
    Ray() {}
    Ray(point3 const& origin, vec3 const& direction)
        : orig(origin), dir(direction)
    {}

    Ray(const point3& origin, const vec3& direction, double time)
        : orig(origin), dir(direction), tm(time)
    {}

    point3 origin() const { return orig; }
    point3 direction() const { return dir; }
    double time() const    { return tm; }

    point3 at(double t) const {
        return orig + t*dir;
    };
};

#endif /* RAY_HPP */