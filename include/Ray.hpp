#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class Ray {
    point3 orig;
    vec3 dir;

public:
    Ray() {}
    Ray(point3 const& origin, vec3 const& direction)
        : orig(origin), dir(direction)
    {}

    point3 origin() const { return orig; }
    point3 direction() const { return dir; }

    point3 at(double t) const;
};

#endif /* RAY_HPP */