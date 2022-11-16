#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"

class Ray {
    point3 orig;
    vec3 dir;
    double tm;

public:
    __device__ Ray() {}
    __device__ Ray(point3 const& origin, vec3 const& direction)
        : orig(origin), dir(direction)
    {}

    __device__ Ray(const point3& origin, const vec3& direction, double time)
        : orig(origin), dir(direction), tm(time)
    {}

    __device__ point3 origin() const { return orig; }
    __device__ point3 direction() const { return dir; }
    __device__ double time() const    { return tm; }

    __device__ point3 at(double t) const {
        return orig + t*dir;
    };
};

#endif /* RAY_HPP */