#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "Hittable.hpp"
#include "Vec3.hpp"
// #include "material.hpp"

class Sphere : public Hittable {
    point3 center;
    float radius;
    float rad_sq;

public:
    __device__ Sphere() {}
    __device__ Sphere(point3 cen, float r)
        : center(cen), radius(r), rad_sq(r*r)
    {}

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, hit_record &rec) const override;
};

#endif /* SPHERE_HPP */