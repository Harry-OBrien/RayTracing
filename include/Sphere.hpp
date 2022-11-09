#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "Hittable.hpp"
#include "vec3.hpp"
#include "material.hpp"

class Sphere : public Hittable {
    point3 center;
    double radius;
    double rad_sq;
    shared_ptr<material> mat_ptr;

public:
    Sphere() {}
    Sphere(point3 cen, double r, shared_ptr<material> m)
        : center(cen), radius(r), rad_sq(r*r) , mat_ptr(m)
    {}

    virtual bool hit(Ray const& r, double t_min, double t_max, hit_record &rec) const override;
};

#endif /* SPHERE_HPP */