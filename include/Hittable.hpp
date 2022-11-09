#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "Ray.hpp"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    shared_ptr<material> mat_ptr;
    double t;
    bool frontFace;

    inline void setFaceNormal(Ray const& r, vec3 const& outwardNormal) {
        frontFace = dot(r.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
public:
    virtual bool hit(Ray const& r, double t_min, double t_max, hit_record &rec) const = 0;
};

#endif /* HITTABLE_HPP */