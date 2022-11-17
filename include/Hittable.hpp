#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "Ray.hpp"

// class material;

struct hit_record {
    point3 p;
    vec3 normal;
    // shared_ptr<material> mat_ptr;
    float t;
    bool frontFace;

    __device__ inline void setFaceNormal(Ray const& r, vec3 const& outwardNormal) {
        frontFace = dot(r.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
public:
    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, hit_record &rec) const = 0;
};

#endif /* HITTABLE_HPP */