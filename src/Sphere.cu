#include "Sphere.hpp"

__device__ bool Sphere::hit(Ray const& r, float t_min, float t_max, hit_record &rec) const {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - rad_sq;
    float discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0)
        return false;

    float sqrtd = sqrt(discriminant);

    // Find nearest root that lies in the acceptable range
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        
        if(root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outwardNormal = (rec.p - center) / radius;
    rec.setFaceNormal(r, outwardNormal);
    // rec.mat_ptr = mat_ptr;
    
    return true;
}