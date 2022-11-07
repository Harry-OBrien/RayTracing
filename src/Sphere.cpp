#include "Sphere.hpp"

bool Sphere::hit(Ray const& r, double t_min, double t_max, hit_record &rec) const {
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(oc, r.direction());
    double c = oc.length_squared() - rad_sq;
    double discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0)
        return false;

    double sqrtd = sqrt(discriminant);

    // Find nearest root that lies in the acceptable range
    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        
        if(root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outwardNormal = (rec.p - center) / radius;
    rec.setFaceNormal(r, outwardNormal);

    return true;
}