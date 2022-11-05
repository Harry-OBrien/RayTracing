#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;

struct vec3 {
    double e[3];

    vec3() : e{0,0,0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    // Access operators
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    vec3 operator-() const;
    vec3& operator+=(const vec3 &v);
    vec3& operator*=(const double t) ;
    vec3& operator/=(const double t);
    double length() const;
    double length_squared() const;

};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using colour = vec3;    // RGB color

#endif /* VEC3_H */