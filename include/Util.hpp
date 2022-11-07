#ifndef UTIL_HPP
#define UTIL_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const double INF = std::numeric_limits<double>::infinity();

// Util functions
inline double deg_to_rad(double degrees) {
    return degrees * M_PI / 180.0;
}

inline double random_double() {
    //random R in [0, 1)
    return ((double) rand() / (RAND_MAX)) + 1;
}

inline double random_double(double min, double max) {
    // [min, max)
    return min + (max-min)*random_double();
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;

    return x;
}

// Common headers
#include "Ray.hpp"
#include "vec3.hpp"

#endif /* UTIL_HPP */