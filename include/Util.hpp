#ifndef UTIL_HPP
#define UTIL_HPP

#include <cmath>
#include <limits>
#include <memory>

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

// Common headers
#include "Ray.hpp"
#include "vec3.hpp"

#endif /* UTIL_HPP */