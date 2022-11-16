#ifndef UTIL_HPP
#define UTIL_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <random>

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

__host__ inline double random_float() {
    //random R in [0, 1)
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

__host__ inline double random_float(double min, double max) {
    // [min, max)
    return min + (max-min)*random_float();
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;

    return x;
}

#endif /* UTIL_HPP */