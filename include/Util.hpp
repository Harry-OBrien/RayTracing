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

// Util functions
__device__ inline float deg_to_rad(float degrees) {
    return degrees * M_PIf / 180.0f;
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

__host__ inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;

    return x;
}

#endif /* UTIL_HPP */