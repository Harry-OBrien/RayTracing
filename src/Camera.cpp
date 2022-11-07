#include "Camera.hpp"

Ray Camera::getRay(double u, double v) const {
    return Ray(origin, llCorner + u*horizontal + v*vertical - origin);
}