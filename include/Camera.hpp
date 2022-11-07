#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "vec3.hpp"
#include "Ray.hpp"

class Camera {
public:
    point3 origin;
    point3 llCorner;
    vec3 horizontal, vertical;
public:
    Camera() {
        double const ASPECT_RATIO = 16.0 / 9.0;
        double viewportHeight = 2.0;
        double viewportWidth = ASPECT_RATIO * viewportHeight;
        double focalLength = 1.0;

        origin = point3(0, 0, 0);
        horizontal = vec3(viewportWidth, 0, 0);
        vertical = vec3(0, viewportHeight, 0);
        llCorner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focalLength);
    }

    Ray getRay(double u, double v) const;
};

#endif /* CAMERA_HPP */