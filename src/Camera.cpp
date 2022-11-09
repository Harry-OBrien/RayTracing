#include "Camera.hpp"

Ray Camera::getRay(double s, double t) const {
    vec3 rd = lens_radius * random_in_unit_disk();
    vec3 offset = u * rd.x() + v * rd.y();
    return Ray(
        origin + offset,
        lower_left_corner + s*horizontal + t*vertical - origin - offset,
        random_double(time0, time1)
    );
}