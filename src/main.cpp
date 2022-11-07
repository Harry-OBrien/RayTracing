#include <iostream>

#include "Util.hpp"

#include "HittableList.hpp"
#include "Colour.hpp"
#include "Sphere.hpp"
#include "Camera.hpp"

colour rayColour(Ray const& r, Hittable const& world) {
    hit_record rec;
    if(world.hit(r, 0, INF, rec)) {
        return 0.5 * (rec.normal + colour(1,1,1));
    }

    vec3 unitDirection = unit_vector(r.direction());
    double t = 0.5 * (unitDirection.y() + 1.0);
    return vec3_lerp(colour(1,1,1), colour(0.5, 0.7, 1.0), t);
}

int main(int argc, char** argv) {
    //Image
    double aspect_ratio = 16.0 / 9.0;
    int const imageWidth = 1024;
    int const imageHeight = static_cast<int>(imageWidth / aspect_ratio);
    int const samplesPerPixel = 100;

    // World
    HittableList world;
    world.add(make_shared<Sphere>(point3(0, -100.5, -1), 100));
    world.add(make_shared<Sphere>(point3(0,0,-1), 0.5));

    // Camera
    Camera cam;

    // Render
    std::cout << "P3\n"
        << imageWidth
        << " "
        << imageHeight
        << "\n255"
        << std::endl;

    for (int j = imageHeight-1; j >= 0; j--) {
        std::cerr << "Scanlines remaining: "
            << j
            << " "
            << std::endl
            << std::flush;

        for (int i = 0; i < imageWidth; i++) {
            colour pixelColour;
            for (int s = 0; s < samplesPerPixel; s++) {
                double u = (i + random_double()) / (imageWidth - 1);
                double v = (j + random_double()) / (imageHeight - 1);
                Ray r = cam.getRay(u, v);
                pixelColour += rayColour(r, world);
            }
            write_colour(std::cout, pixelColour, samplesPerPixel);
        }
    }

    std::cerr << "\nDone!" << std::endl;

    return 0;
}