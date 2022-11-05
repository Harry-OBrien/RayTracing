#include <iostream>
#include "vec3.hpp"
#include "Colour.hpp"

int main(int argc, char** argv) {

    
    //Image
    int const imageWidth = 256;
    int const imageHeight = 256;

    // Render

    std::cout << "P3\n"
        << imageWidth
        << " "
        << imageHeight
        << "\n255"
        << std::endl;

    for(int j = imageHeight-1; j >= 0; j--) {
        std::cerr << "Scanlines remaining: "
            << j
            << std::endl
            << std::flush;

        for (int i = 0; i < imageWidth; i++) {
            colour pixelColour(
                double(i) / (imageWidth - 1),
                double(j) / (imageHeight -1),
                0.25);

            write_colour(std::cout, pixelColour);
        }
    }

    std::cerr << "\nDone!" << std::endl;

    return 0;
}