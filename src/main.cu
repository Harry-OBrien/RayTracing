#include <iostream>
#include <float.h>

#include "Colour.hpp"
#include "Vec3.hpp"
#include "Ray.hpp"
#include "HittableList.hpp"
#include "Sphere.hpp"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(point3 const& center, double radius, Ray const& r)
{
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;
    return (discriminant > 0);
}

__device__ vec3 rayColour(const Ray& r, Hittable** world)
{
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * (rec.normal + 1.0f);
    }
 
   vec3 unit_direction = unit_vector(r.direction());
   float t = 0.5f*(unit_direction.y() + 1.0f);
   return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(
    vec3 *fb, int max_x, int max_y, 
    vec3 lower_left_corner, 
    vec3 horizontal, 
    vec3 vertical, 
    vec3 origin, 
    Hittable** world)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y))
        return;

    int pixel_index = j*max_x + i;
    float u = float(i) / float(max_x - 1);
    float v = float(j) / float(max_y - 1);
    Ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    fb[pixel_index] = rayColour(r, world);
}

__global__ void create_world(Hittable **d_list, Hittable **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *(d_list) = new Sphere(point3(0, -100.5, -1), 100);
        *(d_list + 1) = new Sphere(point3(0, 0, -1), 0.5);
        *d_world = new HittableList(d_list, 2);
    }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
}

int main(int argc, char** argv)
{
    //Image
    double aspect_ratio = 16.0 / 9.0;
    int const imageWidth = 1024;
    int const imageHeight = static_cast<int>(imageWidth / aspect_ratio);
    int numPixels = imageHeight * imageWidth;

    // Camera
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0, 0);
    vec3 vertical = vec3(0, viewport_height, 0);
    vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    // World
    Hittable **d_list; 
    checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(Hittable*))); // creating 2 objects
    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(Hittable*)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate frame buffer (fb)
    size_t fb_size = numPixels*sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    //render buf
    int tx = 8;
    int ty = 8;
    dim3 blocks(imageWidth/tx + 1, imageHeight/ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(
        fb, imageWidth, imageHeight,
        lower_left_corner,
        horizontal,
        vertical,
        origin,
        d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    std::cout << "P3\n"
        << imageWidth
        << " "
        << imageHeight
        << "\n255"
        << std::endl;

    for (int j = imageHeight-1; j >= 0; j--)
    {
        std::cerr << "Scanlines remaining: "
            << j
            << " "
            << std::endl
            << std::flush;

        for (int i = 0; i < imageWidth; i++)
        {
            size_t pixel_idx = j*imageWidth + i;
            write_colour(std::cout, fb[pixel_idx], 1);
        }
    }

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    std::cerr << "\nDone!" << std::endl;

    // Useful for cuda-memcheck --leach-check full
    cudaDeviceReset();

    return 0;
}