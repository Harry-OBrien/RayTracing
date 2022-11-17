#include "HittableList.hpp"

__device__ bool HittableList::hit(Ray const& r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    bool closest_so_far = t_max;

    for (size_t i = 0; i < n_objects; i++) {
        if(objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}