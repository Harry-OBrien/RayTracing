#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include "Hittable.hpp"

class HittableList : public Hittable {
    Hittable **objects;
    size_t n_objects;

public:
    __device__ HittableList() {}
    __device__ HittableList(Hittable **_objects, size_t n) : objects(_objects), n_objects(n) {}

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, hit_record &rec) const override;
};

#endif /* HITTABLE_LIST_HPP */