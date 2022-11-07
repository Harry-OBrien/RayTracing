#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include "Hittable.hpp"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class HittableList : public Hittable {
    std::vector<shared_ptr<Hittable>> objects;

public:
    HittableList() {}
    HittableList(shared_ptr<Hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<Hittable> object) { objects.push_back(object); }

    virtual bool hit(Ray const& r, double t_min, double t_max, hit_record &rec) const override;

};

#endif /* HITTABLE_LIST_HPP */