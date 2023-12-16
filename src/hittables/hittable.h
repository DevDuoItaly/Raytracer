#pragma once 

#include "../structs.h"

class Hittable
{
public:
    __device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) const = 0;
};
