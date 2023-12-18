#pragma once

#include "../structs.h"

#include "glm/glm.hpp"

class Hittable
{
public:
    __device__ virtual bool intersect(const Ray& ray, RayHit& hit) const;
};
