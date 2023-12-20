#pragma once

#include "../structs.h"

#include "glm/glm.hpp"

#ifndef PREFIX
    #define PREFIX
#endif

class Hittable
{
public:
    PREFIX virtual bool intersect(const Ray& ray, RayHit& hit) const = 0;

    PREFIX virtual bool hasIntersect(const Ray& ray) const = 0;
};
