#pragma once

#include "../hittables/hittable.h"

#include "glm/glm.hpp"

#ifndef PREFIX
    #define PREFIX
#endif

class Light
{
public:
    PREFIX virtual bool IsInLight(Hittable** world, const glm::vec3& position) const = 0;
    
    PREFIX virtual void GetLightIntensity(Hittable** world, const glm::vec3& position, const glm::vec3& normal, float& intensity) const = 0;
};
