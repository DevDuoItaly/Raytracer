#pragma once

#include "core.h"
#include "hittables/hittable.h"

#include "glm/glm.hpp"

class Light
{
public:
    PREFIX_DEVICE virtual bool IsInLight(Hittable** world, const glm::vec3& position) const = 0;
    
    PREFIX_DEVICE virtual void GetLightIntensity(Hittable** world, const glm::vec3& position, const glm::vec3& normal, float& intensity) const = 0;
};
