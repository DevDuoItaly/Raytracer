#pragma once

#include "../hittables/hittable.h"

#include "glm/glm.hpp"

class Light
{
public:
    __device__ virtual bool IsInLight(Hittable** world, const glm::vec3& position) const;

    __device__ virtual void GetLightIntensity(Hittable** world, const glm::vec3& position, const glm::vec3& normal, float& intensity) const;
};
