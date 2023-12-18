#pragma once

#include "glm/glm.hpp"

class Light
{
public:
    __device__ virtual void IsInLight(const glm::vec3& position, const glm::vec3& normal, float& intensity) const;
};
