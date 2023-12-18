#pragma once

#include "light.h"

class DirectionalLight : public Light
{
public:
    __device__ DirectionalLight(const glm::vec3& direction)
        : m_Direction(glm::normalize(-direction)) {}
    
    __device__ virtual void IsInLight(const glm::vec3& position, const glm::vec3& normal, float& intensity) const
    {
        intensity = max(glm::dot(normal, m_Direction), 0.0f);
    }

public:
    glm::vec3 m_Direction{ 0.0f, 0.0f, 0.0f };
};
