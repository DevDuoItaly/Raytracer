#pragma once

#include "light.h"

class DirectionalLight : public Light
{
public:
    PREFIX DirectionalLight(const glm::vec3& direction)
        : m_Direction(glm::normalize(-direction)) {}
    
    PREFIX virtual bool IsInLight(Hittable** world, const glm::vec3& position) const
    {
        if(debug)
            printf("Check in light: %f %f %f - %f %f %f\n", position.x, position.y, position.z, m_Direction.x, m_Direction.y, m_Direction.z);

        return !(*world)->hasIntersect(Ray{ position, m_Direction });
    }
    
    PREFIX virtual void GetLightIntensity(Hittable** world, const glm::vec3& position, const glm::vec3& normal, float& intensity) const
    {
        float d = glm::dot(normal, m_Direction);
        intensity = d > 0 ? d : 0;
    }

public:
    glm::vec3 m_Direction{ 0.0f, 0.0f, 0.0f };
};
