#pragma once

#include "hittable.h"

#include <glm/glm.hpp>

class Plane : public Hittable
{
public:
    PREFIX Plane(const glm::vec3& center, const glm::vec3& normal, uint8_t materialIndx)
        : m_Center(center), m_Normal(glm::normalize(normal)), m_MaterialIndx(materialIndx) {}
    
    PREFIX virtual bool intersect(const Ray& ray, RayHit& hit) const
    {
        float denom = glm::dot(m_Normal, ray.direction);
        if(debug)
            printf("Denom: %f %d\n", denom, (std::abs(denom) <= 1e-6) ? 1 : 0);

        if (std::abs(denom) <= 1e-6)
            return false;

        const glm::vec3 a = m_Center - ray.origin;
        if(debug)
            printf("%f %f %f - %f %f %f\n", a.x, a.y, a.z, m_Normal.x, m_Normal.y, m_Normal.z);
        
        float t = glm::dot(ray.origin - m_Center, m_Normal) / denom;
        if(debug)
            printf("T: %f\n", t);

        if (t < 0)
            return false;
            
        hit.distance = t;
        hit.position = ray.origin + ray.direction * t;
        hit.normal = m_Normal;
        hit.materialIndx = m_MaterialIndx;
        return true;
    }

    PREFIX virtual bool hasIntersect(const Ray& ray) const
    {
        float denom = glm::dot(m_Normal, ray.direction);
        if (std::abs(denom) <= 1e-6)
            return false;
        
        return glm::dot(m_Center - ray.origin, m_Normal) / denom;
    }

private:
    glm::vec3 m_Center, m_Normal;
    int m_MaterialIndx;
};