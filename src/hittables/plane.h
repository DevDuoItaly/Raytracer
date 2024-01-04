#pragma once

#include "hittable.h"

#include "glm/glm.hpp"

class Plane : public Hittable
{
public:
    PREFIX_DEVICE Plane(const glm::vec3& center, const glm::vec3& normal, uint8_t materialIndx)
        : m_Center(center), m_Normal(glm::normalize(normal)), m_MaterialIndx(materialIndx) {}
    
    PREFIX_DEVICE virtual bool intersect(const Ray& ray, RayHit& hit) const
    {
        float denom = glm::dot(m_Normal, ray.direction);
        if (std::abs(denom) <= 1e-6)
            return false;

        float t = glm::dot(ray.origin - m_Center, m_Normal) / denom;
        if (t < 0)
            return false;
            
        hit.distance = t;
        hit.position = ray.origin + ray.direction * t;
        hit.normal = m_Normal;
        hit.materialIndx = m_MaterialIndx;
        return true;
    }

    PREFIX_DEVICE virtual bool hasIntersect(const Ray& ray) const
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