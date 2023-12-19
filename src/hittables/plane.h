#pragma once

#include "hittable.h"
#include <glm/glm.hpp>

class Plane : public Hittable
{
public:
    __device__ Plane(const glm::vec3& normal, float distance, uint8_t materialIndx)
        : m_Normal(glm::normalize(normal)), m_Distance(distance), m_MaterialIndx(materialIndx) {}
    
    __device__ virtual bool intersect(const Ray& ray, RayHit& hit) const
    {
        float denom = glm::dot(m_Normal, ray.direction);
        if (abs(denom) > 1e-6) {
            glm::vec3 p0 = m_Normal * m_Distance;
            float t = glm::dot(p0 - ray.origin, m_Normal) / denom;
            if (t >= 0) {
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = m_Normal;
                hit.materialIndx = m_MaterialIndx;
                return true;
            }
        }
        return false;
    }

private:
    glm::vec3 m_Normal;
    float m_Distance;
    int m_MaterialIndx;
};