#pragma once

#include "hittable.h"

#include "glm/glm.hpp"

class Cube : public Hittable
{
public:
    PREFIX_DEVICE Cube(const glm::vec3& position, const glm::vec3& size, uint32_t materialIndx)
        : m_Position(position), m_Size(size), m_MaterialIndx(materialIndx) {}
    
    PREFIX_DEVICE virtual bool intersect(const Ray& ray, RayHit& hit) const
    {
        glm::vec3 invDir = 1.0f / ray.direction;
        glm::vec3 t0 = (m_Position - ray.origin) * invDir;
        glm::vec3 t1 = (m_Position + m_Size - ray.origin) * invDir;

        glm::vec3 tmin = glm::min(t0, t1);
        glm::vec3 tmax = glm::max(t0, t1);

        float tMin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
        float tMax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);
        
        if (tMax < 0 || tMin > tMax)
            return false;

        glm::vec3 hitPoint = ray.origin + ray.direction * tMin;
        glm::vec3 normal;

        if (glm::abs(hitPoint.x - m_Position.x) < 1e-4)
            normal = glm::vec3(-1, 0, 0);
        else if (glm::abs(hitPoint.x - (m_Position.x + m_Size.x)) < 1e-4)
            normal = glm::vec3(1, 0, 0);
        else if (glm::abs(hitPoint.y - m_Position.y) < 1e-4)
            normal = glm::vec3(0, -1, 0);
        else if (glm::abs(hitPoint.y - (m_Position.y + m_Size.y)) < 1e-4)
            normal = glm::vec3(0, 1, 0);
        else if (glm::abs(hitPoint.z - m_Position.z) < 1e-4)
            normal = glm::vec3(0, 0, -1);
        else
            normal = glm::vec3(0, 0, 1);
        hit.distance     = tMin;
        hit.position     = hitPoint;
        hit.normal       = normal;
        hit.materialIndx = m_MaterialIndx;

        return true;
    }

private:
    glm::vec3 m_Position;
    glm::vec3 m_Size;
    uint32_t  m_MaterialIndx;
};