#pragma once

#include "hittable.h"
#include "glm/glm.hpp"

class Cube : public Hittable
{
public:
    __device__ Cube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color)
        : m_Position(position), m_Size(size), m_Color(color) {}

    __device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) const
    {
        glm::vec3 invDir = 1.0f / ray.Direction;
        glm::vec3 t0 = (m_Position - ray.Origin) * invDir;
        glm::vec3 t1 = (m_Position + m_Size - ray.Origin) * invDir;

        glm::vec3 tmin = glm::min(t0, t1);
        glm::vec3 tmax = glm::max(t0, t1);

        float tMin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
        float tMax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

        if (tMax < 0 || tMin > tMax)
            return false;

        glm::vec3 hitPoint = ray.Origin + ray.Direction * tMin;
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

        info.hit.distance = tMin;
        info.hit.position = hitPoint;
        info.hit.normal = normal;
        info.hit.color = m_Color;

        return true;
    }

private:
    glm::vec3 m_Position;
    glm::vec3 m_Size;
    glm::vec3 m_Color;
};