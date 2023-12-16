#pragma once

#include "hittable.h"

class Sphere : public Hittable
{
public:
    __device__ Sphere(const glm::vec3& position, float radius, const glm::vec3& color) //uint8_t materialIndx)
        : m_Position(position), m_Radius(radius), m_Color(color) {} //m_MaterialIndx(materialIndx) {}
    
    __device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) const
    {
        glm::vec3 origin = ray.Origin + m_Position;

        float a = glm::dot(ray.Direction, ray.Direction);
        float b = 2.0f * glm::dot(origin, ray.Direction);
        float c = glm::dot(origin, origin) - m_Radius * m_Radius;
        
        float discriminant = b * b - 4.0f * a * c;
        
        if (discriminant < 0.0f)
            return false;

        float dist = (-b - sqrt(discriminant)) / (2.0f * a);
        if(dist < 0.0f)
            return false;

        info.hit.distance = dist;
        info.hit.position = origin + ray.Direction * dist;
        info.hit.normal = glm::normalize(info.hit.position);
        info.hit.position += m_Position;
        info.hit.color = m_Color;
        return true;
    }

private:
    glm::vec3 m_Position;
    float m_Radius;

    glm::vec3 m_Color;

    // float m_MaterialIndx;
};
