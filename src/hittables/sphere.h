#pragma once

#include "hittable.h"

#include <cstdio>

class Sphere : public Hittable
{
public:
    PREFIX_DEVICE Sphere(const glm::vec3& position, const float radius, uint8_t materialIndx)
        : m_Position(position), m_Radius(radius), m_MaterialIndx(materialIndx) {}
    
    PREFIX_DEVICE virtual bool intersect(const Ray& ray, RayHit& hit) const
    {
        glm::vec3 origin = ray.origin - m_Position;

        float a = glm::dot(ray.direction, ray.direction);
        float b = 2.0f * glm::dot(origin, ray.direction);
        float c = glm::dot(origin, origin) - m_Radius * m_Radius;
        
        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f)
            return false;

        float sqrtd = sqrtf(discriminant);
        float t1 = (-b - sqrtd) / 2.0f;
        float t2 = (-b + sqrtd) / 2.0f;

        if(debug)
            printf("%f %f\n", t1, t2);

        if(t1 < 0 && t2 < 0)
            return false;
        
        if(t2 > t1)
        {
            if(t1 >= 0) hit.distance = t1;
            else hit.distance = t2;
        }
        else
        {
            if(t2 >= 0) hit.distance = t2;
            else hit.distance = t1;
        }
        
        hit.position = origin + ray.direction * hit.distance;
        hit.normal = glm::normalize(hit.position);
        hit.position += m_Position;
        hit.materialIndx = m_MaterialIndx;
        return true;
    }

    PREFIX_DEVICE virtual bool hasIntersect(const Ray& ray) const
    {
        glm::vec3 origin = ray.origin - m_Position;

        float a = glm::dot(ray.direction, ray.direction);
        float b = 2.0f * glm::dot(origin, ray.direction);
        float c = glm::dot(origin, origin) - m_Radius * m_Radius;
        
        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f)
            return false;

        float sqrtd = sqrtf(discriminant);
        float t1 = (-b - sqrtd) / 2.0f;
        float t2 = (-b + sqrtd) / 2.0f;

        if(t1 < 0 && t2 < 0)
            return false;
        
        return true;
    }

private:
    glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
    float m_Radius     = -1.0f;

    int m_MaterialIndx = -1;
};
