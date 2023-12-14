#include "hittable.h"

class Sphere : Hittable
{
public:
    Sphere(const glm::vec3& position, uint8_t materialIndx)
        : Hittable(position, materialIndx) {}
    
    bool intersect(const Ray& ray, intersectInfo& info)
    {
        glm::vec3 origin = ray.Origin - m_Position;

        float a = glm::dot(ray.Direction, ray.Direction);
        float b = 2.0f * glm::dot(origin, ray.Direction);
        float c = glm::dot(origin, origin) - m_Radius * m_Radius;
        
        float discriminant = b * b - 4.0f * a * c;
        
        if (discriminant < 0.0f)
            return false;

        distance = (-b - sqrt(discriminant)) / (2.0f * a);
        hitPoint = origin + ray.Direction * distance;
        normal = glm::normalize(hitPoint);
        hitPoint += m_Position;
        return true;
    }

private:
    float m_Radius;
};
