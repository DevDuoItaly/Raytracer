#include "glm/glm.hpp"

class Hittable
{
public:
    Hittable(const glm::vec3& position, uint8_t materialIndx)
        : m_Position(position), m_MaterialIndx(materialIndx) {}
    
    virtual bool intersect(const Ray& ray, intersectInfo& info);

protected:
    glm::vec3 m_Position;
    uint8_t m_MaterialIndx;
};
