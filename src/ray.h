#pragma once

#include "glm/glm.hpp"

class Ray
{
public:

    __host__ __device__ Ray() : Origin({}), Direction({ 0.0f, 0.0f, -1.0f }) {}
    __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction) : Origin(origin), Direction(direction) {}

    __host__ __device__  glm::vec3 pointAtParameter(float t) const
    {
        return Origin + Direction * t;
    }

    __host__ __device__ const glm::vec3& getOrigin() const
    {
        return Origin;
    }

    __host__ __device__ const glm::vec3& getDirection() const
    {
        return Direction;
    }

public:
    glm::vec3 Origin;
    glm::vec3 Direction;
};
