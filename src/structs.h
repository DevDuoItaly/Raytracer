#pragma once

#include "glm/glm.hpp"

struct pixel
{
    unsigned char x, y, z;

    __host__ __device__ void Set(const glm::vec3& c)
    {
        x = (unsigned char)(c.x * 255.0f);
        y = (unsigned char)(c.y * 255.0f);
        z = (unsigned char)(c.z * 255.0f);
    }
};

struct RayHit
{
	glm::vec3 position = { 0.0f, 0.0f, 0.0f };
    glm::vec3 normal   = { 0.0f, 0.0f, 0.0f };
    glm::vec3 color    = { 0.0f, 0.0f, 0.0f };
    float distance = -1;
};

struct IntersectInfo
{
public:
    __host__ __device__ IntersectInfo() {}

    __host__ __device__ IntersectInfo(const IntersectInfo& o)
    {
        hit.position = o.hit.position;
        hit.normal   = o.hit.normal;
        hit.color    = o.hit.color;
        hit.distance = o.hit.distance;
        objectIndx   = o.objectIndx;
    }

public:
    RayHit hit;
    float Roughness = 0.25f;
    uint32_t objectIndx = 0;
};
