#pragma once

#include "core.h"

#include "glm/glm.hpp"

struct pixel
{
    unsigned char x = 0, y = 0, z = 0;

    PREFIX void Set(const glm::vec3& c)
    {
        x = (unsigned char)(sqrt(c.x) * 255.0f);
        y = (unsigned char)(sqrt(c.y) * 255.0f);
        z = (unsigned char)(sqrt(c.z) * 255.0f);
    }
};

struct Ray
{
    glm::vec3 origin   { 0.0f, 0.0f,  0.0f };
    glm::vec3 direction{ 0.0f, 0.0f, -1.0f };
};

struct RayHit
{
public:
    PREFIX_DEVICE RayHit() {}

    PREFIX_DEVICE RayHit(float dist)
        : distance(dist) {}

    PREFIX_DEVICE void copy(const RayHit& o)
    {
        position     = o.position;
        normal       = o.normal;
        distance     = o.distance;
        materialIndx = o.materialIndx;
        objectIndx   = o.objectIndx;
    }

public:
	glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 normal  { 0.0f, 0.0f, 0.0f };
    float distance   = -1;

    int materialIndx = -1, objectIndx = -1;
};

struct TraceInfo
{
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 normal  { 0.0f, 0.0f, 0.0f };
    glm::vec3 color   { 0.0f, 0.0f, 0.0f };
    float roughness = 0;
};
