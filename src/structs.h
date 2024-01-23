#pragma once

#include "core.h"

#include "glm/glm.hpp"

// Structure representing a basic pixel with RGB color channels
struct pixel
{
    unsigned char x = 0, y = 0, z = 0;

    PREFIX inline void Set(const glm::vec3& c)
    {
        x = (unsigned char)(sqrt(c.x) * 255.0f);
        y = (unsigned char)(sqrt(c.y) * 255.0f);
        z = (unsigned char)(sqrt(c.z) * 255.0f);
    }
    
    PREFIX inline void Add(const glm::vec3& c)
    {
        x = glm::min(x + (unsigned char)(sqrt(c.x) * 255.0f), 255);
        y = glm::min(y + (unsigned char)(sqrt(c.y) * 255.0f), 255);
        z = glm::min(z + (unsigned char)(sqrt(c.z) * 255.0f), 255);
    }
};

// Structure representing a pixel with the emission color and strength
struct emissionPixel
{
    glm::vec3 emission{ 0.0f, 0.0f, 0.0f };
    float strenght = 0.0f;
    
    PREFIX void Set(const emissionPixel& o)
    {
        emission = o.emission;
        strenght = o.strenght;
    }

    PREFIX void Set(const glm::vec3& e, float s)
    {
        emission = e;
        strenght = s;
    }
};

// Structure representing a base ray with origin and direction
struct Ray
{
    glm::vec3 origin   { 0.0f, 0.0f,  0.0f };
    glm::vec3 direction{ 0.0f, 0.0f, -1.0f };
};

// Structure representing the result of a ray hit
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
    float distance   = -1.0f;

    int materialIndx = -1.0f, objectIndx = -1.0f;
};

// Structure representing information from a ray tracing hit
struct TraceInfo
{
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 normal  { 0.0f, 0.0f, 0.0f };
    glm::vec3 color   { 0.0f, 0.0f, 0.0f };
    float roughness = 0.0f;
};

// Structure representing color and emission information after a hit
struct HitColorGlow
{
    glm::vec3 color   { 0.0f, 0.0f, 0.0f };
    glm::vec3 emission{ 0.0f, 0.0f, 0.0f };
    float emissionStrenght = 0.0f;
};
