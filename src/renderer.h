#pragma once

#ifndef PREFIX
    #define PREFIX
#endif

PREFIX bool debug = false;

#include "camera.h"
#include "structs.h"

#include "material.h"

#include "lights/light.h"

#include "hittables/hittable.h"

#include "glm/glm.hpp"

#include <cstdio>

#define MAX_DEPTH 1

PREFIX inline void UVToDirection(float u, float v, const glm::mat4& invProj, const glm::mat4& invView, glm::vec3& direction)
{
    glm::vec4 target = invProj * glm::vec4(u, v, 1.0f, 1.0f); // Clip Space
    direction = glm::vec3(invView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f)); // World space
}

PREFIX glm::vec3 TraceRay(Ray ray, Hittable** world, Light** lights, Material* materials, float multiplier, int depth)
{
    RayHit hit;
    if(!(*world)->intersect(ray, hit))
        return depth != 0 ? glm::vec3{ 0.0f, 0.0f, 0.0f } : glm::vec3{ 0.52f, 0.80f, 0.92f } * multiplier;

    if(debug)
        printf("Hitted: %d\n", hit.materialIndx);

    const glm::vec3 offPosition = hit.position + hit.normal * 0.0001f;

    float intensity = 0.0f;
    (*lights)->GetLightIntensity(world, offPosition, hit.normal, intensity);

    if(debug)
        printf("Intensity: %f\n", intensity);

    const Material& material = materials[hit.materialIndx];
    glm::vec3 color = material.color * intensity * multiplier;

    if(depth < MAX_DEPTH)
    {
        if(material.refraction <= 0)
        {
            // Reflection
            ray.origin = offPosition;
            ray.direction = glm::reflect(ray.direction, hit.normal + material.roughness);
        }
        else
        {
            // Refraction
            // ray.direction = glm::refraction(ray.direction, hit.normal, material.refraction);
        }

        color += TraceRay(ray, world, lights, materials, multiplier * 0.25f, depth + 1);
    }

    return color;
}

PREFIX glm::vec3 AntiAliasing(float u, float v, float pixelOffX, float pixelOffY, const Camera& camera, Hittable** world, Light** lights, Material* materials)
{
    const glm::mat4& invProj = camera.GetInverseProjectionMatrix();
    const glm::mat4& invView = camera.GetInverseViewMatrix();

    glm::vec3 color{ 0.0f, 0.0f, 0.0f };
    Ray ray{ camera.GetPosition() , glm::vec3{ 0.0f, 0.0f, 0.0f } };

    UVToDirection(u - pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    UVToDirection(u + pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    UVToDirection(u - pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    UVToDirection(u + pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    color *= glm::vec3{ 0.25f, 0.25f, 0.25f };
    
    /*
    float error = 0.001f;
    if(std::abs(u - 0.5f) < error && std::abs(v - 0.5f) < error)
    {
        debug = true;

        UVToDirection(u, v, invProj, invView, ray.direction);
        color = TraceRay(ray, world, lights, materials, 1, 0);

        debug = false;
    }
    */

    return color;
}