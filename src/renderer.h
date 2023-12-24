#pragma once

#include "core.h"

PREFIX bool debug = false;

#include "camera.h"
#include "structs.h"

#include "material.h"

#include "lights/light.h"

#include "hittables/hittable.h"

#include "glm/glm.hpp"

#include <cstdio>

#define MAX_DEPTH 50

PREFIX inline void UVToDirection(float u, float v, const glm::mat4& invProj, const glm::mat4& invView, glm::vec3& direction)
{
    glm::vec4 target = invProj * glm::vec4(u, v, 1.0f, 1.0f); // Clip Space
    direction = glm::vec3(invView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f)); // World space
}

PREFIX glm::vec3 TraceRay(Ray ray, Hittable** world, Light** lights, Material* materials, float multiplier, int depth, curandState* randState, int& maxDepth)
{
    if(multiplier < 0.001f)
        return glm::vec3{ 0.0f, 0.0f, 0.0f };

    RayHit hit;
    if(!(*world)->intersect(ray, hit))
    {
        maxDepth = depth;
        
        float a = (ray.direction.y + 1.0f) * 0.5f;
        return ((1.0f - a) * glm::vec3{ 1.0f, 1.0f, 1.0f } + a * glm::vec3{ 0.2f, 0.3f, 0.8f }) * multiplier;
    }

    if(debug)
        printf("Hitted: %d - %f, %f, %f\n", hit.materialIndx, hit.position.x, hit.position.y, hit.position.z);

    const glm::vec3 offPosition = hit.position + hit.normal * 0.005f;

    float intensity = 0.0f;
    (*lights)->GetLightIntensity(world, offPosition, hit.normal, intensity);

    if(debug)
        printf("Intensity: %f\n", intensity);

    const Material& material = materials[hit.materialIndx];
    glm::vec3 color = material.color * intensity * multiplier;

    if(depth < MAX_DEPTH)
    {
        ray.origin = offPosition;

        glm::vec3 rayDir = glm::normalize(ray.direction);

        if(material.reflection > 0)
        {
            // Reflection
            ray.direction = glm::reflect(ray.direction, hit.normal);
            ray.direction = glm::normalize(ray.direction + material.roughness * RANDOM_UNIT_EMISPHERE(randState, hit.normal));

            if(glm::dot(ray.direction, hit.normal) > 0)
                color += TraceRay(ray, world, lights, materials, multiplier * material.reflection, depth + 1, randState, maxDepth);
        }

        if(material.refraction > 0)
        {
            // Refraction
            glm::vec3 outNorm;
            float ir, cosine;

            if(glm::dot(rayDir, hit.normal) > 0)
            {
                outNorm = -hit.normal;
                ir = material.refraction;
                cosine = ir * glm::dot(rayDir, hit.normal) / glm::length(rayDir);
            }
            else
            {
                outNorm = hit.normal;
                ir = 1.0f / material.refraction;
                cosine = -glm::dot(rayDir, hit.normal) / glm::length(rayDir);
            }

            if(!refract(rayDir,  outNorm, ir, ray.direction))
            {
                ray.direction = glm::reflect(rayDir, hit.normal);
            }

            ray.direction = glm::normalize(ray.direction);
            ray.origin = hit.position + ray.direction * 0.01f;

            color += TraceRay(ray, world, lights, materials, multiplier * 0.9f, depth + 1, randState, maxDepth);
        }

        /*
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
        */
    }
    else
        maxDepth = depth;

    return color;
}

PREFIX glm::vec3 AntiAliasing(float u, float v, float pixelOffX, float pixelOffY, const Camera& camera, Hittable** world, Light** lights, Material* materials, curandState* randState)
{
    v = -v;

    const glm::mat4& invProj = camera.GetInverseProjectionMatrix();
    const glm::mat4& invView = camera.GetInverseViewMatrix();

    glm::vec3 color{ 0.0f, 0.0f, 0.0f };
    Ray ray{ camera.GetPosition() , glm::vec3{ 0.0f, 0.0f, 0.0f } };

    int maxDepth = 0;

    #define AA 1
#if AA
    UVToDirection(u - pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0, randState, maxDepth) / glm::vec3(maxDepth + 1);

    maxDepth = 0;

    UVToDirection(u + pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0, randState, maxDepth) / glm::vec3(maxDepth + 1);

    maxDepth = 0;

    UVToDirection(u - pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0, randState, maxDepth) / glm::vec3(maxDepth + 1);

    maxDepth = 0;

    UVToDirection(u + pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0, randState, maxDepth) / glm::vec3(maxDepth + 1);

    color *= glm::vec3{ 0.25f, 0.25f, 0.25f };
#else
    UVToDirection(u, v, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0, randState, maxDepth) / glm::vec3(maxDepth + 1);
#endif
    
    /*
    float error = 0.0018f;
    if(std::abs(u - +0.572f) < error && std::abs(v - +0.15f) < error)
    {
        // debug = true;

        UVToDirection(u, v, invProj, invView, ray.direction);
        TraceRay(ray, world, lights, materials, 1, 0, randState, maxDepth);

        debug = false;
    }
    */

    return color;
}