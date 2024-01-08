#pragma once

#include "core.h"

PREFIX_DEVICE bool debug = false;

#include "camera.h"
#include "structs.h"

#include "material.h"

#include "lights/light.h"

#include "hittables/hittable.h"

#include "glm/glm.hpp"

#include <cstdio>

#define MAX_DEPTH 5

PREFIX_DEVICE inline void UVToDirection(float u, float v, const glm::mat4& invProj, const glm::mat4& invView, glm::vec3& direction)
{
    glm::vec4 target = invProj * glm::vec4(u, v, 1.0f, 1.0f); // Clip Space
    direction = glm::vec3(invView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f)); // World space
}

PREFIX_DEVICE HitColorGlow TraceRay(Ray ray, Hittable** world, Light** lights, Material* materials, float multiplier, int depth, curandState* randState, int& maxDepth)
{
    if(multiplier < 0.001f)
    {
        maxDepth = depth;
        return HitColorGlow{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 0 };
    }

    if(debug)
        printf("Try hit depth: %d\n", depth);

    RayHit hit;
    if(!(*world)->intersect(ray, hit))
    {
        maxDepth = depth;

        if(debug)
            printf("Ray hitted Sky!\n");
        
        float a = (ray.direction.y + 1.0f) * 0.5f;
        glm::vec3 skyColor = ((1.0f - a) * glm::vec3{ 1.0f, 1.0f, 1.0f } + a * glm::vec3{ 0.2f, 0.3f, 0.8f }) * multiplier;
        return HitColorGlow{ skyColor, { 0.0f, 0.0f, 0.0f }, 0 };
    }

    if(debug)
        printf("Hitted: %d - %f, %f, %f\n", hit.materialIndx, hit.position.x, hit.position.y, hit.position.z);

    const glm::vec3 offPosition = hit.position + hit.normal * 0.005f;

    float intensity = 0.0f;
    (*lights)->GetLightIntensity(world, offPosition, hit.normal, intensity);

    if(debug)
        printf("Intensity: %f\n", intensity);

    const Material& material = materials[hit.materialIndx];
    glm::vec3 color    = material.color * intensity * multiplier;
    glm::vec3 emission = material.emissionColor;
    float emissionStrenght = material.glowStrength;

    if(depth <= MAX_DEPTH)
    {
        ray.origin = offPosition;

        glm::vec3 rayDir = glm::normalize(ray.direction);

        if(material.reflection > 0)
        {
            // Reflection
            ray.direction = glm::reflect(ray.direction, hit.normal);
            ray.direction = glm::normalize(ray.direction + material.roughness * RANDOM_UNIT_EMISPHERE(randState, hit.normal));

            if(glm::dot(ray.direction, hit.normal) > 0)
            {
                HitColorGlow hitInfo = TraceRay(ray, world, lights, materials, multiplier * material.reflection, depth + 1, randState, maxDepth);
                color += hitInfo.color;

                if(hitInfo.emissionStrenght > 0)
                {
                    emission = glm::normalize(emission * glm::vec3(emissionStrenght) + hitInfo.emission * glm::vec3(hitInfo.emissionStrenght));
                    emissionStrenght = glm::max(emissionStrenght, hitInfo.emissionStrenght * material.reflection * 1.25f);
                }
            }
        }

        if(material.refraction > 0)
        {
            // Refraction
            glm::vec3 outNorm;
            float ir;

            if(glm::dot(rayDir, hit.normal) > 0)
            {
                outNorm = -hit.normal;
                ir = material.refraction;
            }
            else
            {
                outNorm = hit.normal;
                ir = 1.0f / material.refraction;
            }

            if(!refract(rayDir,  outNorm, ir, ray.direction))
            {
                ray.direction = glm::reflect(rayDir, hit.normal);
            }

            ray.direction = glm::normalize(ray.direction + material.roughness * RANDOM_UNIT_EMISPHERE(randState, hit.normal));
            ray.origin = hit.position + ray.direction * 0.01f;

            HitColorGlow hitInfo = TraceRay(ray, world, lights, materials, multiplier * 0.9f, depth, randState, maxDepth);
            color += hitInfo.color;
            
            if(hitInfo.emissionStrenght > 0)
            {
                emission = glm::normalize(emission * glm::vec3(emissionStrenght) + hitInfo.emission * glm::vec3(hitInfo.emissionStrenght));
                emissionStrenght = glm::max(emissionStrenght, hitInfo.emissionStrenght * 0.95f);
            }
        }

        if(material.reflection == 0 && material.refraction == 0)
            maxDepth = depth;
    }
    else
        maxDepth = depth;

    return HitColorGlow{ color, emission, emissionStrenght };
}

PREFIX_DEVICE HitColorGlow AntiAliasing(float u, float v, float pixelOffX, float pixelOffY, Camera* camera, Hittable** world, Light** lights, Material* materials, curandState* randState)
{
    v = -v;

    const glm::mat4& invProj = camera->GetInverseProjectionMatrix();
    const glm::mat4& invView = camera->GetInverseViewMatrix();
    
    HitColorGlow result{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 0.0f };
    Ray ray{ camera->GetPosition() , glm::vec3{ 0.0f, 0.0f, 0.0f } };

    int maxDepth = 0;

    /*
    float error = 0.003f;
    if(std::abs(u - 0.0f) < error && std::abs(v - 0.0f) < error)
    {
        // debug = true;

        // printf("DEBUG!\n");

        UVToDirection(u, v, invProj, invView, ray.direction);
        TraceRay(ray, world, lights, materials, 1, 1, randState, maxDepth);

        debug = false;
    }

    maxDepth = 0;
    */

    UVToDirection(u - pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    HitColorGlow info = TraceRay(ray, world, lights, materials, 1, 1, randState, maxDepth);
    result.color += info.color / glm::vec3(maxDepth);
    result.emission += info.emission;
    result.emissionStrenght = info.emissionStrenght;

    maxDepth = 0;

    UVToDirection(u + pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    HitColorGlow info1 = TraceRay(ray, world, lights, materials, 1, 1, randState, maxDepth);
    result.color += info1.color / glm::vec3(maxDepth);
    result.emission += info1.emission;
    result.emissionStrenght = glm::max(result.emissionStrenght, info1.emissionStrenght);

    maxDepth = 0;

    UVToDirection(u - pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    HitColorGlow info2 = TraceRay(ray, world, lights, materials, 1, 1, randState, maxDepth);
    result.color += info2.color / glm::vec3(maxDepth);
    result.emission += info2.emission;
    result.emissionStrenght = glm::max(result.emissionStrenght, info2.emissionStrenght);

    maxDepth = 0;

    UVToDirection(u + pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    HitColorGlow info3 = TraceRay(ray, world, lights, materials, 1, 1, randState, maxDepth);
    result.color += info3.color / glm::vec3(maxDepth);
    result.emission += info3.emission;
    result.emissionStrenght = glm::max(result.emissionStrenght, info3.emissionStrenght);

    result.color *= glm::vec3(0.25f);
    result.emission *= glm::vec3(0.25f);

    return result;
}