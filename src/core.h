#pragma once

#include "glm/glm.hpp"

#ifdef GPU_RUNNER
    #define PREFIX __device__

    // #include <curand_kernel.h>

    // #define RANDOM_UNIT(randState) curand_uniform(randState)
#else
    #define PREFIX

    // #include "cudarand.h"

    // #define RANDOM_UNIT(randState) randState->randomUnit()
#endif

//#define RANDOM_UNIT_VECTOR(randState) glm::normalize(glm::vec3{ \
    RANDOM_UNIT(randState), RANDOM_UNIT(randState), RANDOM_UNIT(randState) })

/*
PREFIX glm::vec3 RANDOM_UNIT_EMISPHERE(curandState* randState, const glm::vec3& normal)
{
    glm::vec3 rnd = RANDOM_UNIT_VECTOR(randState);
    if(glm::dot(rnd, normal) > 0)
        return rnd;
    else
        return -rnd;
}
*/

PREFIX bool refract(const glm::vec3& v, const glm::vec3& n, float ir, glm::vec3& refracted)
{
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float descriminant = 1.0f - ir * ir * (1.0f - dt * dt);
    if(descriminant <= 0)
        return false;
    
    refracted = glm::normalize(ir * (uv - n * dt) - n * std::sqrt(descriminant));
    return true;
}
