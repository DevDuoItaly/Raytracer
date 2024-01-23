#pragma once

#include "glm/glm.hpp"

#ifdef GPU_RUNNER
    #define PREFIX_DEVICE __device__  // Macro for GPU device code.
    #define PREFIX_HOST __host__      // Macro for GPU host code.
    #define PREFIX __host__ __device__// Macro for code usable in both GPU host and device.

    #include <curand_kernel.h>        // Include CUDA random number generation library.

    #define RANDOM_UNIT(randState) curand_uniform(randState) // Macro for generating a random unit using CUDA.
#else
    #define PREFIX_DEVICE  // Empty macro for non-GPU device code.
    #define PREFIX_HOST    // Empty macro for non-GPU host code.
    #define PREFIX         // Empty macro for non-GPU general code.

    #include "rand.h"      // Include custom random number generation header.

    #define RANDOM_UNIT(randState) randState->randomUnit() // Macro for generating a random unit using custom RNG.
#endif

// Macro for generating a random unit vector.
#define RANDOM_UNIT_VECTOR(randState) glm::normalize(glm::vec3{ \
    RANDOM_UNIT(randState), RANDOM_UNIT(randState), RANDOM_UNIT(randState) })

// Function to generate a random unit vector in a hemisphere oriented around a normal vector.
PREFIX_DEVICE glm::vec3 RANDOM_UNIT_EMISPHERE(curandState* randState, const glm::vec3& normal)
{
    glm::vec3 rnd = RANDOM_UNIT_VECTOR(randState);
    if(glm::dot(rnd, normal) > 0)
        return rnd;
    else
        return -rnd;
}

// Function to compute the refraction of a vector 'v' with a normal 'n' and index of refraction 'ir'.
PREFIX_DEVICE bool refract(const glm::vec3& v, const glm::vec3& n, float ir, glm::vec3& refracted)
{
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float descriminant = 1.0f - ir * ir * (1.0f - dt * dt);
    if(descriminant <= 0)
        return false;
    
    refracted = glm::normalize(ir * (uv - n * dt) - n * std::sqrt(descriminant));
    return true;
}
