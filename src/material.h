#pragma once

#include "glm/glm.hpp"

struct Material
{
    glm::vec3 color;
    float roughness, reflection, refraction;
    glm::vec3 emissionColor;
    float glowStrength;
};
