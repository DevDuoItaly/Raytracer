#pragma once

#include "light.h"

class LightsList : public Light
{
public:
    __device__ LightsList(Light** lights, uint32_t length)
        : m_Lights(lights), m_Length(length) {}
    
    __device__ virtual void IsInLight(const glm::vec3& position, const glm::vec3& normal, float& intensity) const
    {
        intensity = 0;
        for(int i = 0; i < m_Length; ++i)
        {
            float intens = 0.0f;
            m_Lights[i]->IsInLight(position, normal, intens);

            if(intens > intensity)
                intensity = intens;
        }
    }
    
public:
    Light** m_Lights = nullptr;
    int m_Length = -1;
};
