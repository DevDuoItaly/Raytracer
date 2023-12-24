#pragma once

#include "light.h"

class LightsList : public Light
{
public:
    PREFIX LightsList(Light** lights, uint32_t length)
        : m_Lights(lights), m_Length(length) {}

    PREFIX virtual bool IsInLight(Hittable** world, const glm::vec3& position) const
    {
        return false;
    }
    
    PREFIX virtual void GetLightIntensity(Hittable** world, const glm::vec3& position, const glm::vec3& normal, float& intensity) const
    {
        intensity = 0.085f;
        for(int i = 0; i < m_Length; ++i)
        {
            Light* light = m_Lights[i];
            if(!light->IsInLight(world, position))
                continue;

            float intens = 0.0f;
            light->GetLightIntensity(world, position, normal, intens);

            if(intens > intensity)
                intensity = intens;
        }
    }
    
public:
    Light** m_Lights = nullptr;
    int m_Length = -1;
};
