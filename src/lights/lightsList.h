#pragma once

#include "light.h"

class LightsList : public Light
{
public:
    PREFIX_DEVICE LightsList(Light** lights, uint32_t length)
        : m_Lights(lights), m_Length(length) {}
    
    PREFIX_DEVICE ~LightsList()
    {
        for(int i = 0; i < m_Length; ++i)
            delete m_Lights[i];
    }
    
    PREFIX_DEVICE virtual bool IsInLight(Hittable** world, const glm::vec3& position) const
    {
        return false;
    }
    
    PREFIX_DEVICE virtual void GetLightIntensity(Hittable** world, const glm::vec3& position, const glm::vec3& normal, float& intensity) const
    {
        intensity = 0.085f;

        // For each light in the list check if the light illuminates the spot
        for(int i = 0; i < m_Length; ++i)
        {
            Light* light = m_Lights[i];
            if(!light->IsInLight(world, position))
                continue;
            
            // Calculate the light amount that hit the spot
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
