#pragma once

#include "hittable.h"

class HittablesList : public Hittable
{
public:
    __device__ HittablesList(Hittable** hittables, uint32_t length)
        : m_Hittables(hittables), m_Length(length) {}
    
    __device__ virtual bool intersect(const Ray& ray, RayHit& hit) const
    {
        hit.distance = FLT_MAX;

        RayHit currHit;
        bool hasHit = false;

        for(int i = 0; i < m_Length; ++i)
        {
            if(!m_Hittables[i]->intersect(ray, currHit) || currHit.distance >= hit.distance)
                continue;

            hit.copy(currHit);
            hasHit = true;
        }

        return hasHit;
    }

    __device__ virtual bool hasIntersect(const Ray& ray) const
    {
        RayHit hit;
        for(int i = 0; i < m_Length; ++i)
        {
            if(m_Hittables[i]->intersect(ray, hit))
                return true;
        }
        return false;
    }

public:
    Hittable** m_Hittables = nullptr;
    int m_Length = -1;
};
