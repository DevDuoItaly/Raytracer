#pragma once

#include "hittable.h"

class HittablesList : public Hittable
{
public:
    PREFIX_DEVICE HittablesList(Hittable** hittables, uint32_t length)
        : m_Hittables(hittables), m_Length(length) {}
    
    PREFIX_DEVICE ~HittablesList()
    {
        for(int i = 0; i < m_Length; ++i)
            delete m_Hittables[i];
    }

    PREFIX_DEVICE bool intersect(const Ray& ray, RayHit& hit) const
    {
        hit.distance = FLT_MAX;

        RayHit currHit;
        bool hasHit = false;

        // For each hittable in the list try intersect and compare distance
        for(int i = 0; i < m_Length; ++i)
        {
            if(!m_Hittables[i]->intersect(ray, currHit) || currHit.distance >= hit.distance)
                continue;

            hit.copy(currHit);
            hit.objectIndx = i;
            hasHit = true;
        }

        // Return the nearest hittable
        return hasHit;
    }

    PREFIX_DEVICE virtual bool hasIntersect(const Ray& ray) const
    {
        RayHit hit;

        // Returns True to the first intersection found
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
