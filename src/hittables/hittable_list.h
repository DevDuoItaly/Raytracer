#pragma once

#include "hittable.h"

class HittableList : public Hittable
{
public:
    __device__ HittableList() {}
    __device__ HittableList(Hittable** hittables, uint32_t len)
        : m_Hittables(hittables), m_Len(len) {}
    
    __device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) const
    {
        IntersectInfo currInfo;
        bool hasHit = false;

        info.hit.distance = 0xffffffff;

        for(int i = 0; i < m_Len; ++i)
        {
            if (!m_Hittables[i]->intersect(ray, currInfo) || currInfo.hit.distance >= info.hit.distance)
                continue;

            hasHit = true;
            info = IntersectInfo{ currInfo };
            // info.hit.color = { 1.0f, 0.0f, 1.0f };
        }

        return hasHit;
    }
    
public:
    Hittable** m_Hittables;
    uint32_t m_Len;
};
