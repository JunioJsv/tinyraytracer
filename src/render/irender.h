#ifndef IRENDER_H
#define IRENDER_H

#include "../cuda/cuda_raytracer.h"

class IRender
{
public:
    virtual ~IRender() = default;

    static void Initialize()
    {
        CudaRaytracer::Setup();
    }

    virtual void Draw() = 0;
};

#endif
