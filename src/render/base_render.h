#ifndef BASE_RENDER_H
#define BASE_RENDER_H

#include <string>

#include "../cuda/cuda_raytracer.h"

class BaseRender
{
public:
    BaseRender()
    {
        CudaRaytracer::Setup();
    }

    virtual ~BaseRender() = default;

    static void SetBackground(
        const std::string &fileName
    );

    virtual void Draw() = 0;

    virtual void DrawGUI();
};

#endif
