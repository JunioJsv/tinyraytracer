#ifndef BASE_RENDER_H
#define BASE_RENDER_H

#include <raylib.h>
#include <string>

#include "../cuda/cuda_raytracer.h"

class BaseRender
{
public:
    BaseRender()
    {
        CudaRaytracer::Initialize();
    }

    virtual ~BaseRender()
    {
        CudaRaytracer::Destroy();
    }

    static void SetBackground(
        const std::string &fileName
    );

    virtual void Draw() = 0;

    virtual void DrawGUI();

    virtual void Resize(
        int width,
        int height
    ) = 0;

    static Image SetupImage(
        void *data,
        const int width,
        const int height
    )
    {
        return {data, width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    }
};

#endif
