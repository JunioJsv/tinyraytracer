#ifndef RENDER_V2_H
#define RENDER_V2_H

#include "base_render.h"

#include <raylib.h>

class RenderV2 final : public BaseRender
{
public:
    RenderV2(
        int width,
        int height
    );

    ~RenderV2() override;

    void Draw() override;

    void Resize(
        int width,
        int height
    ) override;

    const CudaRaytracer::RenderInfo& UpdateRenderInfo() override;

private:
    Texture2D texture;
    cudaGraphicsResource *cudaTexture;
};


#endif
