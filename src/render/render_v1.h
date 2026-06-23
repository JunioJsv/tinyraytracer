#ifndef RENDER_V1_H
#define RENDER_V1_H

#include "base_render.h"

#include <raylib.h>
#include <vector>

class RenderV1 final : public BaseRender
{
public:
    RenderV1(
        int width,
        int height
    );

    ~RenderV1() override;

    void Draw() override;

    void Resize(
        int width,
        int height
    ) override;

    const CudaRaytracer::RenderInfo& UpdateRenderInfo() override;

private:
    std::vector<uint32_t> pixels;
    Image image;
    Texture2D texture;
};


#endif
