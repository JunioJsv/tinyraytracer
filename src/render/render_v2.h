#ifndef RENDER_V2_H
#define RENDER_V2_H

#include "base_render.h"

#include <raylib.h>

#include "../camera_controller.h"

class RenderV2 final : public BaseRender
{
public:
    RenderV2(
        int width,
        int height,
        const CameraController &camera
    );

    ~RenderV2() override;

    void Draw() override;

private:
    const CameraController &camera;
    const Texture2D texture;
    cudaGraphicsResource *cudaTexture;
};


#endif
