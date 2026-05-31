#ifndef RENDER_V1_H
#define RENDER_V1_H

#include "irender.h"

#include <raylib.h>
#include <vector>

#include "../camera_controller.h"

class RenderV1 final : public IRender
{
public:
    RenderV1(
        int width,
        int height,
        const CameraController &camera
    );

    ~RenderV1() override;

    void Draw() override;

private:
    std::vector<uint32_t> pixels;
    const CameraController &camera;
    const Image image;
    const Texture2D texture;
};


#endif
