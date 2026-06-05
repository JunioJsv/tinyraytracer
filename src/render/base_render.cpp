#include "base_render.h"

void BaseRender::SetBackground(
    const std::string &fileName
)
{
    Image background = LoadImage(fileName.c_str());
    if (background.data == nullptr) return;

    ImageFormat(&background, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

    CudaRaytracer::SetBackground({
        static_cast<CudaRaytracer::Background::data_t *>(background.data),
        background.width, background.height
    });
    UnloadImage(background);
}

void BaseRender::DrawGUI()
{
    DrawFPS(10, 10);
}
