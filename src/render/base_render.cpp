#include "base_render.h"

void BaseRender::SetBackground(
    const std::string &fileName
)
{
    Image background = LoadImage(fileName.c_str());
    if (background.data == nullptr) return;

    const bool isHDR = fileName.ends_with(".hdr");

    ImageFormat(&background, isHDR ? PIXELFORMAT_UNCOMPRESSED_R32G32B32 : PIXELFORMAT_UNCOMPRESSED_R8G8B8);

    CudaRaytracer::SetBackground({
        background.data,
        isHDR ? CudaRaytracer::Background::Format::RGB32F : CudaRaytracer::Background::Format::RGB8,
        background.width,
        background.height,
        3 /*channels*/
    });
    UnloadImage(background);
    ResetAccumulator();
}

void BaseRender::ResetAccumulator()
{
    CudaRaytracer::ResetAccumulator();
    sample = 0;
}

void BaseRender::BeforeDraw()
{
    const CameraState &cameraState = camera.GetState();
    if (lastCameraState != cameraState) {
        ResetAccumulator();
    }

    lastCameraState = cameraState;
}

void BaseRender::AfterDraw()
{
    sample++;
}

void BaseRender::DrawGUI()
{
    DrawFPS(10, 10);
    DrawText(TextFormat("SAMPLES: %d", sample), 10, 38, 20, WHITE);
}

void BaseRender::Resize(
    const int width,
    const int height
)
{
    CudaRaytracer::ResizeAccumulator(width, height);
}
