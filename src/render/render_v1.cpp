#include "render_v1.h"

RenderV1::RenderV1(
    const int width,
    const int height,
    const CameraController &camera
) : pixels(width * height)
  , camera(camera)
  , image(pixels.data(), width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
  , texture(LoadTextureFromImage(image))
{
    Initialize();
}

RenderV1::~RenderV1()
{
    UnloadTexture(texture);
}

void RenderV1::Draw()
{
    BeginDrawing();
    CudaRaytracer::Render(static_cast<uint32_t *>(image.data),
                          image.width, image.height, camera.GetState());
    UpdateTexture(texture, image.data);
    DrawTexture(texture, 0, 0, WHITE);
    DrawFPS(10, 10);
    EndDrawing();
}
