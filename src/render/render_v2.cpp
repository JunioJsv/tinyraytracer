#include "render_v2.h"

RenderV2::RenderV2(
    const int width,
    const int height,
    const CameraController &camera
) : camera(camera)
  , texture(LoadTextureFromImage(
      {nullptr, width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8}))
  , cudaTexture(CudaRaytracer::SetupCudaTexture(texture.id))
{
    Initialize();
}

RenderV2::~RenderV2()
{
    UnloadTexture(texture);
}

void RenderV2::Draw()
{
    BeginDrawing();
    CudaRaytracer::Render(cudaTexture, texture.width, texture.height, camera.GetState());
    DrawTexture(texture, 0, 0, WHITE);
    DrawFPS(10, 10);
    EndDrawing();
}
