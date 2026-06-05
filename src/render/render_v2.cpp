#include "render_v2.h"

RenderV2::RenderV2(
    const int width,
    const int height,
    const CameraController &camera
) : camera(camera)
  , texture(LoadTextureFromImage(SetupImage(nullptr, width, height)))
  , cudaTexture(CudaRaytracer::SetupCudaTexture(texture.id)) {}

RenderV2::~RenderV2()
{
    UnloadTexture(texture);
}

void RenderV2::Draw()
{
    BeginDrawing();
    CudaRaytracer::Render(cudaTexture, texture.width, texture.height, camera.GetState());
    DrawTexture(texture, 0, 0, WHITE);
    DrawGUI();
    EndDrawing();
}

void RenderV2::Resize(
    const int width,
    const int height
)
{
    CudaRaytracer::DestroyCudaTexture(cudaTexture);
    UnloadTexture(texture);
    texture = LoadTextureFromImage(SetupImage(nullptr, width, height));
    cudaTexture = CudaRaytracer::SetupCudaTexture(texture.id);
}
