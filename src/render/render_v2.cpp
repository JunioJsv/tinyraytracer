#include "render_v2.h"

RenderV2::RenderV2(
    const int width,
    const int height,
    const CameraController &camera
) : BaseRender(width, height, camera)
  , texture(LoadTextureFromImage(SetupImage(nullptr, width, height)))
  , cudaTexture(CudaRaytracer::SetupCudaTexture(texture.id)) {}

RenderV2::~RenderV2()
{
    UnloadTexture(texture);
}

void RenderV2::Draw()
{
    BeforeDraw();
    BeginDrawing();
    CudaRaytracer::Render(cudaTexture, texture.width, texture.height, camera.GetState(), sample);
    DrawTexture(texture, 0, 0, WHITE);
    DrawGUI();
    EndDrawing();
    AfterDraw();
}

void RenderV2::Resize(
    const int width,
    const int height
)
{
    BaseRender::Resize(width, height);
    CudaRaytracer::DestroyCudaTexture(cudaTexture);
    UnloadTexture(texture);
    texture = LoadTextureFromImage(SetupImage(nullptr, width, height));
    cudaTexture = CudaRaytracer::SetupCudaTexture(texture.id);
}
