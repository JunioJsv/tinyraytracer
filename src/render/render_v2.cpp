#include "render_v2.h"

RenderV2::RenderV2(
    const int width,
    const int height
) : BaseRender(width, height)
  , texture(LoadTextureFromImage(SetupImage(nullptr, width, height)))
  , cudaTexture(CudaRaytracer::SetupCudaTexture(texture.id)) {}

RenderV2::~RenderV2()
{
    UnloadTexture(texture);
}

void RenderV2::Draw()
{
    BeforeDraw();
    CudaRaytracer::Render(cudaTexture, camera.GetState(), UpdateRenderInfo());
    DrawTexture(texture, 0, 0, WHITE);
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

const CudaRaytracer::RenderInfo &RenderV2::UpdateRenderInfo()
{
    BaseRender::UpdateRenderInfo();
    renderInfo.width = texture.width;
    renderInfo.height = texture.height;
    return renderInfo;
}
