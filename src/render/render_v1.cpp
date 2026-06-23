#include "render_v1.h"

RenderV1::RenderV1(
    const int width,
    const int height
) : BaseRender(width, height)
  , pixels(width * height)
  , image(SetupImage(pixels.data(), width, height))
  , texture(LoadTextureFromImage(image)) {}

RenderV1::~RenderV1()
{
    UnloadTexture(texture);
}

void RenderV1::Draw()
{
    BeforeDraw();
    CudaRaytracer::Render(static_cast<uint32_t *>(image.data),
                          camera.GetState(), UpdateRenderInfo());
    UpdateTexture(texture, image.data);
    DrawTexture(texture, 0, 0, WHITE);
    AfterDraw();
}

void RenderV1::Resize(
    const int width,
    const int height
)
{
    BaseRender::Resize(width, height);
    UnloadTexture(texture);
    pixels.clear();
    pixels.resize(width * height);
    image = SetupImage(pixels.data(), width, height);
    texture = LoadTextureFromImage(image);
}

const CudaRaytracer::RenderInfo &RenderV1::UpdateRenderInfo()
{
    BaseRender::UpdateRenderInfo();
    renderInfo.width = image.width;
    renderInfo.height = image.height;
    return renderInfo;
}
