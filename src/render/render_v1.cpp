#include "render_v1.h"

RenderV1::RenderV1(
    const int width,
    const int height,
    const CameraController &camera
) : BaseRender(width, height, camera)
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
    BeginDrawing();
    CudaRaytracer::Render(static_cast<uint32_t *>(image.data),
                          image.width, image.height, camera.GetState(), sample);
    UpdateTexture(texture, image.data);
    DrawTexture(texture, 0, 0, WHITE);
    DrawGUI();
    EndDrawing();
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
