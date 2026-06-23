#include <memory>
#include <raylib.h>

#include "render/render_v2.h"

int main()
{
    InitWindow(1024, 720, "Tiny Raytracer");

    const int width = GetScreenWidth(), height = GetScreenHeight();
    const std::unique_ptr<BaseRender> render = std::make_unique<RenderV2>(width, height);
    render->SetBackground("resources/background.hdr");

    while (!WindowShouldClose()) {
        render->Draw();
    }
}
