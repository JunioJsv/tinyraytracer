#include <memory>
#include <raylib.h>

#include "camera_controller.h"
#include "render/render_v2.h"

namespace
{
    CameraController camera;

    void UpdateCamera(
        const float deltaTime
    )
    {
        CameraInput input{};
        const Vector2 mouseDelta(GetMouseDelta());
        input.moveForward = IsKeyDown(KEY_W);
        input.moveBackward = IsKeyDown(KEY_S);
        input.moveLeft = IsKeyDown(KEY_A);
        input.moveRight = IsKeyDown(KEY_D);
        input.moveUp = IsKeyDown(KEY_SPACE);
        input.moveDown = IsKeyDown(KEY_LEFT_CONTROL);
        input.mouseDeltaX = mouseDelta.x;
        input.mouseDeltaY = mouseDelta.y;
        camera.Update(input, deltaTime);
    }
} // namespace

int main()
{
    InitWindow(1024, 720, "Tiny Raytracer");
    DisableCursor();

    const int width = GetScreenWidth(), height = GetScreenHeight();
    const std::unique_ptr<BaseRender> render = std::make_unique<RenderV2>(width, height, camera);
    render->SetBackground("resources/background.png");

    while (!WindowShouldClose()) {
        UpdateCamera(GetFrameTime());
        render->Draw();
    }
}
