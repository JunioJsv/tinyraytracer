#include "input_manager.h"

#include <raylib.h>

#include "render/base_render.h"

void InputManager::RenderInputs(
    BaseRender &render
)
{
    if (IsKeyPressed(KEY_F)) {
        ToggleFullscreen();
        const int width = GetScreenWidth(), height = GetScreenHeight();
        render.Resize(width, height);
    }
}

void InputManager::CameraInputs(
    CameraController &camera
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
    input.walk = IsKeyDown(KEY_LEFT_ALT);
    input.sprint = IsKeyDown(KEY_LEFT_SHIFT);
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        input.mouseDeltaX = mouseDelta.x;
        input.mouseDeltaY = mouseDelta.y;
    }
    input.mouseWheelDelta = GetMouseWheelMove();
    camera.Update(input);
}
