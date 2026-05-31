#include <raylib.h>

#include "camera_controller.h"
#include "cuda/cuda_raytracer.h"

namespace
{
    constexpr unsigned int WIDTH = 800;
    constexpr unsigned int HEIGHT = 600;

    uint32_t pixels[WIDTH][HEIGHT];
    CameraController camera;

    Texture2D SetupTexture()
    {
        constexpr Image image{pixels, WIDTH, HEIGHT, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        return LoadTextureFromImage(image);
    }

    void Render(
        const Texture2D &texture
    )
    {
        CudaRaytracer::Render(&pixels[0][0], WIDTH, HEIGHT, camera.GetState());
        UpdateTexture(texture, pixels);
    }

    void UpdateDrawFrame(
        const Texture2D &texture
    )
    {
        BeginDrawing();
        Render(texture);
        DrawTexture(texture, 0, 0, WHITE);
        DrawFPS(10, 10);
        EndDrawing();
    }

    void UpdateCamera(
        const float deltaTime
    )
    {
        CameraInput input{};
        Vector2 mouseDelta(GetMouseDelta());
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
    InitWindow(WIDTH, HEIGHT, "Tiny Raytracer");

    DisableCursor();

    const Texture2D texture(SetupTexture());

    while (!WindowShouldClose()) {
        UpdateCamera(GetFrameTime());
        UpdateDrawFrame(texture);
    }

    UnloadTexture(texture);
}
