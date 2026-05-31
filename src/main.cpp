#include <raylib.h>
#include <vector>

#include "camera_controller.h"
#include "cuda/cuda_raytracer.h"

namespace
{
    CameraController camera;

    void Render(
        const Image &image
    )
    {
        CudaRaytracer::Render(static_cast<uint32_t *>(image.data),
                              image.width, image.height, camera.GetState());
    }

    void UpdateDrawFrame(
        const Image &image,
        const Texture2D &texture
    )
    {
        BeginDrawing();
        Render(image);
        UpdateTexture(texture, image.data);
        DrawTexture(texture, 0, 0, WHITE);
        DrawFPS(10, 10);
        EndDrawing();
    }

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
    InitWindow(1280, 720, "Tiny Raytracer");
    DisableCursor();

    const int width = GetScreenWidth(), height = GetScreenHeight();
    std::vector<uint32_t> pixels(width * height);

    const Image image{pixels.data(), width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    const Texture2D texture{LoadTextureFromImage(image)};

    while (!WindowShouldClose()) {
        UpdateCamera(GetFrameTime());
        UpdateDrawFrame(image, texture);
    }

    UnloadTexture(texture);
}
