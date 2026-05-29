#include <raylib.h>

#include "cuda/cuda_raytracer.h"
#include "camera_controller.h"

namespace
{
	constexpr unsigned int WIDTH = 800;
	constexpr unsigned int HEIGHT = 600;

	uint32_t pixels[WIDTH][HEIGHT];
	CameraController camera;

	static Texture2D SetupTexture()
	{
		const Image image{
			pixels,
			WIDTH,
			HEIGHT,
			1,
			PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
		};
		return LoadTextureFromImage(image);
	}

	static void Render(
		const Texture2D& texture
	)
	{
		const CameraState& cameraState = camera.GetState();
		CudaRaytracer::Render(&pixels[0][0], WIDTH, HEIGHT,
			cameraState.position, cameraState.pitch, cameraState.yaw);
		UpdateTexture(texture, pixels);
	}

	static void UpdateDrawFrame(
		const Texture2D& texture
	)
	{
		BeginDrawing();
		Render(texture);
		DrawTexture(texture, 0, 0, WHITE);
		DrawFPS(10, 10);
		EndDrawing();
	}

	static void UpdateCamera(
		float deltaTime
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
}

int main(int argc, char* argv[]) {
	InitWindow(WIDTH, HEIGHT, "Tiny Raytracer");

	SetTargetFPS(60);
	DisableCursor();

	const Texture2D texture(SetupTexture());

	while (!WindowShouldClose())
	{
		UpdateCamera(GetFrameTime());
		UpdateDrawFrame(texture);
	}
}