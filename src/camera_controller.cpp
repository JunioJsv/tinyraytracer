#include <cmath>
#include <format>

#include "camera_controller.h"

#include <algorithm>
#include <raylib.h>

CameraController::CameraController()
{
    state.fov = 90.0f * DEG2RAD;
}

void CameraController::Update(
    const CameraInput &input,
    const float deltaTime
)
{
    UpdateAxis(input);
    UpdateMovement(input, deltaTime);
}

void CameraController::UpdateMovement(
    const CameraInput &input,
    const float deltaTime
)
{
    const float speed = 10.f * deltaTime;

    CudaRaytracer::Vec3 movement{};
    if (input.moveForward) {
        movement += state.forward;
    }
    if (input.moveBackward) {
        movement -= state.forward;
    }
    if (input.moveLeft) {
        movement -= state.right;
    }
    if (input.moveRight) {
        movement += state.right;
    }
    if (input.moveUp) {
        movement += state.up;
    }
    if (input.moveDown) {
        movement -= state.up;
    }

    state.position += movement.Normalized() * speed;
}

void CameraController::UpdateAxis(
    const CameraInput &input
)
{
    constexpr float PITCH_LIMIT = 89.f * DEG2RAD;
    constexpr float YAW_LIMIT = 360.f * DEG2RAD;
    constexpr float SENSITIVITY = 0.1f * DEG2RAD;

    state.yaw -= input.mouseDeltaX * SENSITIVITY;
    state.pitch -= input.mouseDeltaY * SENSITIVITY;

    state.yaw = std::fmod(state.yaw, YAW_LIMIT);
    state.pitch = std::clamp(state.pitch, -PITCH_LIMIT, PITCH_LIMIT);
    UpdateDirections();
}

void CameraController::UpdateDirections()
{
    constexpr CudaRaytracer::Vec3 up{0.f, 1.f, 0.f};
    const CudaRaytracer::Vec3 forward{
        std::cosf(state.pitch) * std::sinf(state.yaw),
        std::sinf(state.pitch),
        -std::cosf(state.pitch) * std::cosf(state.yaw)
    };

    state.forward = forward.Normalized();

    state.right = up.Cross(state.forward).Normalized();
    state.up = state.forward.Cross(state.right);
}

std::string CameraController::ToString() const
{
    return std::format("CameraController{{ state: {} }}", state.ToString());
}
