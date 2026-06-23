#include <cmath>
#include <format>

#include "camera_controller.h"

#include <algorithm>

CameraController::CameraController()
{
    state.fov = DEFAULT_FOV;
}

void CameraController::Update(
    const CameraInput &input
)
{
    UpdateAxis(input);
    UpdateMovement(input);
    UpdateAttributes(input);
}

void CameraController::UpdateMovement(
    const CameraInput &input
)
{
    float speed = 10.f * GetFrameTime();
    if (input.walk) {
        speed *= 0.2;
    }

    if (input.sprint) {
        speed *= 3;
    }

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
    const float deltaTime = GetFrameTime();
    state.yaw -= input.mouseDeltaX * MOUSE_SENSITIVITY * deltaTime;
    state.pitch -= input.mouseDeltaY * MOUSE_SENSITIVITY * deltaTime;

    state.yaw = std::fmod(state.yaw, YAW_LIMIT);
    state.pitch = std::clamp(state.pitch, -PITCH_LIMIT, PITCH_LIMIT);
    UpdateDirections();
}

void CameraController::UpdateAttributes(
    const CameraInput &input
)
{
    if (const float factor = -input.mouseWheelDelta * 8 * GetFrameTime(); factor != 0) {
        state.fov = std::clamp(state.fov + factor, MIN_FOV, MAX_FOV);
    }
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
