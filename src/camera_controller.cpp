#include <cmath>

#include "camera_controller.h"

void CameraController::Update(
    const CameraInput &input,
    float deltaTime
)
{
    constexpr float moveSpeed = 5.f;
    constexpr float mouseSensitivity = 0.01f;
    if (input.moveForward) {
        state.position -= CudaRaytracer::Vec3{std::sin(state.yaw), 0.f, std::cos(state.yaw)} * moveSpeed * deltaTime;
    }
    if (input.moveBackward) {
        state.position += CudaRaytracer::Vec3{std::sin(state.yaw), 0.f, std::cos(state.yaw)} * moveSpeed * deltaTime;
    }
    if (input.moveLeft) {
        state.position -= CudaRaytracer::Vec3{std::cos(state.yaw), 0.f, -std::sin(state.yaw)} * moveSpeed * deltaTime;
    }
    if (input.moveRight) {
        state.position += CudaRaytracer::Vec3{std::cos(state.yaw), 0.f, -std::sin(state.yaw)} * moveSpeed * deltaTime;
    }
    if (input.moveUp) {
        state.position.y += moveSpeed * deltaTime;
    }
    if (input.moveDown) {
        state.position.y -= moveSpeed * deltaTime;
    }
    state.yaw -= input.mouseDeltaX * mouseSensitivity;
    state.pitch -= input.mouseDeltaY * mouseSensitivity;
}
