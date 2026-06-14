#ifndef CAMERA_STATE_H
#define CAMERA_STATE_H

#include <string>

#include "cuda/cuda_raytracer.h"

struct CameraState
{
    friend bool operator==(
        const CameraState &lhs,
        const CameraState &rhs
    )
    {
        return lhs.position == rhs.position
               && lhs.forward == rhs.forward
               && lhs.right == rhs.right
               && lhs.up == rhs.up
               && lhs.pitch == rhs.pitch
               && lhs.yaw == rhs.yaw
               && lhs.fov == rhs.fov;
    }

    friend bool operator!=(
        const CameraState &lhs,
        const CameraState &rhs
    ) { return !(lhs == rhs); }

    [[nodiscard]] std::string ToString() const;

    CudaRaytracer::Vec3 position;
    CudaRaytracer::Vec3 forward;
    CudaRaytracer::Vec3 right;
    CudaRaytracer::Vec3 up;
    float pitch;
    float yaw;
    float fov;
};

struct CameraInput
{
    bool moveForward;
    bool moveBackward;
    bool moveLeft;
    bool moveRight;
    bool moveUp;
    bool moveDown;

    bool walk;
    bool sprint;

    float mouseDeltaX;
    float mouseDeltaY;
    float mouseWheelDelta;
};

#endif // !CAMERA_STATE_H
