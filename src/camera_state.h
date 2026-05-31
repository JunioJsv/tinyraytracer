#ifndef CAMERA_STATE_H
#define CAMERA_STATE_H

#include <string>

#include "cuda/cuda_raytracer.h"

struct CameraState
{
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

    float mouseDeltaX;
    float mouseDeltaY;
};

#endif // !CAMERA_STATE_H
