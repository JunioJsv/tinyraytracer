#ifndef CAMERA_STATE_H
#define CAMERA_STATE_H

#include "cuda/cuda_raytracer.h"

struct CameraState
{
    CudaRaytracer::Vec3 position;
    float pitch;
    float yaw;
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
