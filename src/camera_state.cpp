#include "camera_state.h"

#include <format>

std::string CameraState::ToString() const
{
    auto &[x,y,z] = position;
    return std::format("CameraState{{ x: {} y: {} z: {}, pitch: {}, yaw: {}, fov: {} }}",
                       x, y, z, pitch, yaw, fov);
}