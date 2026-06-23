#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include <raylib.h>
#include <string>

#include "camera_state.h"

class CameraController
{
public:
    CameraController();

    void Update(
        const CameraInput &input
    );

    [[nodiscard]] const CameraState &GetState() const
    {
        return state;
    }

    [[nodiscard]] std::string ToString() const;

    static constexpr float DEFAULT_FOV = 90.f * DEG2RAD;
    static constexpr float MAX_FOV = 120.f * DEG2RAD;
    static constexpr float MIN_FOV = .5f * DEG2RAD;

    static constexpr float PITCH_LIMIT = 89.f * DEG2RAD;
    static constexpr float YAW_LIMIT = 360.f * DEG2RAD;
    static constexpr float MOUSE_SENSITIVITY = 24.f * DEG2RAD;

private:
    void UpdateMovement(
        const CameraInput &input
    );

    void UpdateAxis(
        const CameraInput &input
    );

    void UpdateAttributes(
        const CameraInput &input
    );

    void UpdateDirections();

    CameraState state{};
};

#endif // !CAMERA_CONTROLLER_H
