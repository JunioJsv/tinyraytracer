#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include <string>

#include "camera_state.h"

class CameraController
{
public:
    CameraController();

    void Update(
        const CameraInput &input,
        float deltaTime
    );

    [[nodiscard]] const CameraState &GetState() const
    {
        return state;
    }

    [[nodiscard]] std::string ToString() const;

private:
    void UpdateMovement(
        const CameraInput &input,
        float deltaTime
    );

    void UpdateAxis(
        const CameraInput &input
    );

    void UpdateDirections();

    CameraState state{};
};

#endif // !CAMERA_CONTROLLER_H
