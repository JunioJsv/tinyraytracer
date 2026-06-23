#ifndef INPUT_MANAGER_H
#define INPUT_MANAGER_H
#include "camera_controller.h"

class BaseRender;

class InputManager
{
public:
    static void RenderInputs(
        BaseRender &render
    );

    static void CameraInputs(
        CameraController &camera
    );
};


#endif
