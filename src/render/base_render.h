#ifndef BASE_RENDER_H
#define BASE_RENDER_H

#include <raylib.h>
#include <string>

#include "../camera_controller.h"
#include "../cuda/cuda_raytracer.h"

class BaseRender
{
public:
    BaseRender(
        const int width,
        const int height,
        const CameraController &camera
    ) : camera(camera)
      , lastCameraState(camera.GetState())
      , sample(0)
    {
        CudaRaytracer::Initialize();
        CudaRaytracer::SetupAccumulator(width, height);
    }

    virtual ~BaseRender()
    {
        CudaRaytracer::Destroy();
    }

    void SetBackground(
        const std::string &fileName
    );

    void ResetAccumulator();

    void BeforeDraw();

    void AfterDraw();

    virtual void Draw() = 0;

    virtual void DrawGUI();

    virtual void Resize(
        int width,
        int height
    );

    static Image SetupImage(
        void *data,
        const int width,
        const int height
    )
    {
        return {data, width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    }

protected:
    const CameraController &camera;
    CameraState lastCameraState;
    size_t sample;
};

#endif
