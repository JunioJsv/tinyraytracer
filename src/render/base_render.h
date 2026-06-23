#ifndef BASE_RENDER_H
#define BASE_RENDER_H

#include <raylib.h>
#include <rlImGui.h>
#include <string>

#include "../camera_controller.h"
#include "../cuda/cuda_raytracer.h"

class BaseRender
{
public:
    BaseRender(
        const int width,
        const int height
    ) : renderInfo{
          .enableAccumulator = true,
          .enableReflection = true,
          .enableRefraction = true,
          .enableDiffuse = true,
          .enableLights = true,
          .maxDepth = 4
      }
      , lastCameraState(camera.GetState())
      , samples(0)
      , frames(0)
    {
        CudaRaytracer::Initialize();
        CudaRaytracer::SetupAccumulator(width, height);
        rlImGuiSetup(true /*darkTheme*/);
    }

    virtual ~BaseRender()
    {
        CudaRaytracer::Destroy();
        rlImGuiShutdown();
    }

    void SetBackground(
        const std::string &fileName
    );

    void ResetAccumulator();

    void BeforeDraw();

    void ProcessInputs();

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

    bool WantCaptureCursor() const;

    virtual const CudaRaytracer::RenderInfo &UpdateRenderInfo();

protected:
    CudaRaytracer::RenderInfo renderInfo;
    CameraController camera;
    CameraState lastCameraState;
    size_t samples;
    size_t frames;
};

#endif
