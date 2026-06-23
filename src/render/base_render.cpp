#include "base_render.h"

#include <imgui.h>

#include "../input_manager.h"

void BaseRender::SetBackground(
    const std::string &fileName
)
{
    Image background = LoadImage(fileName.c_str());
    if (background.data == nullptr) return;

    const bool isHDR = fileName.ends_with(".hdr");

    ImageFormat(&background, isHDR ? PIXELFORMAT_UNCOMPRESSED_R32G32B32 : PIXELFORMAT_UNCOMPRESSED_R8G8B8);

    CudaRaytracer::SetBackground({
        background.data,
        isHDR ? CudaRaytracer::Background::Format::RGB32F : CudaRaytracer::Background::Format::RGB8,
        background.width,
        background.height,
        3 /*channels*/
    });
    UnloadImage(background);
    ResetAccumulator();
}

void BaseRender::ResetAccumulator()
{
    CudaRaytracer::ResetAccumulator();
    samples = 0;
}

void BaseRender::BeforeDraw()
{
    const CameraState &cameraState = camera.GetState();
    if (lastCameraState != cameraState) {
        ResetAccumulator();
    }

    lastCameraState = cameraState;
    BeginDrawing();
    rlImGuiBegin();
    ProcessInputs();
}

void BaseRender::ProcessInputs()
{
    InputManager::RenderInputs(*this);
    if (!WantCaptureCursor()) {
        InputManager::CameraInputs(camera);
    }
}

void BaseRender::AfterDraw()
{
    ImGui::SetNextWindowPos(
        ImVec2(10, 10),
        ImGuiCond_FirstUseEver
    );

    ImGui::SetNextWindowSize(
        ImVec2(250, 235),
        ImGuiCond_FirstUseEver
    );

    DrawGUI();
    rlImGuiEnd();
    EndDrawing();
    if (renderInfo.enableAccumulator) {
        samples++;
    }
    frames++;
}

void BaseRender::DrawGUI()
{
    bool shouldResetAccumulator = false;
    ImGui::Begin("Render Settings");

    ImGui::Text("FPS: %d", GetFPS());
    ImGui::Text("Samples: %lld", samples);

    ImGui::Separator();

    shouldResetAccumulator |= ImGui::Checkbox("Enable Accumulator", &renderInfo.enableAccumulator);
    shouldResetAccumulator |= ImGui::Checkbox("Enable Lights", &renderInfo.enableLights);

    ImGui::SeparatorText("Rays");
    shouldResetAccumulator |= ImGui::Checkbox("Enable Reflection", &renderInfo.enableReflection);
    shouldResetAccumulator |= ImGui::Checkbox("Enable Refraction", &renderInfo.enableRefraction);
    shouldResetAccumulator |= ImGui::Checkbox("Enable Diffuse", &renderInfo.enableDiffuse);
    shouldResetAccumulator |= ImGui::SliderInt("Max Depth", &renderInfo.maxDepth, 0, 8);

    if (shouldResetAccumulator) {
        ResetAccumulator();
    }

    ImGui::End();
}

void BaseRender::Resize(
    const int width,
    const int height
)
{
    CudaRaytracer::ResizeAccumulator(width, height);
}

bool BaseRender::WantCaptureCursor() const
{
    const ImGuiIO &io = ImGui::GetIO();

    return io.WantCaptureMouse;
}

const CudaRaytracer::RenderInfo &BaseRender::UpdateRenderInfo()
{
    renderInfo.frames = frames;
    renderInfo.samples = samples;
    return renderInfo;
}
