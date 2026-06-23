#include "cuda_raytracer.h"

#include <external/glfw/deps/glad/gl.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "../camera_state.h"
#include "stack.cuh"

namespace CudaRaytracer
{
    constexpr float PI = 3.14159265358979323846f;

    constexpr Material IVORY = {1.0, {0.9, 0.5, 0.1, 0.0}, {0.4, 0.4, 0.3}, 50.};
    constexpr Material GLASS = {1.5, {0.0, 0.9, 0.1, 0.8}, {0.6, 0.7, 0.8}, 125.};
    constexpr Material RED_NEON = {1.0, {1.4, 0.3, 0.0, 0.0}, {5., 0.1, 0.1}, 10.};
    constexpr Material MIRROR = {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.};

    constexpr int N_SPHERES = 4;
    constexpr Sphere CPU_SPHERES[] = {
        {{-3, 0, -16}, 2, IVORY},
        {{-1.0, -1.5, -12}, 2, GLASS},
        {{1.5, -0.5, -18}, 3, RED_NEON},
        {{7, 5, -18}, 4, MIRROR}
    };

    constexpr int N_LIGHTS = 3;
    constexpr SphereLight CPU_LIGHTS[] = {
        {-20, 20, 20, 2.5f, 1500.f}, {30, 50, -25, 8.f, 2000.f}, {30, 20, 30, 5.f, 5000.f}
    };

    GPU __constant__ Sphere GPU_SPHERES[N_SPHERES];
    GPU __constant__ SphereLight GPU_LIGHTS[N_LIGHTS];
    GPU __constant__ Background GPU_BACKGROUND;
    GPU __constant__ CameraState GPU_CAMERA;
    GPU __constant__ Vec3 *GPU_ACCUMULATOR;
    Vec3 *accumulatorPtr = nullptr;
    size_t accumulatorBytes{0};

    float Vec3::Length() const
    {
        return sqrtf(x * x + y * y + z * z);
    }

    Vec3 Vec3::Normalized() const
    {
        if (const float length = Length(); length > 1e-6) {
            return *this * (1.f / length);
        }

        return {0, 0, 0};
    }

    Vec3 Vec3::Cross(
        const Vec3 &v
    ) const
    {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }

    unsigned int Background::GetIndex(
        const Vec3 &dir
    ) const
    {
        const float u = atan2(dir.z, dir.x) / (2.0f * PI) + 0.5f;
        const float v = acos(fmin(fmax(dir.y, -1.0f), 1.0f)) / PI;

        const int x = static_cast<int>(u * static_cast<float>(width - 1));

        const int y = static_cast<int>(v * static_cast<float>(height - 1));

        const unsigned int index = (y * width + x) * channels;

        return index;
    }

    Vec3 Background::GetLDRColor(
        const Vec3 &dir
    ) const
    {
        Vec3 color{};
        const uint8_t *channel = static_cast<uint8_t *>(data) + GetIndex(dir);

        if (channels > 0) {
            color.x = static_cast<float>(channel[0]) / 255.f;
        }
        if (channels > 1) {
            color.y = static_cast<float>(channel[1]) / 255.f;
        }
        if (channels > 2) {
            color.z = static_cast<float>(channel[2]) / 255.f;
        }

        return color;
    }

    Vec3 Background::GetHDRColor(
        const Vec3 &dir
    ) const
    {
        Vec3 color{};
        const float *channel = static_cast<float *>(data) + GetIndex(dir);

        if (channels > 0) {
            color.x = channel[0];
        }
        if (channels > 1) {
            color.y = channel[1];
        }
        if (channels > 2) {
            color.z = channel[2];
        }

        return color;
    }

    Vec3 Reflect(
        const Vec3 &I,
        const Vec3 &N
    )
    {
        return I - N * 2.f * (I * N);
    }

    Vec3 Refract(
        const Vec3 I,
        Vec3 N,
        float etaT,
        float etaI
    )
    {
        float cosi = -fmaxf(
            -1.f,
            fminf(1.f, I * N)
        );

        if (cosi < 0.f) {
            cosi = -cosi;
            N = -N;

            const float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
        }

        const float eta = etaI / etaT;

        const float k = 1.f - eta * eta *
                        (1.f - cosi * cosi);

        if (k < 0.f) {
            return Vec3{};
        }

        return I * eta + N * (eta * cosi - sqrtf(k));
    }

    RayIntersection RaySphereIntersect(
        const Vec3 &orig,
        const Vec3 &dir,
        const Sphere &s
    )
    {
        const Vec3 L = s.center - orig;
        const float tca = L * dir;
        const float d2 = L * L - tca * tca;
        if (d2 > s.radius * s.radius)
            return {false, 0};
        const float thc = sqrtf(s.radius * s.radius - d2);
        const float t0 = tca - thc, t1 = tca + thc;
        if (t0 > .001f)
            return {true, t0};
        if (t1 > .001f)
            return {true, t1};
        return {false, 0};
    }

    GPU Vec3 RandomSphereDir(
        RNG &rng
    )
    {
        const float u = rng.RandomFloat();
        const float v = rng.RandomFloat();

        const float phi = 2.0f * PI * u;

        const float cosTheta = v;
        const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

        return {
            cosf(phi) * sinTheta,
            cosTheta,
            sinf(phi) * sinTheta
        };
    }

    GPU Vec3 RandomHemisphereDir(
        const Vec3 &N,
        RNG &rng
    )
    {
        Vec3 dir = RandomSphereDir(rng);

        if (dir * N < 0.0f) {
            dir = -dir;
        }

        return dir;
    }

    GPU uint32_t Hash(
        uint32_t x
    )
    {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }

    void Initialize()
    {
        CUDA_CHECK(cudaMemcpyToSymbol(GPU_SPHERES, CPU_SPHERES, N_SPHERES * sizeof(Sphere)));
        CUDA_CHECK(cudaMemcpyToSymbol(GPU_LIGHTS, CPU_LIGHTS, N_LIGHTS * sizeof(SphereLight)));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    }

    void Destroy()
    {
        if (accumulatorPtr) {
            CUDA_CHECK(cudaFree(accumulatorPtr));
        }
    }

    void SetupAccumulator(
        const int width,
        const int height
    )
    {
        accumulatorBytes = width * height * sizeof(Vec3);
        CUDA_CHECK(cudaMalloc(&accumulatorPtr, accumulatorBytes));
        CUDA_CHECK(cudaMemcpyToSymbol(GPU_ACCUMULATOR, &accumulatorPtr, sizeof(Vec3*)));
        ResetAccumulator();
    }

    void ResetAccumulator()
    {
        CUDA_CHECK(cudaMemset(accumulatorPtr, 0, accumulatorBytes));
    }

    void ResizeAccumulator(
        const int width,
        const int height
    )
    {
        if (accumulatorPtr) {
            CUDA_CHECK(cudaFree(accumulatorPtr));
        }
        SetupAccumulator(width, height);
    }

    void SetBackground(
        const Background &background
    )
    {
        if (GPU_BACKGROUND.data) {
            CUDA_CHECK(cudaFree(GPU_BACKGROUND.data));
        }

        void *data;

        CUDA_CHECK(cudaMalloc(
            &data,
            background.GetDataSizeInBytes()
        ));

        CUDA_CHECK(cudaMemcpy(
            data,
            background.data,
            background.GetDataSizeInBytes(),
            cudaMemcpyHostToDevice
        ));

        Background copy(background);
        copy.data = data;

        CUDA_CHECK(cudaMemcpyToSymbol(GPU_BACKGROUND, &copy, sizeof(Background)));
    }

    cudaGraphicsResource *SetupCudaTexture(
        const unsigned int glTextureId
    )
    {
        cudaGraphicsResource *texture;

        CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &texture,
            glTextureId,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsSurfaceLoadStore
        ));

        return texture;
    }

    void DestroyCudaTexture(
        cudaGraphicsResource *texture
    )
    {
        CUDA_CHECK(cudaGraphicsUnregisterResource(texture));
    }

    struct SceneIntersection
    {
        bool hit;
        Vec3 pt, N;
        Material material;
    };

    GPU SceneIntersection SceneIntersect(
        const Vec3 &orig,
        const Vec3 &dir
    )
    {
        Vec3 pt{}, N{};
        Material material{};

        float nearestDist = 1e10f;

        if (fabsf(dir.y) > .001f) {
            const float d = -(orig.y + 4.f) / dir.y;
            const Vec3 p = orig + dir * d;
            if (d > .001f && d < nearestDist && fabsf(p.x) < 10.f && p.z < -10.f && p.z > -30.f) {
                nearestDist = d;
                pt = p;
                N = {0, 1, 0};
                material.diffuseColor =
                        (int(.5f * pt.x + 1000.f) + (int) (.5f * pt.z)) & 1 ? Vec3{.3f, .3f, .3f} : Vec3{.3f, .2f, .1f};
            }

            material.refractiveIdx = 1.0f;
            material.albedo[0] = 1.0f;
            material.albedo[1] = 0.0f;
            material.albedo[2] = 0.0f;
            material.albedo[3] = 0.0f;
            material.specularExponent = 0.0f;
        }

        for (int i = 0; i < N_SPHERES; ++i) {
            auto [intersection, d] = RaySphereIntersect(orig, dir, GPU_SPHERES[i]);
            if (!intersection || d > nearestDist)
                continue;
            nearestDist = d;
            pt = orig + dir * nearestDist;
            N = (pt - GPU_SPHERES[i].center).Normalized();
            material = GPU_SPHERES[i].material;
        }

        return {nearestDist < 1000.f, pt, N, material};
    }

    GPU bool ReflectionRay(
        const RayState &current,
        const SceneIntersection &intersection,
        RayState &out
    )
    {
        constexpr float offset = 1e-3f;
        const auto &[hit, point, N, material] = intersection;

        if (material.albedo[2] < 0.01f) return false;

        const Vec3 reflectDir = Reflect(current.dir, N).Normalized();
        const Vec3 reflectOrig = point + N * offset;
        const Vec3 throughput = current.throughput * material.albedo[2];

        const unsigned int depth = current.depth + 1;
        out = RayState{
            reflectOrig, reflectDir, depth, throughput,
            RayType::REFLECTION, RNG{current.rng.state ^ depth}
        };

        return true;
    }

    GPU bool RefractionRay(
        const RayState &current,
        const SceneIntersection &intersection,
        RayState &out
    )
    {
        constexpr float offset = 1e-3f;
        const auto &[hit, point, N, material] = intersection;

        if (material.albedo[3] < 0.01f) return false;

        Vec3 refractDir = Refract(current.dir, N, material.refractiveIdx, 1.0f);

        if (refractDir.Length() < 0.1f) return false;

        refractDir = refractDir.Normalized();

        const float dotProduct = current.dir * N;
        Vec3 refractOrig{};
        if (dotProduct < 0) {
            refractOrig = point - N * offset;
        } else {
            refractOrig = point + N * offset;
        }

        const Vec3 throughput = current.throughput * material.albedo[3];

        const unsigned int depth = current.depth + 1;
        out = RayState{
            refractOrig, refractDir, depth, throughput,
            RayType::REFRACTION, RNG{current.rng.state ^ depth}
        };

        return true;
    }

    GPU bool DiffuseRay(
        RayState &current,
        const SceneIntersection &intersection,
        RayState &out
    )
    {
        constexpr float offset = 1e-3f;
        const auto &[hit, point, N, material] = intersection;

        if (material.albedo[0] < 0.01f) return false;

        const Vec3 diffuseDir = RandomHemisphereDir(N, current.rng);
        const Vec3 diffuseOrig = point + N * offset;
        const Vec3 throughput = current.throughput.Mul(material.diffuseColor) * material.albedo[0];

        const unsigned int depth = current.depth + 1;
        out = RayState{
            diffuseOrig, diffuseDir, depth, throughput,
            RayType::DIFFUSE, RNG{current.rng.state ^ depth}
        };

        return true;
    }

    GPU float ComputeAO(
        const SceneIntersection &intersection,
        RNG &rng
    )
    {
        constexpr int samples = 8;
        constexpr float offset = 1e-3f;

        const auto &point = intersection.pt;
        const auto &N = intersection.N;

        int visible = 0;

        for (int i = 0; i < samples; ++i) {
            Vec3 dir = RandomHemisphereDir(N, rng);

            const bool hit = SceneIntersect(point + N * offset, dir).hit;

            if (!hit) {
                visible++;
            }
        }

        return static_cast<float>(visible) / static_cast<float>(samples);
    }

    GPU Vec3 ComputeLights(
        RayState &current,
        const SceneIntersection &intersection
    )
    {
        constexpr int samples = 8;
        constexpr float offset = 1e-3f;
        const auto &[hit, point, N, material] = intersection;
        float diffuseLightIntensity = 0, specularLightIntensity = 0;

        for (int i = 0; i < N_LIGHTS; ++i) {
            const SphereLight &light = GPU_LIGHTS[i];
            for (int sample = 0; sample < samples; ++sample) {
                const Vec3 samplePoint = light.origin + RandomSphereDir(current.rng) * light.radius;
                Vec3 toLight = samplePoint - point;

                const float distance = toLight.Length();

                Vec3 lightDir = toLight / distance;

                auto [hitShadow, shadowPt, trashnrm, trashmat] = SceneIntersect(point + N * offset, lightDir);
                if (hitShadow && (shadowPt - point).Length() < distance)
                    continue;

                const float attenuation = light.intensity / (distance * distance);

                diffuseLightIntensity += fmaxf(0.f, lightDir * N) * attenuation / samples;
                specularLightIntensity +=
                        powf(fmaxf(0.f, -Reflect(-lightDir, N) * current.dir), material.specularExponent)
                        * attenuation / samples;
            }
        }

        return material.diffuseColor * diffuseLightIntensity * material.albedo[0] +
               Vec3{1., 1., 1.} * specularLightIntensity * material.albedo[1];
    }

    GPU Vec3 CastRay(
        const Vec3 &orig,
        const Vec3 &dir,
        const RenderInfo &info,
        const uint32_t seed
    )
    {
        Stack<RayState, 10> stack;
        stack.Push(RayState{orig, dir, 0, Vec3{1.f, 1.f, 1.f}, RayType::PRIMARY, RNG{seed}});

        Vec3 color{};
        RayState current{};
        RayState next{};
        while (stack.Pop(current)) {
            if (current.depth > info.maxDepth) {
                continue;
            }

            const SceneIntersection intersection = SceneIntersect(current.orig, current.dir);

            if (!intersection.hit) {
                const Vec3 ambient = GPU_BACKGROUND.GetColor(current.dir);
                color += ambient.Mul(current.throughput);
                continue;
            }

            if (info.enableLights) {
                color += ComputeLights(current, intersection).Mul(current.throughput) *
                        ComputeAO(intersection, current.rng);
            }

            if (info.enableDiffuse && DiffuseRay(current, intersection, next)) {
                stack.Push(static_cast<RayState &&>(next));
            }

            if (info.enableRefraction && RefractionRay(current, intersection, next)) {
                stack.Push(static_cast<RayState &&>(next));
            }

            if (info.enableReflection && ReflectionRay(current, intersection, next)) {
                stack.Push(static_cast<RayState &&>(next));
            }
        }

        return color;
    }

    GPU Vec3 CastRayDir(
        const unsigned int x,
        const unsigned int y,
        const int width,
        const int height,
        const CameraState &camera
    )
    {
        const float dirX = (x + 0.5f) - width / 2.f;
        const float dirY = -(y + 0.5f) + height / 2.f;
        const float dirZ = -height / (2.f * tanf(camera.fov / 2.f));

        return (camera.forward * -dirZ +
                camera.right * dirX +
                camera.up * dirY).Normalized();
    }

    GPU Vec3 ToneMap(
        const Vec3 &color,
        const float exposure = 1.0f
    )
    {
        return Vec3(
            1.0f - expf(-color.x * exposure),
            1.0f - expf(-color.y * exposure),
            1.0f - expf(-color.z * exposure)
        );
    }

    GPU Vec3 GammaCorrect(
        const Vec3 &color,
        const float gamma = 1.f
    )
    {
        const float factor = 1.0f / gamma;
        return Vec3(
            powf(color.x, factor),
            powf(color.y, factor),
            powf(color.z, factor)
        );
    }

    GPU uchar4
    CastRay(
        const unsigned int x,
        const unsigned int y,
        const CameraState &camera,
        const RenderInfo &info
    )
    {
        const unsigned int pixelIdx = y * info.width + x;
        const Vec3 dir = CastRayDir(x, y, info.width, info.height, camera);

        const uint32_t seed = Hash(pixelIdx + info.frames * 1000003u);
        Vec3 color = CastRay(camera.position, dir, info, seed);
        if (info.enableAccumulator) {
            GPU_ACCUMULATOR[pixelIdx] += color;
            color = GPU_ACCUMULATOR[pixelIdx] / static_cast<float>(info.samples + 1);
        }
        if (GPU_BACKGROUND.IsHDR()) {
            color = GammaCorrect(ToneMap(color));
        }
        const auto r = static_cast<unsigned char>(fminf(fmaxf(color.x * 255.0f, 0.0f), 255.0f));
        const auto g = static_cast<unsigned char>(fminf(fmaxf(color.y * 255.0f, 0.0f), 255.0f));
        const auto b = static_cast<unsigned char>(fminf(fmaxf(color.z * 255.0f, 0.0f), 255.0f));

        return make_uchar4(r, g, b, 255);
    }

    __global__ void CastRayKernel(
        uint32_t *pixels,
        const RenderInfo info
    )
    {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int pixelIdx = y * info.width + x;

        const auto [r, g, b, a] = CastRay(x, y, GPU_CAMERA, info);
        const uint32_t color = a << 24 | b << 16 | g << 8 | r;
        pixels[pixelIdx] = color;
    }

    __global__ void CastRayKernel(
        const cudaSurfaceObject_t surface,
        const RenderInfo info
    )
    {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        const auto pixel = CastRay(x, y, GPU_CAMERA, info);

        surf2Dwrite(
            pixel,
            surface,
            x * sizeof(uchar4),
            y
        );
    }

    void Render(
        uint32_t *output,
        const CameraState &camera,
        const RenderInfo &info
    )
    {
        uint32_t *pixels = nullptr;
        const size_t pixelsBytes = info.width * info.height * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&pixels, pixelsBytes));
        CUDA_CHECK(cudaMemcpyToSymbol(GPU_CAMERA, &camera, sizeof(CameraState)));

        constexpr dim3 block(8, 8);

        const dim3 grid(
            (info.width + block.x - 1) / block.x,
            (info.height + block.y - 1) / block.y
        );

        CastRayKernel<<<grid, block>>>(pixels, info);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output, pixels, pixelsBytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(pixels));
    }

    void Render(
        cudaGraphicsResource *texture,
        const CameraState &camera,
        const RenderInfo &info
    )
    {
        CUDA_CHECK(cudaMemcpyToSymbol(GPU_CAMERA, &camera, sizeof(CameraState)));
        CUDA_CHECK(cudaGraphicsMapResources(1, &texture));

        cudaArray_t array;
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
            &array,
            texture,
            0,
            0
        ));

        cudaResourceDesc desc{};
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = array;

        cudaSurfaceObject_t surface;
        CUDA_CHECK(cudaCreateSurfaceObject(&surface, &desc));

        constexpr dim3 block(8, 8);

        const dim3 grid(
            (info.width + block.x - 1) / block.x,
            (info.height + block.y - 1) / block.y
        );

        CastRayKernel<<<grid, block>>>(surface, info);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaDestroySurfaceObject(surface));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &texture));
    }
}
