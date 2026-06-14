#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#include "cuda_defs.cuh"

struct CameraState;
struct cudaGraphicsResource;

namespace CudaRaytracer
{
    struct alignas(16) Vec3
    {
        float x, y, z;

        ANY float &operator[](
            const int i
        )
        {
            return i == 0 ? x : (1 == i ? y : z);
        }

        ANY const float &operator[](
            const int i
        ) const
        {
            return i == 0 ? x : (1 == i ? y : z);
        }

        ANY Vec3 operator*(
            const float v
        ) const
        {
            return {x * v, y * v, z * v};
        }

        ANY float operator*(
            const Vec3 &v
        ) const
        {
            return x * v.x + y * v.y + z * v.z;
        }

        ANY Vec3 operator+(
            const Vec3 &v
        ) const
        {
            return {x + v.x, y + v.y, z + v.z};
        }

        ANY Vec3 operator-(
            const Vec3 &v
        ) const
        {
            return {x - v.x, y - v.y, z - v.z};
        }

        ANY Vec3 operator-() const
        {
            return {-x, -y, -z};
        }

        ANY Vec3 &operator+=(
            const Vec3 &vec3
        )
        {
            x += vec3.x;
            y += vec3.y;
            z += vec3.z;
            return *this;
        }

        ANY Vec3 &operator-=(
            const Vec3 &vec3
        )
        {
            x -= vec3.x;
            y -= vec3.y;
            z -= vec3.z;
            return *this;
        }

        ANY Vec3 operator/(
            const Vec3 &vec3
        ) const
        {
            return Vec3(
                x / vec3.x,
                y / vec3.y,
                z / vec3.z
            );
        }

        ANY Vec3 operator/(
            const float scalar
        ) const
        {
            return Vec3(
                x / scalar,
                y / scalar,
                z / scalar
            );
        }

        ANY float Length() const;

        ANY Vec3 Normalized() const;

        ANY Vec3 Cross(
            const Vec3 &v
        ) const;

        ANY Vec3 Mul(
            const Vec3 &v
        ) const
        {
            return {x * v.x, y * v.y, z * v.z};
        }
    };

    struct RNG
    {
        uint32_t state;

        GPU uint32_t NextRandom()
        {
            state = state * 1664525u + 1013904223u;
            return state;
        }

        GPU float RandomFloat()
        {
            return static_cast<float>(NextRandom() & 0x00FFFFFF) * (1.0f / 16777216.0f);
        }
    };

    enum class RayType
    {
        PRIMARY,
        REFLECTION,
        REFRACTION,
        DIFFUSE,
    };

    struct RayState
    {
        Vec3 orig;
        Vec3 dir;
        unsigned int depth;
        Vec3 throughput;
        RayType rayType;
        RNG rng;
    };

    struct Background
    {
        static constexpr Vec3 DEFAULT_COLOR{0.2f, 0.7f, 0.8f};

        enum class Format
        {
            RGB8,
            RGB32F
        };

        ANY unsigned int GetChannelSizeInBytes() const
        {
            switch (format) {
                case Format::RGB8:
                    return sizeof(uint8_t);
                case Format::RGB32F:
                    return sizeof(float);
                default:
                    return 0;
            }
        }

        ANY unsigned int GetDataSizeInBytes() const
        {
            return width * height * channels * GetChannelSizeInBytes();
        }

        ANY bool IsHDR() const
        {
            return format == Format::RGB32F;
        }

        ANY bool IsValid() const
        {
            return data && width > 0 && height > 0 && channels > 0;
        }

        ANY unsigned int GetIndex(
            const Vec3 &dir
        ) const;

        ANY Vec3 GetColor(
            const Vec3 &dir
        ) const
        {
            if (!IsValid()) return DEFAULT_COLOR;

            if (IsHDR()) {
                return GetHDRColor(dir);
            }

            return GetLDRColor(dir);
        }

        ANY Vec3 GetLDRColor(
            const Vec3 &dir
        ) const;

        ANY Vec3 GetHDRColor(
            const Vec3 &dir
        ) const;

        void *data;
        Format format;
        int width;
        int height;
        int channels;
    };

    struct Material
    {
        float refractiveIdx;
        float albedo[4];
        Vec3 diffuseColor;
        float specularExponent;
    };

    struct Sphere
    {
        Vec3 center;
        float radius;
        Material material;
    };

    ANY Vec3 Reflect(
        const Vec3 &I,
        const Vec3 &N
    );

    ANY Vec3 Refract(
        const Vec3 &I,
        const Vec3 &N,
        float etaT,
        float etaI
    );

    struct RayIntersection
    {
        bool hit;
        float distance;
    };

    ANY RayIntersection RaySphereIntersect(
        const Vec3 &orig,
        const Vec3 &dir,
        const Sphere &s
    );

    CPU void Initialize();

    CPU void Destroy();

    CPU void SetBackground(
        const Background &background
    );

    CPU cudaGraphicsResource *SetupCudaTexture(
        unsigned int glTextureId
    );

    CPU void DestroyCudaTexture(
        cudaGraphicsResource *texture
    );

    CPU void Render(
        uint32_t *output,
        int width,
        int height,
        const CameraState &camera
    );

    CPU void Render(
        cudaGraphicsResource *texture,
        int width,
        int height,
        const CameraState &camera
    );
} // namespace CudaRaytracer

#endif // CUDA_RAYTRACER_H
