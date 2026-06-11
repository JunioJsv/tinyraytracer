#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif
#define HD __host__ __device__

struct CameraState;
struct cudaGraphicsResource;

namespace CudaRaytracer
{
    struct alignas(16) Vec3
    {
        float x, y, z;

        HD float &operator[](
            const int i
        )
        {
            return i == 0 ? x : (1 == i ? y : z);
        }

        HD const float &operator[](
            const int i
        ) const
        {
            return i == 0 ? x : (1 == i ? y : z);
        }

        HD Vec3 operator*(
            const float v
        ) const
        {
            return {x * v, y * v, z * v};
        }

        HD float operator*(
            const Vec3 &v
        ) const
        {
            return x * v.x + y * v.y + z * v.z;
        }

        HD Vec3 operator+(
            const Vec3 &v
        ) const
        {
            return {x + v.x, y + v.y, z + v.z};
        }

        HD Vec3 operator-(
            const Vec3 &v
        ) const
        {
            return {x - v.x, y - v.y, z - v.z};
        }

        HD Vec3 operator-() const
        {
            return {-x, -y, -z};
        }

        HD Vec3 &operator+=(
            const Vec3 &vec3
        )
        {
            x += vec3.x;
            y += vec3.y;
            z += vec3.z;
            return *this;
        }

        HD Vec3 &operator-=(
            const Vec3 &vec3
        )
        {
            x -= vec3.x;
            y -= vec3.y;
            z -= vec3.z;
            return *this;
        }

        HD Vec3 operator/(
            const Vec3 &vec3
        ) const
        {
            return Vec3(
                x / vec3.x,
                y / vec3.y,
                z / vec3.z
            );
        }

        HD Vec3 operator/(
            const float scalar
        ) const
        {
            return Vec3(
                x / scalar,
                y / scalar,
                z / scalar
            );
        }

        HD float Length() const;

        HD Vec3 Normalized() const;

        HD Vec3 Cross(
            const Vec3 &v
        ) const;

        HD Vec3 Mul(
            const Vec3 &v
        ) const
        {
            return {x * v.x, y * v.y, z * v.z};
        }
    };

    enum class RayType
    {
        PRIMARY = 0,
        REFLECTION = 1,
        REFRACTION = 2
    };

    struct RayState
    {
        Vec3 orig;
        Vec3 dir;
        int depth;
        float weight;
        RayType rayType;
    };

    struct Background
    {
        static constexpr Vec3 DEFAULT_COLOR{0.2f, 0.7f, 0.8f};

        enum class Format
        {
            RGB8,
            RGB32F
        };

        HD unsigned int GetChannelSizeInBytes() const
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

        HD unsigned int GetDataSizeInBytes() const
        {
            return width * height * channels * GetChannelSizeInBytes();
        }

        HD bool IsHDR() const
        {
            return format == Format::RGB32F;
        }

        HD bool IsValid() const
        {
            return data && width > 0 && height > 0 && channels > 0;
        }

        HD unsigned int GetIndex(
            const Vec3 &dir
        ) const;

        HD Vec3 GetColor(
            const Vec3 &dir
        ) const
        {
            if (!IsValid()) return DEFAULT_COLOR;

            if (IsHDR()) {
                return GetHDRColor(dir);
            }

            return GetLDRColor(dir);
        }

        HD Vec3 GetLDRColor(
            const Vec3 &dir
        ) const;

        HD Vec3 GetHDRColor(
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

    HD Vec3 Reflect(
        const Vec3 &I,
        const Vec3 &N
    );

    HD Vec3 Refract(
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

    HD RayIntersection RaySphereIntersect(
        const Vec3 &orig,
        const Vec3 &dir,
        const Sphere &s
    );

    __host__ void Initialize();

    __host__ void Destroy();

    __host__ void SetBackground(
        const Background &background
    );

    __host__ cudaGraphicsResource *SetupCudaTexture(
        unsigned int glTextureId
    );

    __host__ void DestroyCudaTexture(
        cudaGraphicsResource *texture
    );

    __host__ void Render(
        uint32_t *output,
        int width,
        int height,
        const CameraState &camera
    );

    __host__ void Render(
        cudaGraphicsResource *texture,
        int width,
        int height,
        const CameraState &camera
    );
} // namespace CudaRaytracer

#endif // CUDA_RAYTRACER_H
