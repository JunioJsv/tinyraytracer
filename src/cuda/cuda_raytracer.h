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

        HD float Length() const;

        HD Vec3 Normalized() const;

        HD Vec3 Cross(
            Vec3 v
        ) const;
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
        using data_t = uint32_t;

        data_t *data;
        int width;
        int height;

        HD Vec3 GetColor(
            const Vec3 &dir
        ) const;
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
