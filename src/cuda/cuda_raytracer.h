#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#include <tuple>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif
#define HD __host__ __device__

namespace CudaRaytracer
{
    struct Vec3
    {
        float x, y, z;

        HD float &Vec3::operator[](
            const int i
        )
        {
            return i == 0 ? x : (1 == i ? y : z);
        }

        HD const float &Vec3::operator[](
            const int i
        ) const
        {
            return i == 0 ? x : (1 == i ? y : z);
        }

        HD Vec3 Vec3::operator*(
            const float v
        ) const
        {
            return {x * v, y * v, z * v};
        }

        HD float Vec3::operator*(
            const Vec3 &v
        ) const
        {
            return x * v.x + y * v.y + z * v.z;
        }

        HD Vec3 Vec3::operator+(
            const Vec3 &v
        ) const
        {
            return {x + v.x, y + v.y, z + v.z};
        }

        HD Vec3 Vec3::operator-(
            const Vec3 &v
        ) const
        {
            return {x - v.x, y - v.y, z - v.z};
        }

        HD Vec3 Vec3::operator-() const
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

        HD float Norm() const;

        HD Vec3 Normalized() const;

        HD Vec3 Rotated(
            float pitch,
            float yaw
        ) const;

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

    HD std::tuple<bool, float> RaySphereIntersect(
        const Vec3 &orig,
        const Vec3 &dir,
        const Sphere &s
    );

    __host__ int Render(
        uint32_t *output,
        int width,
        int height,
        const Vec3 &position,
        float pitch,
        float yaw
    );
} // namespace CudaRaytracer

#endif // CUDA_RAYTRACER_H
