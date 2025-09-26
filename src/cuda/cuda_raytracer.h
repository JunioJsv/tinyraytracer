#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#include <cmath>
#include <algorithm>
#include <tuple>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif
#define HD __host__ __device__

namespace CudaRaytracer {
    struct vec3 {
        float x, y, z;

        HD float &operator[](int i);

        HD const float &operator[](int i) const;

        HD vec3 operator*(float v) const;

        HD float operator*(const vec3 &v) const;

        HD vec3 operator+(const vec3 &v) const;

        HD vec3 operator-(const vec3 &v) const;

        HD vec3 operator-() const;

        HD float norm() const;

        HD vec3 normalized() const;

        HD vec3 &operator+=(const vec3 &vec3) {
            x += vec3.x;
            y += vec3.y;
            z += vec3.z;
            return *this;
        }

        HD vec3 rotated(float pitch, float yaw) const;

        HD vec3 cross(vec3 v) const;
    };


    struct RayState {
        vec3 orig;
        vec3 dir;
        int depth;
        float weight;
        int ray_type; // 0 = raio principal, 1 = reflexão, 2 = refração
    };

    struct Material {
        float refractive_index;
        float albedo[4];
        vec3 diffuse_color;
        float specular_exponent;
    };

    struct Sphere {
        vec3 center;
        float radius;
        Material material;
    };

    HD vec3 reflect(const vec3 &I, const vec3 &N);

    HD vec3 refract(const vec3 &I, const vec3 &N, float eta_t, float eta_i);

    HD std::tuple<bool, float> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s);

    HD std::tuple<bool, vec3, vec3, Material> scene_intersect(const vec3 &orig, const vec3 &dir,
                                                              const Sphere *spheres_gpu);

    HD vec3 cast_ray_iterative(const vec3 &orig, const vec3 &dir,
                               const Sphere *spheres_gpu, const vec3 *lights_gpu, int max_depth);
}

using namespace CudaRaytracer;

extern "C" int render_cuda(uint32_t *image, int width, int height, const vec3 &position, float pitch, float yaw);

#endif // CUDA_RAYTRACER_H
