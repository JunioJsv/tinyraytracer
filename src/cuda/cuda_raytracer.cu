#include "cuda_raytracer.h"
#include <iostream>
#include <stack>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

using CudaRaytracer::Sphere;
using CudaRaytracer::vec3;

constexpr Material ivory = {1.0, {0.9, 0.5, 0.1, 0.0}, {0.4, 0.4, 0.3}, 50.};
constexpr Material glass = {1.5, {0.0, 0.9, 0.1, 0.8}, {0.6, 0.7, 0.8}, 125.};
constexpr Material red_rubber = {1.0, {1.4, 0.3, 0.0, 0.0}, {0.3, 0.1, 0.1}, 10.};
constexpr Material mirror = {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.};

constexpr int N_SPHERES = 4;
constexpr Sphere spheres_cpu[] = {
    {{-3, 0, -16}, 2, ivory},
    {{-1.0, -1.5, -12}, 2, glass},
    {{1.5, -0.5, -18}, 3, red_rubber},
    {{7, 5, -18}, 4, mirror}
};

constexpr int N_LIGHTS = 3;
constexpr vec3 lights_cpu[] = {
    {-20, 20, 20},
    {30, 50, -25},
    {30, 20, 30}
};

__constant__ Sphere spheres_gpu[N_SPHERES];
__constant__ vec3 lights_gpu[N_LIGHTS];

namespace CudaRaytracer {
    float &vec3::operator[](const int i) { return i == 0 ? x : (1 == i ? y : z); }

    const float &vec3::operator[](const int i) const { return i == 0 ? x : (1 == i ? y : z); }

    vec3 vec3::operator*(const float v) const { return {x * v, y * v, z * v}; }

    float vec3::operator*(const vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

    vec3 vec3::operator+(const vec3 &v) const { return {x + v.x, y + v.y, z + v.z}; }

    vec3 vec3::operator-(const vec3 &v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    vec3 vec3::operator-() const {
        return {-x, -y, -z};
    }

    float vec3::norm() const {
        return sqrtf(x * x + y * y + z * z);
    }

    vec3 vec3::normalized() const {
        return (*this) * (1.f / norm());
    }

    vec3 vec3::rotated(const float pitch, const float yaw) const {
        float cos_pitch = cosf(pitch);
        float sin_pitch = sinf(pitch);
        vec3 v_pitch = {
            x,
            y * cos_pitch - z * sin_pitch,
            y * sin_pitch + z * cos_pitch
        };

        // yaw
        float cos_yaw = cosf(yaw);
        float sin_yaw = sinf(yaw);
        return {
            v_pitch.x * cos_yaw + v_pitch.z * sin_yaw,
            v_pitch.y,
            -v_pitch.x * sin_yaw + v_pitch.z * cos_yaw
        };
    }

    vec3 vec3::cross(const vec3 v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }

    HD vec3 reflect(const vec3 &I, const vec3 &N) {
        return I - N * 2.f * (I * N);
    }

    HD vec3 refract(const vec3 &I, const vec3 &N, const float eta_t, const float eta_i) {
        float cosi = -fmaxf(-1.f, fminf(1.f, I * N));
        if (cosi < 0) return refract(I, -N, eta_i, eta_t);
        float eta = eta_i / eta_t;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? vec3{1, 0, 0} : I * eta + N * (eta * cosi - sqrtf(k));
    }

    HD std::tuple<bool, float> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s) {
        vec3 L = s.center - orig;
        float tca = L * dir;
        float d2 = L * L - tca * tca;
        if (d2 > s.radius * s.radius) return {false, 0};
        float thc = sqrtf(s.radius * s.radius - d2);
        float t0 = tca - thc, t1 = tca + thc;
        if (t0 > .001f) return {true, t0};
        if (t1 > .001f) return {true, t1};
        return {false, 0};
    }

    HD std::tuple<bool, vec3, vec3, Material> scene_intersect(const vec3 &orig, const vec3 &dir,
                                                              const Sphere *spheres_gpu) {
        vec3 pt, N;
        Material material;

        float nearest_dist = 1e10f;

        if (fabsf(dir.y) > .001f) {
            float d = -(orig.y + 4.f) / dir.y;
            vec3 p = orig + dir * d;
            if (d > .001f && d < nearest_dist && fabsf(p.x) < 10.f && p.z < -10.f && p.z > -30.f) {
                nearest_dist = d;
                pt = p;
                N = {0, 1, 0};
                material.diffuse_color = (int(.5f * pt.x + 1000.f) + (int) (.5f * pt.z)) & 1
                                             ? vec3{.3f, .3f, .3f}
                                             : vec3{.3f, .2f, .1f};
            }

            material.refractive_index = 1.0f;
            material.albedo[0] = 1.0f;
            material.albedo[1] = 0.0f;
            material.albedo[2] = 0.0f;
            material.albedo[3] = 0.0f;
            material.specular_exponent = 0.0f;
        }

        // Intersecção com as esferas
        for (int i = 0; i < N_SPHERES; ++i) {
            auto [intersection, d] = ray_sphere_intersect(orig, dir, spheres_gpu[i]);
            if (!intersection || d > nearest_dist) continue;
            nearest_dist = d;
            pt = orig + dir * nearest_dist;
            N = (pt - spheres_gpu[i].center).normalized();
            material = spheres_gpu[i].material;
        }

        return {nearest_dist < 1000.f, pt, N, material};
    }

    HD vec3 cast_ray_iterative(const vec3 &orig, const vec3 &dir,
                               const Sphere *spheres_gpu, const vec3 *lights_gpu, const int max_depth) {
        constexpr int max_stack_size = 64;
        RayState stack[max_stack_size];
        int stack_size = 0;

        vec3 final_color = {0, 0, 0};

        stack[stack_size++] = RayState{orig, dir, 0, 1.0f, 0};

        while (stack_size > 0) {
            RayState current = stack[--stack_size];

            if (current.depth > max_depth) {
                continue;
            }

            auto [hit, point, N, material] = scene_intersect(current.orig, current.dir, spheres_gpu);

            if (!hit) {
                vec3 background_color = {0.2f, 0.7f, 0.8f};
                final_color = final_color + background_color * current.weight;
                continue;
            }

            float diffuse_light_intensity = 0, specular_light_intensity = 0;

            for (int i = 0; i < N_LIGHTS; ++i) {
                const vec3 &light = lights_gpu[i];
                vec3 light_dir = (light - point).normalized();

                auto [hit_shadow, shadow_pt, trashnrm, trashmat] = scene_intersect(point, light_dir, spheres_gpu);
                if (hit_shadow && (shadow_pt - point).norm() < (light - point).norm())
                    continue;

                diffuse_light_intensity += fmaxf(0.f, light_dir * N);
                specular_light_intensity += powf(fmaxf(0.f, -reflect(-light_dir, N) * current.dir),
                                                 material.specular_exponent);
            }

            vec3 local_color = material.diffuse_color * diffuse_light_intensity * material.albedo[0] +
                               vec3{1., 1., 1.} * specular_light_intensity * material.albedo[1];

            final_color = final_color + local_color * current.weight;

            if (current.depth < max_depth && stack_size < (max_stack_size - 2)) {
                constexpr float offset = 1e-3f;
                if (material.albedo[2] > 0.01f) {
                    vec3 reflect_dir = reflect(current.dir, N).normalized();
                    vec3 reflect_orig = point + N * offset;
                    float reflect_weight = current.weight * material.albedo[2];

                    stack[stack_size++] = RayState{
                        reflect_orig, reflect_dir, current.depth + 1,
                        reflect_weight, 1
                    };
                }

                if (material.albedo[3] > 0.01f) {
                    vec3 refract_dir = refract(current.dir, N, material.refractive_index, 1.0f);

                    if (refract_dir.norm() > 0.1f) {
                        refract_dir = refract_dir.normalized();

                        float dot_product = current.dir * N;
                        vec3 refract_orig;
                        if (dot_product < 0) {
                            refract_orig = point - N * offset;
                        } else {
                            // Raio saindo do material
                            refract_orig = point + N * offset;
                        }

                        float refract_weight = current.weight * material.albedo[3];

                        stack[stack_size++] = RayState{
                            refract_orig, refract_dir, current.depth + 1,
                            refract_weight, 2
                        };
                    }
                }
            }
        }

        return final_color;
    }
}

__global__ void cast_ray_kernel(uint32_t *image, int width, int height, const vec3 position, float pitch, float yaw) {
    using namespace CudaRaytracer;

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    int i = pixel_idx % width;
    int j = pixel_idx / width;

    float fov = 1.05;

    float dir_x = (i + 0.5f) - width / 2.f;
    float dir_y = -(j + 0.5f) + height / 2.f;
    float dir_z = -height / (2.f * tanf(fov / 2.f));

    vec3 dir = vec3{dir_x, dir_y, dir_z}.normalized();
    dir = dir.rotated(pitch, yaw);

    auto c = cast_ray_iterative(position, dir, spheres_gpu, lights_gpu, 4);
    auto r = static_cast<unsigned char>(fminf(fmaxf(c.x * 255.0f, 0.0f), 255.0f));
    auto g = static_cast<unsigned char>(fminf(fmaxf(c.y * 255.0f, 0.0f), 255.0f));
    auto b = static_cast<unsigned char>(fminf(fmaxf(c.z * 255.0f, 0.0f), 255.0f));
    uint32_t color = (255 << 24) | r << 16 | g << 8 | b;
    image[pixel_idx] = color;
}

extern "C" int render_cuda(uint32_t *image, int width, int height, const vec3 &position, float pitch, float yaw) {
    using namespace CudaRaytracer;

    uint32_t *image_gpu;
    size_t image_size = width * height * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&image_gpu), image_size));

    CUDA_CHECK(cudaMemcpyToSymbol(spheres_gpu, spheres_cpu, N_SPHERES * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(lights_gpu, lights_cpu, N_LIGHTS * sizeof(vec3)));

    constexpr int BLOCK_SIZE = 16;
    int num_blocks = (width * height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

    cast_ray_kernel << <num_blocks, BLOCK_SIZE >> >(image_gpu, width, height, position, pitch, yaw);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(image, image_gpu, image_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(image_gpu));

    return 0;
}
