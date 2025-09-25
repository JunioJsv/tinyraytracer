#ifndef TINYRAYTRACER_RAYTRACER_H
#define TINYRAYTRACER_RAYTRACER_H

#include <tuple>
#include <fstream>

namespace Raytracer {
    struct vec3 {
        float x = 0, y = 0, z = 0;

        float &operator[](int i);

        const float &operator[](int i) const;

        vec3 operator*(float v) const;

        float operator*(const vec3 &v) const;

        vec3 operator+(const vec3 &v) const;

        vec3 operator-(const vec3 &v) const;

        vec3 operator-() const;

        float norm() const;

        vec3 normalized() const;
    };

    vec3 cross(vec3 v1, vec3 v2);

    struct Material {
        float refractive_index = 1;
        float albedo[4] = {2, 0, 0, 0};
        vec3 diffuse_color = {0, 0, 0};
        float specular_exponent = 0;
    };

    struct Sphere {
        vec3 center;
        float radius;
        Material material;
    };

    vec3 reflect(const vec3 &I, const vec3 &N);

    vec3 refract(const vec3 &I, const vec3 &N, float eta_t, float eta_i = 1.f);

    std::tuple<bool, float> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s);

    std::tuple<bool, vec3, vec3, Material> scene_intersect(const vec3 &orig, const vec3 &dir);

    vec3 cast_ray(const vec3 &orig, const vec3 &dir, int depth = 0);
}

#endif //TINYRAYTRACER_RAYTRACER_H
