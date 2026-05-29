#include "cuda_raytracer.h"

#include <iostream>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

namespace CudaRaytracer 
{
	constexpr Material IVORY = { 1.0, {0.9, 0.5, 0.1, 0.0}, {0.4, 0.4, 0.3}, 50. };
	constexpr Material GLASS = { 1.5, {0.0, 0.9, 0.1, 0.8}, {0.6, 0.7, 0.8}, 125. };
	constexpr Material RED_RUBBER = { 1.0, {1.4, 0.3, 0.0, 0.0}, {0.3, 0.1, 0.1}, 10. };
	constexpr Material MIRROR = { 1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425. };

	constexpr int N_SPHERES = 4;
	constexpr Sphere CPU_SPHERES[] = {
		{{-3, 0, -16}, 2, IVORY},
		{{-1.0, -1.5, -12}, 2, GLASS},
		{{1.5, -0.5, -18}, 3, RED_RUBBER},
		{{7, 5, -18}, 4, MIRROR}
	};

	constexpr int N_LIGHTS = 3;
	constexpr Vec3 CPU_LIGHTS[] = {
		{-20, 20, 20},
		{30, 50, -25},
		{30, 20, 30}
	};

	__device__ __constant__ Sphere GPU_SPHERES[N_SPHERES];
	__device__ __constant__ Vec3 GPU_LIGHTS[N_LIGHTS];

	float Vec3::Norm() const
	{
		return sqrtf(x * x + y * y + z * z);
	}

	Vec3 Vec3::Normalized() const
	{
		return (*this) * (1.f / Norm());
	}

	Vec3 Vec3::Rotated(
		const float pitch,
		const float yaw
	) const
	{
		float cosPitch = cosf(pitch);
		float sinPitch = sinf(pitch);
		Vec3 vPitch = {
			x,
			y * cosPitch - z * sinPitch,
			y * sinPitch + z * cosPitch
		};

		// yaw
		float cosYaw = cosf(yaw);
		float sinYaw = sinf(yaw);
		return {
			vPitch.x * cosYaw + vPitch.z * sinYaw,
			vPitch.y,
			-vPitch.x * sinYaw + vPitch.z * cosYaw
		};
	}

	Vec3 Vec3::Cross(
		const Vec3 v
	) const
	{
		return { y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x };
	}

	Vec3 Reflect(
		const Vec3& I,
		const Vec3& N
	)
	{
		return I - N * 2.f * (I * N);
	}

	Vec3 Refract(
		const Vec3& I,
		const Vec3& N,
		const float etaT,
		const float etaI
	)
	{
		const float cosi = -fmaxf(-1.f, fminf(1.f, I * N));
		if (cosi < 0) return Refract(I, -N, etaI, etaT);
		const float eta = etaI / etaT;
		const float k = 1 - eta * eta * (1 - cosi * cosi);
		return k < 0 ? Vec3{ 1, 0, 0 } : I * eta + N * (eta * cosi - sqrtf(k));
	}

	std::tuple<bool, float> RaySphereIntersect(
		const Vec3& orig,
		const Vec3& dir,
		const Sphere& s
	)
	{
		const Vec3 L = s.center - orig;
		const float tca = L * dir;
		const float d2 = L * L - tca * tca;
		if (d2 > s.radius * s.radius) return { false, 0 };
		const float thc = sqrtf(s.radius * s.radius - d2);
		const float t0 = tca - thc, t1 = tca + thc;
		if (t0 > .001f) return { true, t0 };
		if (t1 > .001f) return { true, t1 };
		return { false, 0 };
	}

	static __device__ std::tuple<bool, Vec3, Vec3, Material> SceneIntersect(
		const Vec3& orig,
		const Vec3& dir
	)
	{
		Vec3 pt, N;
		Material material;

		float nearestDist = 1e10f;

		if (fabsf(dir.y) > .001f) {
			float d = -(orig.y + 4.f) / dir.y;
			Vec3 p = orig + dir * d;
			if (d > .001f && d < nearestDist && fabsf(p.x) < 10.f && p.z < -10.f && p.z > -30.f) {
				nearestDist = d;
				pt = p;
				N = { 0, 1, 0 };
				material.diffuseColor = (int(.5f * pt.x + 1000.f) + (int)(.5f * pt.z)) & 1
					? Vec3{ .3f, .3f, .3f }
				: Vec3{ .3f, .2f, .1f };
			}

			material.refractiveIdx = 1.0f;
			material.albedo[0] = 1.0f;
			material.albedo[1] = 0.0f;
			material.albedo[2] = 0.0f;
			material.albedo[3] = 0.0f;
			material.specularExponent = 0.0f;
		}

		// Intersecção com as esferas
		for (int i = 0; i < N_SPHERES; ++i) {
			auto [intersection, d] = RaySphereIntersect(orig, dir, GPU_SPHERES[i]);
			if (!intersection || d > nearestDist) continue;
			nearestDist = d;
			pt = orig + dir * nearestDist;
			N = (pt - GPU_SPHERES[i].center).Normalized();
			material = GPU_SPHERES[i].material;
		}

		return { nearestDist < 1000.f, pt, N, material };
	}

	static __device__ Vec3 CastRay(
		const Vec3& orig,
		const Vec3& dir,
		const int maxDepth
	)
	{
		constexpr int maxStackSize = 64;
		RayState stack[maxStackSize]{};
		int stackIdx = 0;

		Vec3 finalColor = { 0, 0, 0 };

		stack[stackIdx++] = RayState{ orig, dir, 0, 1.0f, RayType::PRIMARY };

		while (stackIdx > 0) {
			RayState current = stack[--stackIdx];

			if (current.depth > maxDepth) {
				continue;
			}

			auto [hit, point, N, material] = SceneIntersect(current.orig, current.dir);

			if (!hit) {
				Vec3 backgroundColor = { 0.2f, 0.7f, 0.8f };
				finalColor = finalColor + backgroundColor * current.weight;
				continue;
			}

			float diffuseLightIntensity = 0, specularLightIntensity = 0;

			for (int i = 0; i < N_LIGHTS; ++i) {
				const Vec3& light = GPU_LIGHTS[i];
				Vec3 lightDir = (light - point).Normalized();

				auto [hit_shadow, shadow_pt, trashnrm, trashmat] = SceneIntersect(point, lightDir);
				if (hit_shadow && (shadow_pt - point).Norm() < (light - point).Norm())
					continue;

				diffuseLightIntensity += fmaxf(0.f, lightDir * N);
				specularLightIntensity += powf(fmaxf(0.f, -Reflect(-lightDir, N) * current.dir),
					material.specularExponent);
			}

			Vec3 localColor = material.diffuseColor * diffuseLightIntensity * material.albedo[0] +
				Vec3{ 1., 1., 1. } *specularLightIntensity * material.albedo[1];

			finalColor = finalColor + localColor * current.weight;

			if (current.depth < maxDepth && stackIdx < (maxStackSize - 2)) {
				constexpr float offset = 1e-3f;
				if (material.albedo[2] > 0.01f) {
					Vec3 reflectDir = Reflect(current.dir, N).Normalized();
					Vec3 reflectOrig = point + N * offset;
					float reflectWeight = current.weight * material.albedo[2];

					stack[stackIdx++] = RayState{
						reflectOrig, reflectDir, current.depth + 1,
						reflectWeight, RayType::REFLECTION
					};
				}

				if (material.albedo[3] > 0.01f) {
					Vec3 refractDir = Refract(current.dir, N, material.refractiveIdx, 1.0f);

					if (refractDir.Norm() > 0.1f) {
						refractDir = refractDir.Normalized();

						float dotProduct = current.dir * N;
						Vec3 refractOrig;
						if (dotProduct < 0) {
							refractOrig = point - N * offset;
						}
						else {
							// Raio saindo do material
							refractOrig = point + N * offset;
						}

						float refractWeight = current.weight * material.albedo[3];

						stack[stackIdx++] = RayState{
							refractOrig, refractDir, current.depth + 1,
							refractWeight, RayType::REFRACTION
						};
					}
				}
			}
		}

		return finalColor;
	}

	static __global__ void CastRayKernal(
		uint32_t* pixels,
		int width,
		int height,
		const Vec3 position,
		float pitch,
		float yaw
	)
	{
		int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (pixelIdx >= width * height) return;

		int i = pixelIdx % width;
		int j = pixelIdx / width;

		float fov = 1.05;

		float dirX = (i + 0.5f) - width / 2.f;
		float dirY = -(j + 0.5f) + height / 2.f;
		float dirZ = -height / (2.f * tanf(fov / 2.f));

		Vec3 dir = Vec3{ dirX, dirY, dirZ }.Normalized();
		dir = dir.Rotated(pitch, yaw);

		auto c = CastRay(position, dir, 4);
		auto r = static_cast<unsigned char>(fminf(fmaxf(c.x * 255.0f, 0.0f), 255.0f));
		auto g = static_cast<unsigned char>(fminf(fmaxf(c.y * 255.0f, 0.0f), 255.0f));
		auto b = static_cast<unsigned char>(fminf(fmaxf(c.z * 255.0f, 0.0f), 255.0f));
		uint32_t color = (255 << 24) | b << 16 | g << 8 | r;
		pixels[pixelIdx] = color;
	}

	int Render(
		uint32_t* pixels,
		int width,
		int height,
		const Vec3& position,
		float pitch,
		float yaw
	)
	{
		uint32_t* gpuPixels = nullptr;
		const size_t gpuPixelsSize = width * height * sizeof(uint32_t);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuPixels), gpuPixelsSize));

		CUDA_CHECK(cudaMemcpyToSymbol(GPU_SPHERES, CPU_SPHERES, N_SPHERES * sizeof(Sphere)));
		CUDA_CHECK(cudaMemcpyToSymbol(GPU_LIGHTS, CPU_LIGHTS, N_LIGHTS * sizeof(Vec3)));

		constexpr int BLOCK_SIZE = 16;
		const int nBlocks = (width * height + BLOCK_SIZE - 1) / BLOCK_SIZE;
		CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

		CastRayKernal << <nBlocks, BLOCK_SIZE >> > (gpuPixels, width, height, position, pitch, yaw);

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(pixels, gpuPixels, gpuPixelsSize, cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(gpuPixels));

		return 0;
	}
}
