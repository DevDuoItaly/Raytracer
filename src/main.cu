#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>

#include "camera.h"
#include "structs.h"

#include "lights/directional_light.h"
#include "lights/lights_list.h"
#include "lights/light.h"

#include "hittables/hittables_list.h"
#include "hittables/hittable.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"

#include "material.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/glm.hpp"

#define MAX_DEPTH 2

void writePPM(const char* path, pixel* img, int width, int height);

__device__ inline void UVToDirection(float u, float v, const glm::mat4& invProj, const glm::mat4& invView, glm::vec3& direction)
{
    glm::vec4 target = invProj * glm::vec4(u, v, 1.0f, 1.0f); // Clip Space
    direction = glm::vec3(invView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f)); // World space
}

__device__ glm::vec3 TraceRay(Ray ray, Hittable** world, Light** lights, Material* materials, float multiplier, int depth)
{
    RayHit hit;
    if(!(*world)->intersect(ray, hit))
        return glm::vec3{ 0.52f, 0.80f, 0.92f } * multiplier;

    float intensity = 0.0f;
    (*lights)->IsInLight(hit.position, hit.normal, intensity);

    const Material& material = materials[hit.materialIndx];
    glm::vec3 color = material.color * intensity * multiplier;

    if(depth < MAX_DEPTH)
    {
        if(material.refraction <= 0)
        {
            // Reflection
            ray.origin = hit.position + hit.normal * 0.0001f;
            ray.direction = glm::reflect(ray.direction, hit.normal + material.roughness);
        }
        else
        {
            // Refraction
            // ray.direction = glm::refraction(ray.direction, hit.normal, material.refraction);
        }

        color += TraceRay(ray, world, lights, materials, multiplier * 0.35f, depth + 1);
    }

    return color;
}

__device__ glm::vec3 AntiAliasing(float u, float v, float pixelOffX, float pixelOffY, const Camera& camera, Hittable** world, Light** lights, Material* materials)
{
    const glm::mat4& invProj = camera.GetInverseProjectionMatrix();
    const glm::mat4& invView = camera.GetInverseViewMatrix();

    glm::vec3 color{ 0.0f, 0.0f, 0.0f };
    Ray ray{ camera.GetPosition() , glm::vec3{ 0.0f, 0.0f, 0.0f } };

    UVToDirection(u - pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    UVToDirection(u + pixelOffX, v - pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    UVToDirection(u - pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    UVToDirection(u + pixelOffX, v + pixelOffY, invProj, invView, ray.direction);
    color += TraceRay(ray, world, lights, materials, 1, 0);

    color *= glm::vec3{ 0.25f, 0.25f, 0.25f };
    return color;
}

__global__ void kernel(pixel* image, int width, int height, Camera camera, Hittable** world, Light** lights, Material* materials)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    
    // -1 / 1
    float u = ((float)x / (float)width ) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    float pixelOffX = 0.5f / width;
    float pixelOffY = 0.5f / height;
    glm::vec3 result = AntiAliasing(u, v, pixelOffX, pixelOffY, camera, world, lights, materials);
    
    result = glm::clamp(result, glm::vec3(0.0f), glm::vec3(1.0f));

    image[x + y * width].Set(result);
}

__global__ void initLights(Light** l_lights, Light** d_lights)
{
    if(threadIdx.x > 0 || threadIdx.y > 0)
        return;

    *(l_lights) = new DirectionalLight({ -0.75f, 1.0f, 0.5f });
    *(d_lights) = new LightsList(l_lights, 1);
}

__global__ void initWorld(Hittable** l_world, Hittable** d_world)
{
    if(threadIdx.x > 0 || threadIdx.y > 0)
        return;

    *(l_world)     = new Sphere({ 0.0f,  0.5f, 5.0f }, 0.5f, 0);
    *(l_world + 1) = new Sphere({ 0.0f, -5.0f, 5.0f }, 5.0f, 1);
    *(l_world + 2) = new Cube  ({ 2.0f,  2.0f, 2.0f }, { 0.5f, 0.5f, 0.5f }, 0);
    *(d_world)     = new HittablesList(l_world, 3);
}

__global__ void cudaFreeList(void** list, void** device_list, int size)
{
    for(int i = 0; i < size; ++i)
        free(list[i]);

    free(device_list);
}

int main(int argc, char **argv) 
{
    // Screen Infos
	int width = 1920, height = 1080;

    // Allocate Texture Memory
	int totalImageBytes = width * height * sizeof(pixel);
	pixel* h_image = (pixel*) malloc(totalImageBytes);
    
	pixel* d_image;
	cudaMalloc(&d_image, totalImageBytes);

	
    // Setup
    Camera camera(60.0f, width, height, 0.01f, 1000.0f);

    // Init Lights
    Light** l_lights;
    cudaMalloc((void**)&l_lights, 1 * sizeof(Light*));

    Light** d_lights;
    cudaMalloc((void**)&d_lights, sizeof(LightsList*));

    initLights<<<1, 1>>>(l_lights, d_lights);

    // Init World
    Hittable** l_world;
    cudaMalloc((void**)&l_world, 3 * sizeof(Hittable*));

    Hittable** d_world;
    cudaMalloc((void**)&d_world, sizeof(HittablesList*));

    initWorld<<<1, 1>>>(l_world, d_world);

    // Init Materials
    Material* d_materials;
    cudaMalloc((void**)&d_materials, 2 * sizeof(Material));

    {
        Material materials[] =
        {
            Material{ glm::vec3{ 1.0f, 0.0f, 1.0f }, 0.0f, 0.0f },
            Material{ glm::vec3{ 0.2f, 0.3f, 0.8f }, 0.0f , 0.0f }
        };

        cudaMemcpy(d_materials, materials, 2 * sizeof(Material), cudaMemcpyHostToDevice);
    }
    

    // Raytrace
	dim3 BlockSize(16, 16, 1);
	dim3 GridSize((width + 15) / 16, (height + 15) / 16, 1);

	kernel<<<GridSize, BlockSize>>>(d_image, width, height, camera, d_world, d_lights, d_materials);
	cudaMemcpy(h_image, d_image, totalImageBytes, cudaMemcpyDeviceToHost);
	

    // Saving and closing
	writePPM("output.ppm", h_image, width, height);

    // Free
    cudaFreeList<<<1, 1>>>((void**)l_lights, (void**)d_lights, 1);
    cudaFreeList<<<1, 1>>>((void**)l_world,  (void**)d_world,  2);

    cudaFree(d_materials);

	cudaFree(d_image);
	free(h_image);
	return 0;
}

void writePPM(const char* path, pixel* img, int width, int height)
{
	FILE* file = fopen(path, "wb");
	
	if (!file)
	{
		fprintf(stderr, "Failed to open file\n");
		return;
	}
	
	fprintf(file, "P6\n%d %d\n255\n", width, height);
	
	fwrite(img, sizeof(pixel), width * height, file);
	
	fclose(file);
}
