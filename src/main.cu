#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <vector>

#include "structs.h"

#include "camera.h"
#include "ray.h"

#include "hittables/hittable_list.h"
#include "hittables/hittable.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/glm.hpp"

inline void registerToWorld(std::vector<Hittable*>& world, uint32_t& worldSize, Hittable& obj, size_t objSize);

void writePPM(const char* path, pixel* img, int width, int height);

__global__ void kernel(pixel* image, int width, int height, glm::vec3 camPos, glm::mat4 invPerspective, glm::mat4 invView, Hittable** world)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    
    // -1 / 1
    float u = ((float)x / (float)width ) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    glm::vec4 target = invPerspective * glm::vec4(u, v, 1.0f, 1.0f);
    glm::vec3 rayDirection = glm::vec3(invView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f)); // World space

    Ray ray(camPos, rayDirection);

    glm::vec3 result(0.0f);
    float multiplier = 1.0f;

	int bounces = 2;
	for (int i = 0; i < bounces; ++i)
	{
        IntersectInfo info;
        if(!(*world)->intersect(ray, info))
        {
            glm::vec3 skyColor{ 0.52f, 0.80f, 0.92f };
            result += skyColor * multiplier;
            break;
        }

        glm::vec3 lightDir = glm::normalize(glm::vec3{ -0.75f, 0.45f, -0.5f });
        float lightIntensity = max(glm::dot(info.hit.normal, -lightDir), 0.0f);
        result += info.hit.color * lightIntensity * multiplier;

        multiplier *= 0.45f;

		ray.Origin = info.hit.position + info.hit.normal * 0.0001f;
        ray.Direction = glm::reflect(ray.Direction, info.hit.normal + info.Roughness);
		//ray.Direction = glm::reflect(ray.Direction,
		//	payload.WorldNormal + material.Roughness * Walnut::Random::Vec3(-0.5f, 0.5f));
        //
		// ray.SetDirection(glm::normalize(payload.normal + glm::normalize(glm::vec3{ -1.0f, 1.0f, 0.0f })));
    }

    result = glm::clamp(result, glm::vec3(0.0f), glm::vec3(1.0f));

    image[x + y * width].Set(result);
}

__global__ void createWorld(Hittable** d_list, Hittable** d_world)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    *(d_list)     = new Sphere(glm::vec3{ 0.0f,  0.5f, 3.5f }, 0.5f, glm::vec3{ 1.0f, 0.0f, 1.0f }); //0);
    *(d_list + 1) = new Sphere(glm::vec3{ 0.0f, -9.0f, 3.5f }, 9.0f, glm::vec3{ 0.2f, 0.3f, 0.9f }); //0);
    *(d_list + 2) = new Cube(glm::vec3{ 2.0f,  2.0f, 2.0f }, glm::vec3{ 0.5f, 0.5f, 0.5f }, glm::vec3{ 0.8f, 0.6f, 0.2f });
    *d_world      = new HittableList(d_list, 2);
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


    // Create World
    Hittable** d_list;
    cudaMalloc((void**)&d_list, 3 * sizeof(Hittable*));

    Hittable** d_world;
    cudaMalloc((void**)&d_world, sizeof(Hittable*));

    createWorld<<<1, 1>>>(d_list, d_world);
    cudaDeviceSynchronize();


    // Raytrace
	dim3 BlockSize(16, 16, 1);
	dim3 GridSize((width + 15) / 16, (height + 15) / 16, 1);
    
	kernel<<<GridSize, BlockSize>>>(d_image, width, height, camera.GetPosition(), camera.GetInverseProjectionMatrix(), camera.GetInverseViewMatrix(), d_world);
	cudaMemcpy(h_image, d_image, totalImageBytes, cudaMemcpyDeviceToHost);


    // Saving and closing
	writePPM("output.ppm", h_image, width, height);
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


/*
interfaccia oggetto
{
    bool intersect(Hit* h, float* d);
}

sfera : oggetto
{
    bool intersect(Hit* h, float* d)
    {
        return bla bla;
    }
}

color
{
    hit, dist;
    for(oggetto nel mondo)
    {
        Hit h;
        float d
        if(!oggetto.intersect(&h, &d))
            return;
        
        if(d < dist)
        {
            hit = h;
            dist = d;
        }
    }

    return hit;
}
*/
