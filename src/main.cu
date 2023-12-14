#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>

#include "camera.h"
#include "ray.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/glm.hpp"

struct pixel
{
    unsigned char x, y, z;

    __host__ __device__ void Set(const glm::vec3& c)
    {
        x = (unsigned char)(c.x * 255.0f);
        y = (unsigned char)(c.y * 255.0f);
        z = (unsigned char)(c.z * 255.0f);
    }
};

void writePPM(const char* path, pixel* img, int width, int height);

struct RayHit
{
	glm::vec3 position = { 0.0f, 0.0f, 0.0f };
    glm::vec3 normal   = { 0.0f, 0.0f, 0.0f };
    glm::vec3 color    = { 0.0f, 0.0f, 0.0f };
    float distance = -1;
};

struct intersectInfo
{
    RayHit hit;
    uint32_t objectIndx = 0;
};

/*
__device__ vec3 calculateCubeNormal(const vec3& hitPoint, const vec3& cubeMin, const vec3& cubeMax, float bias = 1e-4f) {
    vec3 normal = {0, 0, 0};
    if (fabs(hitPoint.x - cubeMin.x) < bias) normal.x = -1;
    else if (fabs(hitPoint.x - cubeMax.x) < bias) normal.x = 1;
    else if (fabs(hitPoint.y - cubeMin.y) < bias) normal.y = -1;
    else if (fabs(hitPoint.y - cubeMax.y) < bias) normal.y = 1;
    else if (fabs(hitPoint.z - cubeMin.z) < bias) normal.z = -1;
    else if (fabs(hitPoint.z - cubeMax.z) < bias) normal.z = 1;

    return normal;
}

__device__ bool intersectCube(const vec3& rayOrigin, const vec3& rayDirection, vec3& hitPoint, vec3& normal, const vec3& cubeMin, const vec3& cubeMax) {
    float tMin = -INFINITY;
    float tMax = INFINITY;

    for (int i = 0; i < 3; i++) {
        float originComponent = (i == 0) ? rayOrigin.x : ((i == 1) ? rayOrigin.y : rayOrigin.z);
        float directionComponent = (i == 0) ? rayDirection.x : ((i == 1) ? rayDirection.y : rayDirection.z);
        float cubeMinComponent = (i == 0) ? cubeMin.x : ((i == 1) ? cubeMin.y : cubeMin.z);
        float cubeMaxComponent = (i == 0) ? cubeMax.x : ((i == 1) ? cubeMax.y : cubeMax.z);

        if (fabs(directionComponent) < 1e-8) {
            // se non c'è intersezione
            if (originComponent < cubeMinComponent || originComponent > cubeMaxComponent) return false;
        } else {
            float ood = 1.0f / directionComponent;
            float t1 = (cubeMinComponent - originComponent) * ood;
            float t2 = (cubeMaxComponent - originComponent) * ood;
            if (t1 > t2) {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }
            tMin = max(tMin, t1);
            tMax = min(tMax, t2);
            if (tMin > tMax) return false;
        }
    }

    // Il raggio hitta
    hitPoint = add(rayOrigin, mul(rayDirection, tMin));
    normal = calculateCubeNormal(hitPoint, cubeMin, cubeMax);

    return true;
}
*/

__device__ bool intersectSphere(const Ray& ray, glm::vec3& color, glm::vec3& hitPoint, float& distance, glm::vec3& normal, const glm::vec3& sphereOrigin, const float sphereRadius)
{
    glm::vec3 origin = ray.Origin - sphereOrigin;

	float a = glm::dot(ray.Direction, ray.Direction);
	float b = 2.0f * glm::dot(origin, ray.Direction);
	float c = glm::dot(origin, origin) - sphereRadius * sphereRadius;
	
	float discriminant = b * b - 4.0f * a * c;
	
	if (discriminant < 0.0f)
		return false;

	distance = (-b - sqrt(discriminant)) / (2.0f * a);
	hitPoint = origin + ray.Direction * distance;
	normal = glm::normalize(hitPoint);
    hitPoint += sphereOrigin;
	return true;
}

__device__ RayHit TraceRay(const Ray& ray)
{
    // vec3 cubeMin = { 0.4f, 0.4f, 0.4f };
    // vec3 cubeMax = { 1.0f, -1.0f, 1.0f };

    glm::vec3 s_origin{ 0.0f, 0.0f, -5.0f };
    float radius = 1.0f;

    glm::vec3 color, hitPoint, normal;
    float distance;

    RayHit hit;
    if (intersectSphere(ray, color, hitPoint, distance, normal, s_origin, radius))
    {
        hit.distance = distance;
        hit.position = hitPoint;
        hit.normal   = normal;
        hit.color    = { 1.0f, 0.0f, 1.0f };
        return hit;
    }

    return hit;
}

__global__ void kernel(pixel* image, int width, int height, glm::vec3 camPos, glm::mat4 invPerspective, glm::mat4 invView)
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

	int bounces = 3;
	for (int i = 0; i < bounces; ++i)
	{
        RayHit hit = TraceRay(ray);
        if(hit.distance < 0.0f)
        {
            glm::vec3 skyColor{ 0.0f, 0.0f, 0.0f };
            result += skyColor * multiplier;
            break;
        }

        glm::vec3 lightDir = glm::normalize(glm::vec3{ -0.1f, 0.15f, 0.0f });
        float lightIntensity = max(glm::dot(hit.normal, -lightDir), 0.0f);
        result += hit.color * lightIntensity * multiplier;

        multiplier *= 0.7f;

		ray.Origin = hit.position + hit.normal * 0.0001f;
        ray.Direction = glm::reflect(rayDirection, hit.normal);
		//ray.Direction = glm::reflect(ray.Direction,
		//	payload.WorldNormal + material.Roughness * Walnut::Random::Vec3(-0.5f, 0.5f));
        //
		// ray.SetDirection(glm::normalize(payload.normal + glm::normalize(glm::vec3{ -1.0f, 1.0f, 0.0f })));
    }

    result = glm::clamp(result, glm::vec3(0.0f), glm::vec3(1.0f));

    image[x + y * width].Set(result);
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
	
    // Settup
    Camera camera(60.0f, width, height, 0.01f, 1000.0f);
    
    // Raytrace
	dim3 BlockSize(16, 16, 1);
	dim3 GridSize((width + 15) / 16, (height + 15) / 16, 1);
    
	kernel<<<GridSize, BlockSize>>>(d_image, width, height, camera.GetPosition(), camera.GetInverseProjectionMatrix(), camera.GetInverseViewMatrix());
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
