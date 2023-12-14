#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>

#define CHANNELS 3

void writePPM(const char* path, unsigned char* img, int width, int height);

struct vec3
{
    float x, y, z;
};

struct hit
{
	vec3 position, normal, color;
	float distance;
};

__host__ __device__ inline vec3 add(const vec3& v, const vec3& v1)
{
	return vec3{ v.x + v1.x, v.y + v1.y, v.z + v1.z };
}

__host__ __device__ inline vec3 div(const vec3& v, float n)
{
	n = 1.0f / n;
	return vec3{ v.x * n, v.y * n, v.z * n };
}

__host__ __device__ inline vec3 sub(const vec3& v, const vec3& v1)
{
	return vec3{ v.x - v1.x, v.y - v1.y, v.z - v1.z };
}

__host__ __device__ inline vec3 mul(const vec3& v, float n)
{
	return vec3{ v.x * n, v.y * n, v.z * n };
}

__host__ __device__ inline float magnitude (const vec3& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
__host__ __device__ inline vec3  normalize (const vec3& v) { return div(v, magnitude(v)); }

__host__ __device__ inline float dot(const vec3& v, const vec3& v1) { return v.x * v1.x + v.y * v1.y + v.z * v1.z; }

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

__device__ hit color(const vec3& dir)
{
    vec3 rayOrigin = { 0.0f, 0.0f, 0.0f };
    vec3 cubeMin = { 0.4f, 0.4f, 0.4f };
    vec3 cubeMax = { 1.0f, -1.0f, 1.0f };

    vec3 hitPoint, normal;
    
    if (intersectCube(rayOrigin, dir, hitPoint, normal, cubeMin, cubeMax)) {
        vec3 color = mul(add(normal, vec3{1.0f, 1.0f, 1.0f}), 0.5f);

        hit h;
        h.distance = magnitude(sub(hitPoint, rayOrigin));
        h.position = hitPoint;
        h.normal = normal;
        h.color = color;
        return h;
    }
    else {
        hit h;
        h.distance = -1;
        h.normal = { 0.0f, 0.0f, 0.0f };
        h.position = { 0.0f, 0.0f, 0.0f };
        h.color = { 0.5f, 0.5f, 0.5f };
        return h;
    }
}

__global__ void kernel(unsigned char* image, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int index = 3 * (x + (height - y - 1) * width);
    float u = float(x) / width;
    float v = float(y) / height;

    vec3 lower_left_corner = {-0.5f, -1.0f, -0.25f};
    vec3 horizontal = {1.0f, 0.25f, 0.25f};
    vec3 vertical = {0.5f, 1.0f, 0.5f};

    vec3 dir = normalize(add(lower_left_corner, add(mul(horizontal, u), mul(vertical, v))));

    hit h = color(dir);

    if (h.distance >= 0.0f) {
        image[index]     = (unsigned char)(h.color.x * 255.0f);
        image[index + 1] = (unsigned char)(h.color.y * 255.0f);
        image[index + 2] = (unsigned char)(h.color.z * 255.0f);
    } else {
        image[index]     = 0;
        image[index + 1] = 0;
        image[index + 2] = 0;
    }
}

int main(int argc, char **argv) 
{
	int width = 1920, height = 1080;
	
	int total_pixels = width * height;
	unsigned char* h_image = (unsigned char *) malloc(sizeof(unsigned char*) * total_pixels * CHANNELS);
	
	unsigned char* d_image;
	cudaMalloc(&d_image, total_pixels * CHANNELS);
	
	dim3 BlockSize(16, 16, 1);
	dim3 GridSize((width + 15) / 16, (height + 15) / 16, 1);
    	
	kernel<<<GridSize, BlockSize>>>(d_image, width, height);
	cudaMemcpy(h_image, d_image, total_pixels * CHANNELS, cudaMemcpyDeviceToHost);
	
	writePPM("output.ppm", h_image, width, height);
	cudaFree(d_image);
	free(h_image);
	return 0;
}

void writePPM(const char* path, unsigned char* img, int width, int height)
{
	FILE* file = fopen(path, "wb");
	
	if (!file)
	{
		fprintf(stderr, "Failed to open file\n");
		return;
	}
	
	fprintf(file, "P6\n%d %d\n255\n", width, height);
	
	fwrite(img, sizeof(unsigned char), width * height * CHANNELS, file);
	
	fclose(file);
}
