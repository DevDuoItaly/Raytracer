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

__device__ hit color(const vec3& dir)
{
	vec3 r_origin{ 0.0f, 0.0f, 0.0f };
	
	vec3 s_origin{ 0.0f, 0.0f, 1.0f };
	float radius = 0.5f;
	
	float a = dot(dir, dir);
	float b = 2.0f * dot(s_origin, dir);
	float c = dot(s_origin, s_origin) - radius * radius;
	
	float discriminant = b * b - 4.0f * a * c;
	
	if (discriminant < 0.0f)
	{
		hit h;
		h.distance = -1;
		h.normal   = { 0.0f, 0.0f, 0.0f };
		h.position = { 0.0f, 0.0f, 0.0f };
		h.color    = { 0.0f, 0.0f, 0.0f };
		return h;
	}
	else
	{
		hit h;
		h.distance = (-b - sqrt(discriminant)) / (2.0f * a);
		h.color    = { 1.0f, 0.0f, 0.0f };
		
		vec3 pos = add(sub(r_origin, s_origin), mul(dir, h.distance));
		h.normal = normalize(pos);
		h.position = add(pos, s_origin);
		return h;
	}
}

__global__ void kernel(unsigned char* image, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	float u = float(x) / width;
	float v = float(y) / height;
	if(x >= width || y >= height)
		return;
	
	int index = 3 * (x + (height - y) * width);
	vec3 zero_corner{ -2.0f, -1.0f, -1.0f };
	vec3 horizontal {  4.0f,  0.0f,  0.0f };
	vec3 vertical   {  0.0f,  2.0f,  0.0f };
	
	hit h = color(add(zero_corner, add(mul(horizontal, u), mul(vertical, v))));
	if(h.distance < 0)
	{
		image[index    ] = 0.0f;
		image[index + 1] = 0.0f;
		image[index + 2] = 0.0f;
		return;
	}
	
	image[index    ] = (unsigned char) ((h.normal.x + 1.0f) * 0.5f * 255.0f);
	image[index + 1] = (unsigned char) ((h.normal.y + 1.0f) * 0.5f * 255.0f);
	image[index + 2] = (unsigned char) ((h.normal.z + 1.0f) * 0.5f * 255.0f);
}

int main(int argc, char **argv) 
{
	int width = 720, height = 405;
	
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
