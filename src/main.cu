#define GPU_RUNNER 1
#include "renderer.h"

#include "lights/directionalLight.h"
#include "lights/lightsList.h"

#include "hittables/hittablesList.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"
#include "hittables/plane.h"

#include "utils/timer.h"

#include "camera.h"
#include "material.h"

#include <iostream>
#include <string>

#define WIDTH 1024
#define HEIGHT 512

#define SAMPLES 10

#define MAXDEPTH 20

#define CUDA(f) { cudaError_t err = f;\
    if(err != cudaSuccess)\
        printf("Cuda Error: %s\n", cudaGetErrorString(err)); }

#define CUDA_LASTERROR() { cudaError_t err = cudaGetLastError();\
    if(err != cudaSuccess)\
        printf("Cuda Error: %s\n", cudaGetErrorString(err)); }

void writePPM(const char* path, pixel* img,              int width, int height);
void writePPM(const char* path, emissionPixel* emission, int width, int height);

__global__ void kernel(pixel* image, emissionPixel* emission, int width, int height, Camera* camera, Hittable** world, Light** lights, Material* materials)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    
    curandState_t randState;
    curand_init(x + y * width, 0, 0, &randState);

    // -1 / 1
    float u = ((float)x / (float)width ) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    float pixelOffX = 0.5f / width;
    float pixelOffY = 0.5f / height;

    HitColorGlow result;
    for(int i = 0; i < SAMPLES; ++i)
    {
        HitColorGlow sample = AntiAliasing(u, v, pixelOffX, pixelOffY, camera, world, lights, materials, &randState, MAXDEPTH);
        result.color            += glm::clamp(sample.color,    glm::vec3(0.0f), glm::vec3(1.0f));
        result.emission         += glm::clamp(sample.emission, glm::vec3(0.0f), glm::vec3(1.0f));
        result.emissionStrenght += sample.emissionStrenght;
    }
    
    image   [x + y * width].Set(result.color    / glm::vec3(SAMPLES));
    emission[x + y * width].Set(result.emission / glm::vec3(SAMPLES), result.emissionStrenght / SAMPLES);
}

__global__ void initLights(Light** l_lights, Light** d_lights)
{
    if(threadIdx.x > 0 || threadIdx.y > 0)
        return;

    *(l_lights) = new DirectionalLight({ -0.25f, -0.75f, 0.45f  });
    *(d_lights) = new LightsList(l_lights, 1);
}

__global__ void initWorld(Hittable** l_world, Hittable** d_world)
{
    if(threadIdx.x > 0 || threadIdx.y > 0)
        return;
    
    *(l_world)     = new Sphere({  0.0f, -1000.0f, -4.0f }, 1000.0f, 0);
    *(l_world + 1) = new Sphere({  0.0f,  1.0f,    -4.0f }, 1.0f,    1);
    *(l_world + 2) = new Sphere({ -3.0f,  1.0f,    -4.0f }, 1.0f,    2);
    *(l_world + 3) = new Sphere({  3.0f,  1.0f,    -4.0f }, 1.0f,    3);
    // *(l_world + 2) = new Cube  ({ 2.0f,  2.0f, 2.0f }, { 0.5f, 0.5f, 0.5f }, 0);
    // *(l_world + 2) = new Plane ({ 0.0f, -4.5f, 5.0f }, { 0.0f,  -1.0f, 0.0f }, 2);
    *(d_world)     = new HittablesList(l_world, 4);
}

__global__ void cudaFreeList(void** list, void** device_list, int size)
{
    for(int i = 0; i < size; ++i)
        free(list[i]);

    free(device_list);
}

__global__ void downsample(emissionPixel* emission, emissionPixel* tempEmission, int width, int downScaleW, int downScaleH, int scaleFactor, const glm::vec3 percStepV)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= downScaleW || y >= downScaleH)
        return;
    
    int index = x + y * width;

    glm::vec3 color{ 0.0f, 0.0f, 0.0f };
    float strenght = 0.0f;

    int emissionsCount = 0;

    for (int nY = 0; nY < scaleFactor; ++nY)
        for (int nX = 0; nX < scaleFactor; ++nX)
        {
            emissionPixel& p = emission[x * scaleFactor + nX + (y * scaleFactor + nY) * width];
            color += p.emission;

            if(p.strenght > 0)
            {
                ++emissionsCount;
                strenght += p.strenght;
            }
        }
    
    if(emissionsCount > 0)
        strenght /= emissionsCount;
    
    tempEmission[index].Set(color * percStepV, strenght);
}

__global__ void upscale(emissionPixel* emission, emissionPixel* tempEmission, int width, int height, int scaleFactor)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    
    emission[x + y * width].Set(tempEmission[(int)(x / scaleFactor) + (int)(y / scaleFactor) * width]);
}

__global__ void addImages(pixel* image, emissionPixel* emission, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    
    int indx = x + y * width;
    
    emissionPixel& p = emission[indx];
	image[indx].Add(p.emission * glm::vec3(0.1f) * p.strenght);
}

__global__ void filterEmission(emissionPixel* emission, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    emissionPixel& p = emission[x + y * width];
    p.strenght *= 0.65f;

    if(p.strenght < 1)
        p.emission = glm::vec3{ 0.0f, 0.0f, 0.0f };
}

__global__ void createKernel(float* kernel, float sigma, int kernelSize)
{
    if(threadIdx.x > 0 || threadIdx.y > 0)
        return;
    
    float sum = 0.0f;

    // calcolo valori del kernel
    for(int i = -kernelSize; i <= kernelSize; ++i)
    {
        float value = exp(-(i * i) / (2 * sigma));
        kernel[i + kernelSize] = value;
        sum += value;
    }

    sum = 1.0f / sum;
    
    // normalizzo il kernel
    for(int i = 0; i < 2 * kernelSize + 1; ++i)
        kernel[i] *= sum;
}

__global__ void gaussianBlurH(emissionPixel* emission, emissionPixel* tempEmission, float* kernel, int width, int height, int downScaleW, int downScaleH, int kernelSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    extern __shared__ emissionPixel s_tile[];

    int s_x = threadIdx.x;
    int s_y = threadIdx.y;
    int s_width = blockDim.x + 2 * kernelSize;

    int indx = x + y * width;
    int s_indx = s_x + kernelSize + s_y * s_width;

    emissionPixel pixel;
    if(x < downScaleW && y < downScaleH)
        pixel = tempEmission[indx];
    
    s_tile[s_indx] = pixel;

    if (s_x < kernelSize)
    {
        // Handle left edge/border
        if (((x - kernelSize) >= 0) && ((x - kernelSize) < downScaleW * downScaleH))
            pixel = tempEmission[indx - kernelSize];
        else
            pixel.Set({ 0.0f, 0.0f, 0.0f }, 0.0f);
        
        s_tile[s_indx - kernelSize] = pixel;

        // Handle right edge/border
        if (((x + blockDim.x) < downScaleW) && (y < downScaleH))
            pixel = tempEmission[indx + blockDim.x];
        else
            pixel.Set({ 0.0f, 0.0f, 0.0f }, 0.0f);
        
        s_tile[s_indx + blockDim.x] = pixel;
    }
    __syncthreads();
    
    pixel.Set({ 0.0f, 0.0f, 0.0f }, 0.0f);

    int emissionsCount = 0;
    for(int i = -kernelSize; i <= kernelSize; ++i)
    {
        float k = kernel[i + kernelSize];
        emissionPixel& p = s_tile[i + s_indx];
        pixel.emission += p.emission * glm::vec3(k);

        if(p.strenght > 0)
        {
            ++emissionsCount;
            pixel.strenght += p.strenght;
        }
    }
    
    if(emissionsCount > 0)
        pixel.strenght /= emissionsCount;
    
    if(x < downScaleW && y < downScaleH)
    emission[indx].Set(pixel.emission, pixel.strenght);
}

__global__ void gaussianBlurV(emissionPixel* emission, emissionPixel* tempEmission, float* kernel, int width, int height, int downScaleW, int downScaleH, int kernelSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    extern __shared__ emissionPixel s_tile[];

    int s_x = threadIdx.x;
    int s_y = threadIdx.y;
    int s_width = blockDim.x;

    int indx = x + y * width;
    int s_indx = s_x + (s_y + kernelSize) * s_width;

    emissionPixel pixel;
    if(x < downScaleW && y < downScaleH)
        pixel = emission[indx];
    
    s_tile[s_indx] = pixel;

    if (s_y < kernelSize)
    {
        // Handle left edge/border
        if (((y - kernelSize) >= 0) && ((y - kernelSize) < downScaleW * downScaleH))
            pixel = emission[indx - kernelSize * width];
        else
            pixel.Set({ 0.0f, 0.0f, 0.0f }, 0.0f);
        
        s_tile[s_indx - kernelSize * s_width] = pixel;

        // Handle right edge/border
        if (((y + blockDim.y) < downScaleH) && (x < downScaleW))
            pixel = emission[indx + blockDim.y * width];
        else
            pixel.Set({ 0.0f, 0.0f, 0.0f }, 0.0f);
        
        s_tile[s_indx + blockDim.y * s_width] = pixel;
    }
    __syncthreads();
    
    pixel.Set({ 0.0f, 0.0f, 0.0f }, 0.0f);

    int emissionsCount = 0;
    for(int i = -kernelSize; i <= kernelSize; ++i)
    {
        float k = kernel[i + kernelSize];
        emissionPixel& p = s_tile[i * s_width + s_indx];
        pixel.emission += p.emission * glm::vec3(k);

        if(p.strenght > 0)
        {
            ++emissionsCount;
            pixel.strenght += p.strenght;
        }
    }
    
    if(emissionsCount > 0)
        pixel.strenght /= emissionsCount;
    
    if(x < downScaleW && y < downScaleH)
    tempEmission[indx].Set(pixel.emission, pixel.strenght);
}

/*
__global__ void gaussianBlur(emissionPixel* emission, emissionPixel* tempEmission, float* kernel, int width, int height, int downScaleW, int downScaleH, int size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= downScaleW || y >= downScaleH)
        return;
    
    emissionPixel pixel{ { 0.0f, 0.0f, 0.0f }, 0.0f };

    int emissionsCount = 0;

    int radius = size * 2 + 1;

    for (int kY = -size; kY <= size; ++kY)
        for (int kX = -size; kX <= size; ++kX)
        {
            int newX = glm::min(glm::max(x + kX, 0), width  - 1);
            int newY = glm::min(glm::max(y + kY, 0), height - 1);

            float k = kernel[(kX + size) + (kY + size) * radius];
            emissionPixel& p = tempEmission[newX + newY * width];
            pixel.emission += p.emission * glm::vec3(k);

            if(p.strenght > 0)
            {
                ++emissionsCount;
                pixel.strenght += p.strenght;
            }
        }
    
    if(emissionsCount > 0)
        pixel.strenght /= emissionsCount;
    
    emission[x + y * width].Set(pixel.emission, pixel.strenght);
}
*/

void applyGlow(pixel* image, emissionPixel* emission, int width, int height)
{
    cudaFuncAttributes attr;
    CUDA(cudaFuncGetAttributes(&attr, gaussianBlurV));

    int maxGlowSize = (int) std::sqrt(attr.maxThreadsPerBlock);


	int scaleFactor = 2, scale = scaleFactor;

	int kernelSigma = 20.0f, kernelSize = 8;

    dim3 BlockSize(maxGlowSize, maxGlowSize, 1);
	dim3 GridSize((WIDTH + BlockSize.x - 1) / BlockSize.x, (HEIGHT + BlockSize.y - 1) / BlockSize.y, 1);

    printf("Glow size: %d %d %d\n", BlockSize.x, BlockSize.y, BlockSize.z);
    
    // Allocate Temp Emission Texture Memory
	emissionPixel* d_tempEmission;
	CUDA(cudaMalloc((void**)&d_tempEmission, width * height * sizeof(emissionPixel)))

    float* d_kernel;
	CUDA(cudaMalloc((void**)&d_kernel, (2 * kernelSize + 1) * sizeof(float)))

    createKernel<<<1, 1>>>(d_kernel, kernelSigma, kernelSize);

    int totalEmissionBytes = width * height * sizeof(emissionPixel);
    emissionPixel* h_tempEmission = (emissionPixel*) malloc(totalEmissionBytes);

    // int totalBytes = width * height * sizeof(pixel);
    // pixel* h_tempImage = (pixel*) malloc(totalBytes);

    int downScaleW = width / scale, downScaleH = height / scale;

	while(downScaleW > 0 && downScaleH > 0)
	{
		// Downscale framebuffer
        glm::vec3 percStepV(1.0f / (scale * scale));

        dim3 DownGridSize((downScaleW + BlockSize.x - 1) / BlockSize.x, (downScaleH + BlockSize.y - 1) / BlockSize.y, 1);
		downsample<<<DownGridSize, BlockSize>>>(emission, d_tempEmission, width, downScaleW, downScaleH, scaleFactor, percStepV);
        // CUDA(cudaDeviceSynchronize())

        // CUDA(cudaMemcpy(h_tempEmission, d_tempEmission, totalEmissionBytes, cudaMemcpyDeviceToHost))
		// writePPM("output_downscale.ppm", h_tempEmission, width, height);

		// Blur downscaled image
        int s_size = (BlockSize.x + 2 * kernelSize) * BlockSize.y * sizeof(emissionPixel);
		gaussianBlurH<<<DownGridSize, BlockSize, s_size>>>(emission, d_tempEmission, d_kernel, width, height, downScaleW, downScaleH, kernelSize);
        // CUDA(cudaDeviceSynchronize())

        // CUDA(cudaMemcpy(h_tempEmission, emission, totalEmissionBytes, cudaMemcpyDeviceToHost))
		// writePPM("output_blur.ppm", h_tempEmission, width, height);

        s_size = BlockSize.x * (BlockSize.y + 2 * kernelSize) * sizeof(emissionPixel);
        gaussianBlurV<<<DownGridSize, BlockSize, s_size>>>(emission, d_tempEmission, d_kernel, width, height, downScaleW, downScaleH, kernelSize);
        // CUDA(cudaDeviceSynchronize())

        // CUDA(cudaMemcpy(h_tempEmission, d_tempEmission, totalEmissionBytes, cudaMemcpyDeviceToHost))
		// writePPM("output_blur.ppm", h_tempEmission, width, height);

		// Upscale blurred image
        upscale<<<GridSize, BlockSize>>>(emission, d_tempEmission, width, height, scale);
        // CUDA(cudaDeviceSynchronize())

        // CUDA(cudaMemcpy(h_tempEmission, d_tempEmission, totalEmissionBytes, cudaMemcpyDeviceToHost))
		// writePPM("output_upscale.ppm", h_tempEmission, width, height);

		// Add upscaled image with base image
        addImages<<<GridSize, BlockSize>>>(image, emission, width, height);
        // CUDA(cudaDeviceSynchronize())

        // CUDA(cudaMemcpy(h_tempImage, image, totalBytes, cudaMemcpyDeviceToHost))
		// writePPM("output_add.ppm", h_tempImage, width, height);

		// Filter downscaled image
        filterEmission<<<DownGridSize, BlockSize>>>(d_tempEmission, width, height);
        // CUDA(cudaDeviceSynchronize())

        // CUDA(cudaMemcpy(h_tempEmission, emission, totalEmissionBytes, cudaMemcpyDeviceToHost))
		// writePPM("output_filter.ppm", h_tempEmission, width, height);

		// Continue applying emission
        scale *= 2;

        downScaleW = width / scale;
        downScaleH = height / scale;

        emissionPixel* temp = d_tempEmission;
        d_tempEmission = emission;
        emission = temp;
	}

    cudaFree(d_kernel);
    cudaFree(d_tempEmission);
}

int main(int argc, char **argv) 
{
    int device;
    CUDA(cudaGetDevice(&device));

    cudaDeviceProp props;
    CUDA(cudaGetDeviceProperties(&props, device));

    cudaFuncAttributes attr;
    CUDA(cudaFuncGetAttributes(&attr, kernel));

    int maxKernelSize = (int) std::sqrt(attr.maxThreadsPerBlock);


    // Set max stack frame size for each thread
    cudaDeviceSetLimit(cudaLimitStackSize, 10240); // Max stress (131072) | Default (10240)

    // Allocate Texture Memory
	int totalImageBytes = WIDTH * HEIGHT * sizeof(pixel);
	pixel* d_image;
	CUDA(cudaMalloc((void**)&d_image, totalImageBytes))

    // Allocate Emission Texture Memory
    int totalEmissionImageBytes = WIDTH * HEIGHT * sizeof(emissionPixel);
	emissionPixel* d_emission;
	CUDA(cudaMalloc((void**)&d_emission, totalEmissionImageBytes))
    
    // Setup
    Camera* d_camera;
    {
        Camera* camera = new Camera(60.0f, WIDTH, HEIGHT, 0.01f, 1000.0f);
        CUDA(cudaMalloc((void**)&d_camera, sizeof(Camera)))

        CUDA(cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice))

        delete camera;
    }

    // Init Lights
    Light** l_lights;
    int lightsCount = 1;
    CUDA(cudaMalloc((void**)&l_lights, lightsCount * sizeof(Light*)))

    Light** d_lights;
    CUDA(cudaMalloc((void**)&d_lights, sizeof(LightsList*)))

    initLights<<<1, 1>>>(l_lights, d_lights);

    // Init World
    Hittable** l_world;
    int worldCount = 4;
    CUDA(cudaMalloc((void**)&l_world, worldCount * sizeof(Hittable*)))

    Hittable** d_world;
    CUDA(cudaMalloc((void**)&d_world, sizeof(HittablesList*)))

    initWorld<<<1, 1>>>(l_world, d_world);

    // Init Materials
    Material* d_materials;
    CUDA(cudaMalloc((void**)&d_materials, 4 * sizeof(Material)))

    {
        Material* materials = new Material[4];
        materials[0] = Material{ { 0.8f, 0.8f, 0.0f }, 0.0f,  0.0f,  0.0f , { 0.0f, 0.0f, 0.0f }, 0.0f };
        materials[1] = Material{ { 0.8f, 0.2f, 0.1f }, 0.08f, 0.02f, 0.0f , { 1.0f, 0.0f, 0.0f }, 4.5f };
        materials[2] = Material{ { 0.8f, 0.8f, 0.8f }, 0.9f,  0.75f, 0.0f , { 0.0f, 0.0f, 0.0f }, 0.0f };
        materials[3] = Material{ { 0.0f, 0.0f, 0.0f }, 0.0f,  0.0f,  1.85f, { 0.0f, 0.0f, 0.0f }, 0.0f };

        CUDA(cudaMemcpy(d_materials, materials, 4 * sizeof(Material), cudaMemcpyHostToDevice))

        delete materials;
    }
    
    // Raytrace
	dim3 BlockSize(maxKernelSize, maxKernelSize, 1);
	dim3 GridSize((WIDTH + BlockSize.x - 1) / BlockSize.x, (HEIGHT + BlockSize.y - 1) / BlockSize.y, 1);

    Timer t;

    printf("Kernel size: %d %d %d (%d %d %d)\n", GridSize.x, GridSize.y, GridSize.z, BlockSize.x, BlockSize.y, BlockSize.z);
	kernel<<<GridSize, BlockSize>>>(d_image, d_emission, WIDTH, HEIGHT, d_camera, d_world, d_lights, d_materials);

    CUDA_LASTERROR()

    CUDA(cudaDeviceSynchronize())

    printf("Ended in %lfms\n", t.ElapsedMillis());

    printf("Applying Glow\n");
    t.Reset();

    // TODO: Bloom
    applyGlow(d_image, d_emission, WIDTH, HEIGHT);

    printf("Ended in %lfms\n", t.ElapsedMillis());

    pixel* h_image = (pixel*) malloc(totalImageBytes);
    CUDA(cudaMemcpy(h_image, d_image, totalImageBytes, cudaMemcpyDeviceToHost))
    
    // Saving and closing
	writePPM("output.ppm", h_image, WIDTH, HEIGHT);

    // Free
    cudaFreeList<<<1, 1>>>((void**)l_lights, (void**)d_lights, lightsCount);
    cudaFreeList<<<1, 1>>>((void**)l_world,  (void**)d_world,  worldCount);

    cudaFree(d_materials);
    cudaFree(d_camera);

    cudaFree(d_emission);
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

void writePPM(const char* path, emissionPixel* emission, int width, int height)
{
	FILE* file = fopen(path, "wb");
	
	if (!file)
	{
		fprintf(stderr, "Failed to open file\n");
		return;
	}
	
	fprintf(file, "P6\n%d %d\n255\n", width, height);

	pixel* img = (pixel*) malloc(width * height * sizeof(pixel));
	int len = width * height;
	for(int i = 0; i < len; ++i)
		img[i].Set(emission[i].emission);
	
	fwrite(img, sizeof(pixel), width * height, file);
	free(img);
	
	fclose(file);
}
