#define GPU_RUNNER 1
#include "renderer.h"

#include "lights/directional_light.h"
#include "lights/lights_list.h"

#include "hittables/hittables_list.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"
#include "hittables/plane.h"

#include <iostream>
#include <string>

#define WIDTH 720
#define HEIGHT 405

#define SAMPLES 1

void writePPM(const char* path, pixel* img, int width, int height);

__global__ void kernel(pixel* image, int width, int height, Camera camera, Hittable** world, Light** lights, Material* materials)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    
    // curandState_t randState;
    // curand_init(x + y * width, 0, 0, &randState);

    // -1 / 1
    float u = ((float)x / (float)width ) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    float pixelOffX = 0.5f / width;
    float pixelOffY = 0.5f / height;

    glm::vec3 result{ 0.0f, 0.0f, 0.0f };
    for(int i = 0; i < SAMPLES; ++i)
        result += glm::clamp(AntiAliasing(u, v, pixelOffX, pixelOffY, camera, world, lights, materials /*, &randState */), glm::vec3(0.0f), glm::vec3(1.0f));
    
    image[x + y * width].Set(result / glm::vec3(SAMPLES));
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

void gaussianBlur(pixel* img, int width, int height, float sigma, int size) {
    if (size % 2 == 0 || size < 3) {
        std::cerr << "La dimensione del kernel deve essere dispari e maggiore di 1." << std::endl;
        return;
    }

    float kernel[size][size];
    float sum = 0.0;

    //calcolo valori del kernel
    for (int x = -size / 2; x <= size / 2; x++) {
        for (int y = -size / 2; y <= size / 2; y++) {
            float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[x + size / 2][y + size / 2] = value;
            sum += value;
        }
    }

    //normalizzo il kernel
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }

    //applico il blur
    pixel* tempImg = (pixel*)malloc(width * height * sizeof(pixel));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sumX = 0.0, sumY = 0.0, sumZ = 0.0;

            for (int k = -size / 2; k <= size / 2; k++) {
                for (int l = -size / 2; l <= size / 2; l++) {
                    int x = min(max(j + k, 0), width - 1);
                    int y = min(max(i + l, 0), height - 1);

                    sumX += img[y * width + x].x * kernel[k + size / 2][l + size / 2];
                    sumY += img[y * width + x].y * kernel[k + size / 2][l + size / 2];
                    sumZ += img[y * width + x].z * kernel[k + size / 2][l + size / 2];
                }
            }

            // Clamping i valori tra 0 e 255
            tempImg[i * width + j].x = (unsigned char)(max(0.0f, min(255.0f, sumX)));
            tempImg[i * width + j].y = (unsigned char)(max(0.0f, min(255.0f, sumY)));
            tempImg[i * width + j].z = (unsigned char)(max(0.0f, min(255.0f, sumZ)));
        }
    }

    // Copiare l'immagine sfocata nell'array originale
    memcpy(img, tempImg, width * height * sizeof(pixel));
    free(tempImg);
}

int main(int argc, char **argv) 
{
    // Allocate Texture Memory
	int totalImageBytes = WIDTH * HEIGHT * sizeof(pixel);
	pixel* h_image = (pixel*) malloc(totalImageBytes);
    
	pixel* d_image;
	cudaMalloc(&d_image, totalImageBytes);

	
    // Setup
    Camera camera(60.0f, WIDTH, HEIGHT, 0.01f, 1000.0f);

    // Init Lights
    Light** l_lights;
    cudaMalloc((void**)&l_lights, 1 * sizeof(Light*));

    Light** d_lights;
    cudaMalloc((void**)&d_lights, sizeof(LightsList*));

    initLights<<<1, 1>>>(l_lights, d_lights);

    // Init World
    Hittable** l_world;
    cudaMalloc((void**)&l_world, 4 * sizeof(Hittable*));

    Hittable** d_world;
    cudaMalloc((void**)&d_world, sizeof(HittablesList*));

    initWorld<<<1, 1>>>(l_world, d_world);

    // Init Materials
    Material* d_materials;
    cudaMalloc((void**)&d_materials, 4 * sizeof(Material));

    {
        Material* materials = new Material[4];
        materials[0] = Material{ glm::vec3{ 0.8f, 0.8f, 0.0f }, 0.0f,  0.0f,  0.0f  };
        materials[1] = Material{ glm::vec3{ 0.8f, 0.2f, 0.1f }, 0.08f, 0.02f, 0.0f  };
        materials[2] = Material{ glm::vec3{ 0.8f, 0.8f, 0.8f }, 0.2f,  0.75f, 0.0f  };
        materials[3] = Material{ glm::vec3{ 0.0f, 0.0f, 0.0f }, 0.05f, 0.0f,  1.85f };

        cudaMemcpy(d_materials, materials, 4 * sizeof(Material), cudaMemcpyHostToDevice);
    }
    

    // Raytrace
	dim3 BlockSize(16, 16, 1);
	dim3 GridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);

    printf("%u %u %u - %u %u %u\n", BlockSize.x, BlockSize.y, BlockSize.z, GridSize.x, GridSize.y, GridSize.z);

	kernel<<<GridSize, BlockSize>>>(d_image, WIDTH, HEIGHT, camera, d_world, d_lights, d_materials);
	cudaMemcpy(h_image, d_image, totalImageBytes, cudaMemcpyDeviceToHost);
	
    //blurring
    // gaussianBlur(h_image, WIDTH, HEIGHT, 10.0f, 11);
    
    // Saving and closing
	writePPM("output.ppm", h_image, WIDTH, HEIGHT);

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

