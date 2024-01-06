#include "renderer.h"

#include "lights/directional_light.h"
#include "lights/lights_list.h"

#include "hittables/hittables_list.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"
#include "hittables/plane.h"

#include "redis.h"

#include "utils/timer.h"
#include "utils/thread_pool.h"

#include <vector>
#include <algorithm>
#include <execution>

#define WIDTH 1024
#define HEIGHT 512

#define SAMPLES 1

#define MAXBLUR 5

void writePPM(const char* path, pixel* img,              int width, int height);
void writePPM(const char* path, emissionPixel* emission, int width, int height);

std::vector<uint32_t> imageHorizontalIter;
std::vector<uint32_t> imageVerticalIter;

emissionPixel* downsample(emissionPixel* emission, int width, int height, int scaleFactor)
{
	int startWidth = width;

	width  /= scaleFactor;
	height /= scaleFactor;

	emissionPixel* result = new emissionPixel[width * height];
	
	glm::vec3 percStepV(1.0f / (scaleFactor * scaleFactor));
	
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			int index = x + y * width;

			glm::vec3 color{ 0.0f, 0.0f, 0.0f };
			float strenght = 0.0f;

			int emissionsCount = 0;

			for (int nY = 0; nY < scaleFactor; ++nY)
				for (int nX = 0; nX < scaleFactor; ++nX)
				{
					emissionPixel& p = emission[x * scaleFactor + nX + (y * scaleFactor + nY) * startWidth];
					color += p.emission;

					if(p.strenght > 0)
					{
						++emissionsCount;
						strenght += p.strenght;
					}
				}
			
			if(emissionsCount > 0)
				strenght /= emissionsCount;
			
			result[index].Set(color * percStepV, strenght);
		}

	return result;
}

emissionPixel* upscale(emissionPixel* emission, int width, int height, int scaleFactor)
{
	int scaledW = width * scaleFactor;
	emissionPixel* result = new emissionPixel[scaledW * height * scaleFactor];

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			emissionPixel& p = emission[x + y * width];

			for (int nY = 0; nY < scaleFactor; ++nY)
				for (int nX = 0; nX < scaleFactor; ++nX)
				{
					result[(x * scaleFactor + nX) + ((y * scaleFactor + nY) * scaledW)].Set(p.emission, p.strenght);
				}
		}
	}

	return result;
}

void gaussianBlur(emissionPixel* emission, int width, int height, float sigma, int size) 
{
    if (size < 1)
	{
        std::cerr << "La dimensione del kernel deve essere dispari e maggiore di 1." << std::endl;
        return;
    }

    float kernel[size * 2 + 1][size * 2 + 1];
	{
		float sum = 0.0f;

		// calcolo valori del kernel
		for (int y = -size; y <= size; ++y)
			for (int x = -size; x <= size; ++x)
			{
				float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
				kernel[x + size][y + size] = value;
				sum += value;
			}
		
		sum = 1.0f / sum;
		
		// normalizzo il kernel
		for (int y = 0; y < size * 2 + 1; ++y)
			for (int x = 0; x < size * 2 + 1; ++x)
				kernel[x][y] *= sum;
	}

	// Copia l'immagine originale
    emissionPixel* tempEmission = (emissionPixel*) malloc(width * height * sizeof(emissionPixel));
    memcpy(tempEmission, emission, width * height * sizeof(emissionPixel));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
		{
			emissionPixel pixel{ { 0.0f, 0.0f, 0.0f }, 0.0f };

			int emissionsCount = 0;

			for (int kX = -size; kX <= size; kX++)
				for (int kY = -size; kY <= size; kY++)
				{
					int newX = glm::min(glm::max(x + kX, 0), width  - 1);
					int newY = glm::min(glm::max(y + kY, 0), height - 1);

					float k = kernel[kX + size][kY + size];
					emissionPixel& p = emission[newX + newY * width];
					pixel.emission += p.emission * glm::vec3(k);

					if(p.strenght > 0)
					{
						++emissionsCount;
						pixel.strenght += p.strenght;
					}
				}
			
			if(emissionsCount > 0)
				pixel.strenght /= emissionsCount;
			
			tempEmission[x + y * width].Set(pixel.emission, pixel.strenght);
        }

    // Copiare i pixel sfocati nell'immagine originale
    memcpy(emission, tempEmission, width * height * sizeof(emissionPixel));
    free(tempEmission);
}

void gaussianBlur(pixel* img, emissionPixel* glowMap, int width, int height, float sigma, int size) 
{
    if (size % 2 == 0 || size < 3) 
	{
        std::cerr << "La dimensione del kernel deve essere dispari e maggiore di 1." << std::endl;
        return;
    }

    float kernel[size][size];
    float sum = 0.0;

    // calcolo valori del kernel
    for (int x = -size / 2; x <= size / 2; x++) 
	{
        for (int y = -size / 2; y <= size / 2; y++) 
		{
            float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[x + size / 2][y + size / 2] = value;
            sum += value;
        }
    }

    // normalizzo il kernel
    for (int i = 0; i < size; i++) 
	{
        for (int j = 0; j < size; j++) 
		{
            kernel[i][j] /= sum;
        }
    }

    // applico il blur solo ai pixel con glow
    pixel* tempImg = (pixel*)malloc(width * height * sizeof(pixel));
    memcpy(tempImg, img, width * height * sizeof(pixel));  // Copia l'immagine originale per preservare i pixel senza glow

    for (int i = 0; i < height; i++) 
	{
        for (int j = 0; j < width; j++) 
		{
            if (glowMap[i * width + j].emission.x > 0 || glowMap[i * width + j].emission.y > 0 || glowMap[i * width + j].emission.z > 0) 
			{
                float sumX = 0.0, sumY = 0.0, sumZ = 0.0;

                for (int k = -size / 2; k <= size / 2; k++) 
				{
                    for (int l = -size / 2; l <= size / 2; l++) 
					{
                        int x = glm::min(glm::max(j + k, 0), width - 1);
                        int y = glm::min(glm::max(i + l, 0), height - 1);

                        sumX += img[y * width + x].x * kernel[k + size / 2][l + size / 2];
                        sumY += img[y * width + x].y * kernel[k + size / 2][l + size / 2];
                        sumZ += img[y * width + x].z * kernel[k + size / 2][l + size / 2];
                    }
                }

                // Clamping i valori tra 0 e 255
                tempImg[i * width + j].x = (unsigned char)(glm::max(0.0f, glm::min(255.0f, sumX)));
                tempImg[i * width + j].y = (unsigned char)(glm::max(0.0f, glm::min(255.0f, sumY)));
                tempImg[i * width + j].z = (unsigned char)(glm::max(0.0f, glm::min(255.0f, sumZ)));
            }
        }
    }

    // Copiare i pixel sfocati nell'immagine originale
    memcpy(img, tempImg, width * height * sizeof(pixel));
    free(tempImg);
}

void applyGlow(pixel* image, emissionPixel* emission, int width, int height)
{
	float max = 1.0f;
	int downScaleFactor = 2, upScaleFactor = downScaleFactor;

	int kernelSigma = 1000.0f, kernelSize = 8;

	int startW = width, startH = height;
	// pixel* tmpImg = (pixel*) malloc(startW * startH * sizeof(pixel));

	while(max >= 1 && width > 0 && height > 0)
	{
		// Downscale framebuffer
		emissionPixel* downscaled = downsample(emission, width, height, downScaleFactor);
		free(emission);

		int downScaleW = width / downScaleFactor, downScaleH = height / downScaleFactor;
		// writePPM("output_downscale.ppm", downscaled, downScaleW, downScaleH);

		// Blur downscaled image
		gaussianBlur(downscaled, downScaleW, downScaleH, kernelSigma, kernelSize);

		// writePPM("output_blur.ppm", downscaled, downScaleW, downScaleH);
		
		// Upscale blurred image
		emissionPixel* upscaled = upscale(downscaled, downScaleW, downScaleH, upScaleFactor);

		// writePPM("output_upscale.ppm", upscaled, startW, startH);

		// Add upscaled image with base image
		for(int y = 0; y < startH; ++y)
			for(int x = 0; x < startW; ++x)
			{
				emissionPixel& p = upscaled[x + y * startW];
				image[x + y * startW].Add(p.emission * glm::vec3(0.1f) * p.strenght);
			}
		
		free(upscaled);

		// writePPM("output_add.ppm", image, startW, startH);

		// Filter downscaled image
		max = 0.0f;
		for(int y = 0; y < downScaleH; ++y)
			for(int x = 0; x < downScaleW; ++x)
			{
				emissionPixel& p = downscaled[x + y * downScaleW];

				p.strenght *= 0.65f;
				if(p.strenght < 1)
					p.emission = glm::vec3{ 0.0f, 0.0f, 0.0f };
				
				max = std::max(max, p.strenght);
			}
		
		// writePPM("output_filter.ppm", downscaled, downScaleW, downScaleH);

		// Continue applying emission
		width /= downScaleFactor; height /= downScaleFactor;
		emission = downscaled;

		upScaleFactor *= 2;
		kernelSize *= 2;
	}
}

int main()
{
	// Setup framebuffer
    pixel* image = new pixel[WIDTH * HEIGHT];

	emissionPixel* emission = new emissionPixel[WIDTH * HEIGHT];

    imageHorizontalIter.resize(WIDTH);
	imageVerticalIter.resize(HEIGHT);
	for (uint32_t i = 0; i < WIDTH; i++)
		imageHorizontalIter[i] = i;
	for (uint32_t i = 0; i < HEIGHT; i++)
		imageVerticalIter[i] = i;


	// Setup world
	Camera camera(60.0f, WIDTH, HEIGHT, 0.01f, 1000.0f);

    // -- Init Lights
	uint32_t light_count = 1;
	Light** l_light = (Light**) malloc(light_count * sizeof(Light*));
	l_light[0] = new DirectionalLight({ -0.25f, -0.75f, 0.45f  });
	//l_light[1] = new DirectionalLight({  0.85f, -0.15f, 0.15f });
	
    Light* lights = new LightsList(l_light, light_count);

    // -- Init World
	// curandState* rnd = new curandState(-1);

	int spawnW = 7;

	uint32_t world_count = 49 + 4;
	Hittable** l_world = (Hittable**) malloc(world_count * sizeof(Hittable*));
	l_world[0] = new Sphere({  0.0f, -1000.0f, 0.0f }, 1000.0f, 0);
	int i = 0;
	for(; i < world_count - 4; ++i)
	{
		l_world[i + 1] = new Sphere({ (((int)i % spawnW) - 3) * 1.5f, 0.3f, ((int)(i / spawnW) - 3) * 1.5f }, 0.3f, (int)(0 /*RANDOM_UNIT(rnd)*/ * 3) + 4);
	}
	++i;

	l_world[i++] = new Sphere({  0.0f, 1.0f, 0.0f }, 1.0f, 1);
	l_world[i++] = new Sphere({ -3.0f, 1.0f, 0.0f }, 1.0f, 2);
	l_world[i++] = new Sphere({  3.0f, 1.0f, 0.0f }, 1.0f, 3);

	// l_world[1] = new Sphere({  0.0f,  0.0f,   -4.0f }, 0.5f,   1);
	// l_world[2] = new Sphere({ -1.5f,  0.5f,   -4.0f }, 1.0f,   2);
	// l_world[3] = new Sphere({  1.5f,  0.5f,   -4.0f }, 1.0f,   3);
	// l_world[4] = new Sphere({  1.5f,  1.0f,   -4.0f }, 0.5f,   4);
	// l_world[4] = new Plane ({  0.0f,  -5.0f, 5.0f }, { 0.0f,  -1.0f, 0.0f }, 3);
	
	Hittable* world = new HittablesList(l_world, world_count);

    // -- Init Materials
    Material* materials = new Material[7];
	materials[0] = Material{ glm::vec3{ 0.8f, 0.8f, 0.0f }, 0.0f,  0.0f,  0.0f,  { 0.0f, 0.0f, 0.0f }, 0.0f };
	materials[1] = Material{ glm::vec3{ 0.0f, 0.0f, 0.0f }, 0.05f, 0.0f,  1.85f, { 0.0f, 0.0f, 0.0f }, 0.0f };
	materials[2] = Material{ glm::vec3{ 0.8f, 0.8f, 0.8f }, 0.2f,  0.75f, 0.0f,  { 0.0f, 0.0f, 0.0f }, 0.0f };
	materials[3] = Material{ glm::vec3{ 0.8f, 0.2f, 0.1f }, 0.05f, 0.0f,  0.0f,  { 0.7f, 0.1f, 0.2f }, 4.5f };
	materials[4] = Material{ glm::vec3{ 0.1f, 0.7f, 0.2f }, 0.08f, 0.02f, 0.0f,  { 0.0f, 0.0f, 0.0f }, 0.0f };
	materials[5] = Material{ glm::vec3{ 0.1f, 0.2f, 0.7f }, 0.08f, 0.02f, 0.0f,  { 0.0f, 0.0f, 0.0f }, 0.0f };
	materials[6] = Material{ glm::vec3{ 0.1f, 0.2f, 0.7f }, 0.1f,  0.05f, 0.0f,  { 0.0f, 0.0f, 0.0f }, 0.0f };
	// materials[1] = Material{ glm::vec3{ 0.7f, 0.3f, 0.3f }, 0.9f,  0.08f, 0.0f  };
	// materials[2] = Material{ glm::vec3{ 0.8f, 0.8f, 0.8f }, 0.3f,  0.25f, 0.0f  };
	// materials[3] = Material{ glm::vec3{ 0.0f, 0.0f, 0.0f }, 0.05f, 0.0f,  1.85f };
	// materials[4] = Material{ glm::vec3{ 0.1f, 0.8f, 0.2f }, 0.1f,  0.09f, 0.0f  };

	// Raytrace
	bool d = false;
	ThreadPool pool(d ? 1 : (std::thread::hardware_concurrency() - 1));

	float totalTasks;

	printf("Running Raytracing MT...\n");
	Timer t;

	int prevElapsed = -1;

	for(const uint32_t& y : imageVerticalIter)
	{
		for(const uint32_t& x : imageHorizontalIter)
		{
			pool.enqueue([&image, &emission, &camera, &world, &lights, &materials, x, y]()
			{
				// -1 / 1
				float u = ((float)x / (float)WIDTH ) * 2.0f - 1.0f;
				float v = ((float)y / (float)HEIGHT) * 2.0f - 1.0f;

				// curandState randState(x + y * WIDTH);

				float pixelOffX = 0.5f / WIDTH;
				float pixelOffY = 0.5f / HEIGHT;

				HitColorGlow result;
				for(int i = 0; i < SAMPLES; ++i)
				{
					HitColorGlow sample = AntiAliasing(u, v, pixelOffX, pixelOffY, &camera, &world, &lights, materials /*, &randState */);
					result.color            += glm::clamp(sample.color,    glm::vec3(0.0f), glm::vec3(1.0f));
					result.emission         += glm::clamp(sample.emission, glm::vec3(0.0f), glm::vec3(1.0f));
					result.emissionStrenght += sample.emissionStrenght;
				}

				image   [x + y * WIDTH].Set(result.color    / glm::vec3(SAMPLES));
				emission[x + y * WIDTH].Set(result.emission / glm::vec3(SAMPLES), result.emissionStrenght / SAMPLES);
			});
		}

		totalTasks = y * WIDTH;

		int elapsed = (int)(t.ElapsedMillis() * 0.001f);
		if(elapsed != prevElapsed)
		{
			prevElapsed = elapsed;

			int invRemain = totalTasks - pool.GetTasksCount();
			printf("Progress: %f%%\n", (float)(invRemain * 100) / totalTasks);
		}
	}

	/*
    Redis redis;
    redis.Connect();

    redis.SendImage(&image[0].x, WIDTH, HEIGHT, sizeof(pixel));

    memset(image, 0, WIDTH * HEIGHT * sizeof(pixel));
    redis.ReceiveImage(&image[0].x, WIDTH, HEIGHT, sizeof(pixel));
	*/

	t.Reset();
	while(pool.GetTasksCount() > 0)
	{
		int elapsed = (int)(t.ElapsedMillis() * 0.001f);
		if(elapsed != prevElapsed)
		{
			prevElapsed = elapsed;
			int invRemain = totalTasks - pool.GetTasksCount();
			printf("Progress: %f%%\n", (float)(invRemain * 100) / totalTasks);
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	printf("Ended in: %lf ms\n", t.ElapsedMillis());

	applyGlow(image, emission, WIDTH, HEIGHT);

    writePPM("output.ppm", image, WIDTH, HEIGHT);

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
