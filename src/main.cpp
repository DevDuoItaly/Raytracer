#include "renderer.h"

#include "lights/light.h"

#include "hittables/hittablesList.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"
#include "hittables/plane.h"

#include "redis.h"
#include "postgres.h"

#include "utils/timer.h"
#include "utils/threadPool.h"

#include <vector>
#include <algorithm>
#include <execution>

#define WIDTH 1024
#define HEIGHT 512

#define SAMPLES 10

#define MAXBLUR 5

#define MAXDEPTH 20

void writePPM(const char* path, pixel* img,              int width, int height);
void writePPM(const char* path, emissionPixel* emission, int width, int height);

emissionPixel* downsample(emissionPixel* emission, int width, int height, int scaleFactor)
{
	int startWidth = width;

	// Calculate new scaled image sizes and create the new image
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

			// Calculate the average emission value
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
	// Calculate new scaled image sizes and create the new image
	int scaledW = width * scaleFactor;
	emissionPixel* result = new emissionPixel[scaledW * height * scaleFactor];

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			emissionPixel& p = emission[x + y * width];

			// Diffuses the pixel value for the tile in the scaled image
			for (int nY = 0; nY < scaleFactor; ++nY)
				for (int nX = 0; nX < scaleFactor; ++nX)
					result[(x * scaleFactor + nX) + ((y * scaleFactor + nY) * scaledW)].Set(p.emission, p.strenght);
		}

	return result;
}

void gaussianBlur(emissionPixel* emission, int width, int height, float sigma, int size) 
{
	assert(size > 0);

	// Create the kernel values for the blur
    float kernel[size * 2 + 1][size * 2 + 1];
	{
		float sum = 0.0f;

		// Calculate exponential value
		for (int y = -size; y <= size; ++y)
			for (int x = -size; x <= size; ++x)
			{
				float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
				kernel[x + size][y + size] = value;
				sum += value;
			}
		
		sum = 1.0f / sum;
		
		// Normalize the values
		for (int y = 0; y < size * 2 + 1; ++y)
			for (int x = 0; x < size * 2 + 1; ++x)
				kernel[x][y] *= sum;
	}

	// Copy the original image in a new one
    emissionPixel* tempEmission = (emissionPixel*) malloc(width * height * sizeof(emissionPixel));
    memcpy(tempEmission, emission, width * height * sizeof(emissionPixel));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
		{
			emissionPixel pixel{ { 0.0f, 0.0f, 0.0f }, 0.0f };

			int emissionsCount = 0;

			// For each pixel in the kernel box add the weighted values
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

    // Copy the blurred pixel in the original image
    memcpy(emission, tempEmission, width * height * sizeof(emissionPixel));
    free(tempEmission);
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

		// Combine upscaled image with base image
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

	free(emission);
}

int main()
{
	// Init Postgres
	postgres db;

	// Setup world
	Camera camera(60.0f, WIDTH, HEIGHT, 0.01f, 1000.0f);

    // -- Init Lights
	Light* lights = db.getLights(0);

	// -- Init Materials
	int materialsCount = 0;
    Material* materials = db.getMaterials(&materialsCount);

	// Init world
	Hittable* world = db.generateRandomScene(materialsCount);

	// Raytrace
	ThreadPool pool(std::thread::hardware_concurrency() - 1);

	printf("Running Raytracing MT...\n");
	Timer t;

	int prevElapsed = -1;

	int tileSize = 16;
	int splitH = WIDTH / tileSize, splitV = HEIGHT / tileSize;

	float totalTasks = splitH * splitV * 2;

	Redis redis;
    redis.Connect();

	std::mutex lock;

	for(int sY = 0; sY < splitV; ++sY)
	{
		for(int sX = 0; sX < splitH; ++sX)
		{
			pool.enqueue([&redis, &lock, &camera, &world, &lights, &materials, sX, sY, &tileSize]()
			{
				// Setup local tile
				pixel* image = new pixel[tileSize * tileSize];
				emissionPixel* emission = new emissionPixel[tileSize * tileSize];

				for(int nY = 0; nY < tileSize; ++nY)
					for(int nX = 0; nX < tileSize; ++nX)
					{
						int x = nX + sX * tileSize, y = nY + sY * tileSize;

						// -1 / 1
						float u = ((float)x / (float)WIDTH ) * 2.0f - 1.0f;
						float v = ((float)y / (float)HEIGHT) * 2.0f - 1.0f;

						curandState randState(x + y * WIDTH);

						float pixelOffX = 0.5f / WIDTH;
						float pixelOffY = 0.5f / HEIGHT;

						HitColorGlow result;
						for(int i = 0; i < SAMPLES; ++i)
						{
							HitColorGlow sample = AntiAliasing(u, v, pixelOffX, pixelOffY, &camera, &world, &lights, materials, &randState, MAXDEPTH);
							result.color            += glm::clamp(sample.color,    glm::vec3(0.0f), glm::vec3(1.0f));
							result.emission         += glm::clamp(sample.emission, glm::vec3(0.0f), glm::vec3(1.0f));
							result.emissionStrenght += sample.emissionStrenght;
						}

						image   [nX + nY * tileSize].Set(result.color    / glm::vec3(SAMPLES));
						emission[nX + nY * tileSize].Set(result.emission / glm::vec3(SAMPLES), result.emissionStrenght / SAMPLES);
					}
				
				std::lock_guard<std::mutex> l(lock);

				// Upload tile to redis
				redis.SendImage(reinterpret_cast<unsigned char*>(image),    sX, sY, tileSize, tileSize, sizeof(pixel));
				redis.SendImage(reinterpret_cast<unsigned char*>(emission), sX, sY, tileSize, tileSize, sizeof(emissionPixel));

				delete image;
				delete emission;
			});
		}

		int elapsed = (int)(t.ElapsedMillis() * 0.001f);
		if(elapsed != prevElapsed)
		{
			prevElapsed = elapsed;

			std::lock_guard<std::mutex> l(lock);
			printf("Progress: %f%%\n", (float)(redis.GetCount() * 100) / totalTasks);
		}
	}

	t.Reset();
	bool waiting = true;

	// Waiting all tasks ended
	while(waiting)
	{
		int elapsed = (int)(t.ElapsedMillis() * 0.001f);
		if(elapsed != prevElapsed)
		{
			prevElapsed = elapsed;

			std::lock_guard<std::mutex> l(lock);
			printf("Progress: %f%%\n", (float)(redis.GetCount() * 100) / totalTasks);
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		std::lock_guard<std::mutex> l(lock);
		waiting = redis.GetCount() < totalTasks;
	}

	// Freeing memories
	delete lights;
	delete world;
	delete materials;

	printf("Ended in: %lf ms\n", t.ElapsedMillis());
	
	printf("Recomposing image!\n");
	t.Reset();

	// Recompose the image
	pixel* image = (pixel*) malloc(WIDTH * HEIGHT * sizeof(pixel));
	emissionPixel* emission = (emissionPixel*) malloc(WIDTH * HEIGHT * sizeof(emissionPixel));

	{
		pixel         img[tileSize * tileSize];
		emissionPixel em [tileSize * tileSize];
		int x = 0, y = 0;
		for(int i = 0; i < totalTasks; i += 2)
		{
			// Retrieve tile
			redis.ReceiveImage(reinterpret_cast<unsigned char*>(img), x, y, tileSize, tileSize, sizeof(pixel));
			redis.ReceiveImage(reinterpret_cast<unsigned char*>(em),  x, y, tileSize, tileSize, sizeof(emissionPixel));

			// Copy tile into final image
			for(int j = 0; j < tileSize; ++j)
			{
				memcpy(image    + x * tileSize + ((y * tileSize) + j) * WIDTH, img + j * tileSize, tileSize * sizeof(pixel));
				memcpy(emission + x * tileSize + ((y * tileSize) + j) * WIDTH, em  + j * tileSize, tileSize * sizeof(emissionPixel));
			}
		}
	}
	
	printf("Ended in: %lf ms\n", t.ElapsedMillis());

	printf("Appling glow!\n");
	t.Reset();

	applyGlow(image, emission, WIDTH, HEIGHT);

	printf("Ended in: %lf ms\n", t.ElapsedMillis());

    writePPM("output.ppm", image, WIDTH, HEIGHT);

	free(image);

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
