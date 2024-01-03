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

#define WIDTH 1920
#define HEIGHT 1080

#define SAMPLES 30

void writePPM(const char* path, pixel* img, int width, int height);

std::vector<uint32_t> imageHorizontalIter;
std::vector<uint32_t> imageVerticalIter;

void gaussianBlur(pixel* img, pixel* glowMap, int width, int height, float sigma, int size) {
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

    // applico il blur solo ai pixel con glow
    pixel* tempImg = (pixel*)malloc(width * height * sizeof(pixel));
    memcpy(tempImg, img, width * height * sizeof(pixel));  // Copia l'immagine originale per preservare i pixel senza glow

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (glowMap[i * width + j].x > 0 || glowMap[i * width + j].y > 0 || glowMap[i * width + j].z > 0) {
                float sumX = 0.0, sumY = 0.0, sumZ = 0.0;

                for (int k = -size / 2; k <= size / 2; k++) {
                    for (int l = -size / 2; l <= size / 2; l++) {
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

pixel* createGlowMap(pixel* renderedImage, Material* materials, int width, int height) {
    pixel* glowMap = (pixel*)malloc(width * height * sizeof(pixel));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            Material mat = materials[index];

            if (mat.hasGlow) {
                // Assegna un valore di intensità basato su glowStrength e il colore del pixel
                glowMap[index].x = (unsigned char)(glm::min(255.0f, renderedImage[index].x * mat.glowStrength));
                glowMap[index].y = (unsigned char)(glm::min(255.0f, renderedImage[index].y * mat.glowStrength));
                glowMap[index].z = (unsigned char)(glm::min(255.0f, renderedImage[index].z * mat.glowStrength));
            } else {
                glowMap[index].x = 0;
                glowMap[index].y = 0;
                glowMap[index].z = 0;
            }
        }
    }

    return glowMap;
}

void applyGlow(pixel* image, pixel* glowMap, int width, int height) {
    // Applica il blur per creare l'immagine sfocata
    pixel* blurredImage = (pixel*)malloc(width * height * sizeof(pixel));
    memcpy(blurredImage, image, width * height * sizeof(pixel));
    gaussianBlur(blurredImage, glowMap, width, height, 10.0f, 11);

    // Mescola l'immagine sfocata con l'originale
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;

            // Calcola il fattore di glow in base alla glowMap
            float glowFactor = glowMap[index].x / 255.0f;

            // Mescola i pixel 
            image[index].x = (unsigned char)(glm::min(255, (int)image[index].x + (int)(blurredImage[index].x * glowFactor)));
            image[index].y = (unsigned char)(glm::min(255, (int)image[index].y + (int)(blurredImage[index].y * glowFactor)));
            image[index].z = (unsigned char)(glm::min(255, (int)image[index].z + (int)(blurredImage[index].z * glowFactor)));
        }
    }

    free(blurredImage);
}

int main()
{
    uint32_t width = 1920, height = 1080;

	// Setup framebuffer
    pixel* image = new pixel[width * height];

    imageHorizontalIter.resize(width);
	imageVerticalIter.resize(height);
	for (uint32_t i = 0; i < width; i++)
		imageHorizontalIter[i] = i;
	for (uint32_t i = 0; i < height; i++)
		imageVerticalIter[i] = i;


	// Setup world
	Camera camera(60.0f, width, height, 0.01f, 1000.0f);

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
	materials[0] = Material{ glm::vec3{ 0.8f, 0.8f, 0.0f }, 0.0f,  0.0f,  0.0f,  false, 0.0f   };
	materials[1] = Material{ glm::vec3{ 0.0f, 0.0f, 0.0f }, 0.05f, 0.0f,  1.85f, false, 0.0f   };
	materials[2] = Material{ glm::vec3{ 0.8f, 0.8f, 0.8f }, 0.2f,  0.75f, 0.0f,  true,  100.0f };
	materials[3] = Material{ glm::vec3{ 0.8f, 0.2f, 0.1f }, 0.08f, 0.02f, 0.0f,  true,  100.0f };

	materials[4] = Material{ glm::vec3{ 0.1f, 0.7f, 0.2f }, 0.08f, 0.02f, 0.0f,  false, 0.0f };
	materials[5] = Material{ glm::vec3{ 0.1f, 0.2f, 0.7f }, 0.08f, 0.02f, 0.0f,  false, 0.0f };
	materials[6] = Material{ glm::vec3{ 0.1f, 0.2f, 0.7f }, 0.1f,  0.05f, 0.0f,  false, 0.0f };
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
			pool.enqueue([&image, &width, &height, &camera, &world, &lights, &materials, x, y]()
			{
				// -1 / 1
				float u = ((float)x / (float)width ) * 2.0f - 1.0f;
				float v = ((float)y / (float)height) * 2.0f - 1.0f;

				// curandState randState(x + y * width);

				float pixelOffX = 0.5f / width;
				float pixelOffY = 0.5f / height;

				glm::vec3 result{ 0.0f, 0.0f, 0.0f };
				for(int i = 0; i < SAMPLES; ++i)
					result += glm::clamp(AntiAliasing(u, v, pixelOffX, pixelOffY, camera, &world, &lights, materials /*, &randState*/), glm::vec3(0.0f), glm::vec3(1.0f));
				
				image[x + y * width].Set(result / glm::vec3(SAMPLES));
			});
		}

		totalTasks = y * width;

		int elapsed = (int)(t.ElapsedMillis() * 0.001f);
		if(elapsed != prevElapsed)
		{
			prevElapsed = elapsed;

			int invRemain = totalTasks - pool.GetTasksCount();
			printf("Progress: %f%%\n", (float)(invRemain * 100) / totalTasks);
		}
	}

	/*
    std::for_each(std::execution::par, imageVerticalIter.begin(), imageVerticalIter.end(),
		[&image, width, height, &camera, &world, &lights, &materials](uint32_t &y)
		{
			std::for_each(std::execution::par, imageHorizontalIter.begin(), imageHorizontalIter.end(),
				[&image, width, height, &camera, &world, &lights, &materials, y](uint32_t &x)
				{
					// -1 / 1
					float u = ((float)x / (float)width ) * 2.0f - 1.0f;
					float v = ((float)y / (float)height) * 2.0f - 1.0f;

					float pixelOffX = 0.5f / width;
					float pixelOffY = 0.5f / height;
					glm::vec3 result = AntiAliasing(u, v, pixelOffX, pixelOffY, camera, &world, &lights, materials);

					result = glm::clamp(result, glm::vec3(0.0f), glm::vec3(1.0f));
					image[x + y * width].Set(result);
				});
		});
	*/

	/*
    Redis redis;
    redis.Connect();

    redis.SendImage(&image[0].x, width, height, sizeof(pixel));

    memset(image, 0, width * height * sizeof(pixel));
    redis.ReceiveImage(&image[0].x, width, height, sizeof(pixel));
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

	writePPM("output.ppm", image, width, height);

	// glowMap gen
	pixel* glowMap = createGlowMap(image, materials, width, height); 

	// applico il glow
    applyGlow(image, glowMap, width, height);

    writePPM("output_with_glow.ppm", image, width, height);

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
