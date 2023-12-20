#include "renderer.h"

#include "lights/directional_light.h"
#include "lights/lights_list.h"

#include "hittables/hittables_list.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"
#include "hittables/plane.h"

#include "redis.h"

#include "utils/timer.h"

#include <vector>
#include <algorithm>
#include <execution>

void writePPM(const char* path, pixel* img, int width, int height);

std::vector<uint32_t> imageHorizontalIter;
std::vector<uint32_t> imageVerticalIter;



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
	Light** l_light = (Light**) malloc(1 * sizeof(Light*));
	l_light[0] = new DirectionalLight({ -0.35f, 1.0f, 0.0f });
	
    Light* lights = new LightsList(l_light, 1);

    // -- Init World
	Hittable** l_world = (Hittable**) malloc(3 * sizeof(Hittable*));
	l_world[0] = new Sphere({ 0.0f, -1.0f, 5.0f }, 0.5f, 0);
	l_world[1] = new Sphere({ 0.0f, -6.5f, 5.0f }, 5.0f, 1);
	l_world[2] = new Plane ({ 0.0f, -4.5f, 5.0f }, { 0.0f,  -1.0f, 0.0f }, 2);
	
	Hittable* world = new HittablesList(l_world, 3);

    // -- Init Materials
    Material* materials = new Material[3];
	materials[0] = Material{ glm::vec3{ 1.0f, 0.0f, 1.0f }, 0.0f, 0.0f };
	materials[1] = Material{ glm::vec3{ 0.2f, 0.3f, 0.8f }, 0.0f, 0.0f };
	materials[2] = Material{ glm::vec3{ 0.15f, 0.15f, 0.15f }, 0.0f, 0.0f };


	// Raytrace
	printf("Running Raytracing MT...\n");
	Timer t;

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

	printf("Ended in: %lf ms\n", t.ElapsedMillis());

	printf("Running Raytracing ST...\n");
	t.Reset();

	for(uint32_t y = 0; y < height; ++y)
		for(uint32_t x = 0; x < width; ++x)
			{
				// -1 / 1
				float u = ((float)x / (float)width ) * 2.0f - 1.0f;
				float v = ((float)y / (float)height) * 2.0f - 1.0f;

				float pixelOffX = 0.5f / width;
				float pixelOffY = 0.5f / height;
				glm::vec3 result = AntiAliasing(u, v, pixelOffX, pixelOffY, camera, &world, &lights, materials);

				result = glm::clamp(result, glm::vec3(0.0f), glm::vec3(1.0f));
				image[x + y * width].Set(result);
			}

	printf("Ended in: %lf ms\n", t.ElapsedMillis());

    Redis redis;
    redis.Connect();

    redis.SendImage(&image[0].x, width, height, sizeof(pixel));

    memset(image, 0, width * height * sizeof(pixel));
    redis.ReceiveImage(&image[0].x, width, height, sizeof(pixel));

    writePPM("output.ppm", image, width, height);

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
