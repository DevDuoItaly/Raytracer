#include "renderer.h"

#include "lights/light.h"

#include "hittables/hittablesList.h"
#include "hittables/sphere.h"
#include "hittables/cube.h"
#include "hittables/plane.h"

#include "postgres.h"

#include "utils/timer.h"
#include "utils/threadPool.h"

#include <vector>
#include <algorithm>
#include <execution>

#define WIDTH 256
#define HEIGHT 128

#define SAMPLES 20

#define MAXDEPTH 20

double Render(Camera& camera, Hittable* world, Light* lights, Material* materials, int threads);

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
	double startTime = Render(camera, world, lights, materials, 1);
	printf("Ended in: %lf\n\n", startTime);

	double time = 0;
	int threads = 2;
	while(time < 5000 && threads <= std::thread::hardware_concurrency() - 2)
	{
		time = Render(camera, world, lights, materials, threads);
		printf("Speed Up: %lf\n", startTime / time);
		threads += 2;
	}

	// Freeing memories
	delete lights;
	delete world;
	delete materials;
	
    return 0;
}

double Render(Camera& camera, Hittable* world, Light* lights, Material* materials, int threads)
{
	ThreadPool pool(threads);

	Timer t;

	int prevElapsed = -1;

	int tileSize = 16;
	int splitH = WIDTH / tileSize, splitV = HEIGHT / tileSize;

	float totalTasks = splitH * splitV;

	std::mutex lock;
	int count = 0;

	for(int sY = 0; sY < splitV; ++sY)
	{
		for(int sX = 0; sX < splitH; ++sX)
		{
			pool.enqueue([&count, &lock, &camera, &world, &lights, &materials, sX, sY, &tileSize]()
			{
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
					}
				
				std::lock_guard<std::mutex> l(lock);
				count++;
			});
		}
	}

	t.Reset();
	bool waiting = true;

	// Waiting all tasks ended
	while(waiting)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		std::lock_guard<std::mutex> l(lock);
		waiting = count < totalTasks;
	}

	return t.ElapsedMillis();
}