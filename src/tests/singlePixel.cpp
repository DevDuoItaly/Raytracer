#include "renderer.h"

#include "lights/directionalLight.h"
#include "lights/lightsList.h"

#include "hittables/hittablesList.h"
#include "hittables/sphere.h"

#include "utils/threadPool.h"
#include "utils/timer.h"

#include "utils/random.h"

#define WIDTH 1024
#define HEIGHT 512

#define MAXDEPTH 20

int main()
{
	// Setup world
	Camera camera(60.0f, WIDTH, HEIGHT, 0.01f, 1000.0f);

	Random rand;

    // -- Init Lights
	int lightsCount = 1;
	Light** l_light = new Light*[lightsCount];
	l_light[0] = new DirectionalLight({ -0.25f, -0.75f, 0.45f });

	Light* lights = new LightsList(l_light, lightsCount);

	// Init world
	int objectsCount = 5;
	Hittable** l_world = new Hittable*[objectsCount];

	// Loop over query results and create spheres
	for(int i = 0; i < objectsCount; ++i)
		l_world[i] = new Sphere(rand.randomPosition(), rand.randomRadius(), 0);

	Hittable* world = new HittablesList(l_world, objectsCount);

    // -- Init Materials
	Material* materials = new Material[4];
    materials[0] = Material{ { 0.8f, 0.8f, 0.8f }, 0.9f,  0.75f, 0.0f , { 0.0f, 0.0f, 0.0f }, 0.0f };

	// Raytrace
	ThreadPool pool(std::thread::hardware_concurrency() - 1);

	printf("Running Raytracing MT...\n");
	Timer t;

	int prevElapsed = -1;

	int tileSize = 16;
	int splitH = WIDTH / tileSize, splitV = HEIGHT / tileSize;

	float totalTasks = splitH * splitV;

	std::mutex lock;
	bool hasMaxDepth = false;

	int count = 0;

	for(int sY = 0; sY < splitV; ++sY)
	{
		for(int sX = 0; sX < splitH; ++sX)
			pool.enqueue([&lock, &hasMaxDepth, &count, &camera, &world, &lights, &materials, sX, sY, &tileSize]()
			{
				bool max = false;
				for(int nY = 0; nY < tileSize; ++nY)
					for(int nX = 0; nX < tileSize; ++nX)
					{
						int x = nX + sX * tileSize, y = nY + sY * tileSize;

						// -1 / 1
						float u = ((float)x / (float)WIDTH ) * 2.0f - 1.0f;
						float v = ((float)y / (float)HEIGHT) * 2.0f - 1.0f;

						curandState randState(x + y * WIDTH);

						max |= MaxRayDistance(u, v, &camera, &world, &lights, materials, &randState, MAXDEPTH);
					}
				
				std::lock_guard<std::mutex> l(lock);

				hasMaxDepth |= max;
				count++;
			});
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
			printf("Progress: %f%%\n", (float)count * 100 / totalTasks);
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		std::lock_guard<std::mutex> l(lock);
		waiting = count < totalTasks;
	}

	// Freeing memories
	delete lights;
	delete world;
	delete materials;

	printf("Ended in: %lf ms\n", t.ElapsedMillis());

	printf("Status: %s!\n", hasMaxDepth ? "Failed" : "Success");

	return 0;
}
