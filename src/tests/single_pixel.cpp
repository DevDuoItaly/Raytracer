#include "renderer.h"

#include "lights/directional_light.h"
#include "lights/lights_list.h"

#include "hittables/hittables_list.h"
#include "hittables/sphere.h"

int main()
{
    uint32_t width = 210, height = 180;

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

	Hittable** l_world = (Hittable**) malloc(1 * sizeof(Hittable*));
	l_world[0] = new Sphere({  0.0f, -1000.0f, 0.0f }, 1000.0f, 0);
	
	Hittable* world = new HittablesList(l_world, 1);

    // -- Init Materials
    Material* materials = new Material[1];
	materials[0] = Material{ glm::vec3{ 0.8f, 0.8f, 0.0f }, 0.0f,  0.0f,  0.0f,  false, 0.0f   };

	// Raytrace
    int x = 0, y = height - 1;

    // -1 / 1
    float u = ((float)x / (float)width ) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    // curandState randState(x + y * width);

    float pixelOffX = 0.5f / width;
    float pixelOffY = 0.5f / height;
	glm::vec3 result = AntiAliasing(u, v, pixelOffX, pixelOffY, &camera, &world, &lights, materials /*, &randState*/);

    return 0;
}
