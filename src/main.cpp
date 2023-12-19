#include "redis.h"

#include "glm/glm.hpp"

#include <vector>
#include <algorithm>
#include <execution>

struct pixel
{
    unsigned char x = 0, y = 0, z = 0;

    void Set(const glm::vec3& c)
    {
        x = (unsigned char)(c.x * 255.0f);
        y = (unsigned char)(c.y * 255.0f);
        z = (unsigned char)(c.z * 255.0f);
    }
};

void writePPM(const char* path, pixel* img, int width, int height);

std::vector<uint32_t> m_ImageHorizontalIter;
std::vector<uint32_t> m_ImageVerticalIter;

int main()
{
    uint32_t width = 1920, height = 1080;

    pixel* image = new pixel[width * height];

    m_ImageHorizontalIter.resize(width);
	m_ImageVerticalIter.resize(height);
	for (uint32_t i = 0; i < width; i++)
		m_ImageHorizontalIter[i] = i;
	for (uint32_t i = 0; i < height; i++)
		m_ImageVerticalIter[i] = i;

    std::for_each(std::execution::par, m_ImageVerticalIter.begin(), m_ImageVerticalIter.end(),
		[image, width](uint32_t y)
		{
			std::for_each(std::execution::par, m_ImageHorizontalIter.begin(), m_ImageHorizontalIter.end(),
				[image, width, y](uint32_t x)
				{
					glm::vec3 color = { 1.0f, 0.0f, 1.0f }; // PerPixel(x, y);

					color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
					image[x + y * width].Set(color);
				});
		});


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
