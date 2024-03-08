#include "glm/glm.hpp"

#include <random>

class Random
{
public:
	Random()
	{
		std::random_device rd;
		m_Engine = std::mt19937(rd());

		m_RandPos    = std::uniform_real_distribution<float>(-10.0f, 10.0f);
		m_RandRadius = std::uniform_real_distribution<float>(1.0f, 5.0f);
	}

	glm::vec3 randomPosition()
	{
		return glm::vec3{ m_RandPos(m_Engine), 1.5f, m_RandPos(m_Engine) };
	}
	
	float randomRadius()
	{
		return m_RandRadius(m_Engine);
	}

private:
	std::mt19937 m_Engine;

	std::uniform_real_distribution<float> m_RandPos;
	std::uniform_real_distribution<float> m_RandRadius;
};
