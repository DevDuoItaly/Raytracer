#pragma once

#include "core.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera
{
public:
    PREFIX Camera(float fov, float width, float height, float nearPlane, float farPlane)
        : m_Position({ -3.0f, 6.0f, 10.0f })
    {
        // Projection matrix
        m_ProjectionMatrix = glm::perspectiveFov(glm::radians(fov), (float)width, (float)height, nearPlane, farPlane);
        m_invProjectionMatrix = glm::inverse(m_ProjectionMatrix);

        // View matrix
        glm::vec3 upDirection = glm::vec3(0.0f, 1.0f, 0.0f); //up lungo l'asse y
        m_ViewMatrix = glm::lookAt(m_Position, glm::vec3{ 0.0f, 3.0f, 0.0f }, upDirection);
        m_invViewMatrix = glm::inverse(m_ViewMatrix);
    }

    PREFIX const glm::vec3& GetPosition() const
    {
        return m_Position;
    }

    PREFIX const glm::mat4& GetProjectionMatrix() const
    {
        return m_ProjectionMatrix;
    }

    PREFIX const glm::mat4& GetInverseProjectionMatrix() const
    {
        return m_invProjectionMatrix;
    }

    PREFIX const glm::mat4& GetViewMatrix() const
    {
        return m_ViewMatrix;
    }

    PREFIX const glm::mat4& GetInverseViewMatrix() const
    {
        return m_invViewMatrix;
    }
    
private:
    glm::vec3 m_Position           { 0.0f, 0.0f, 0.0f };
    glm::mat4 m_ProjectionMatrix   { 0.0f };
    glm::mat4 m_invProjectionMatrix{ 0.0f };
    glm::mat4 m_ViewMatrix         { 0.0f };
    glm::mat4 m_invViewMatrix      { 0.0f };
};
