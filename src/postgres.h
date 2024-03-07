#pragma once

#include "material.h"

#include "lights/light.h"
#include "lights/lightsList.h"
#include "lights/directionalLight.h"

#include "hittables/sphere.h"
#include "hittables/hittable.h"
#include "hittables/hittablesList.h"

#include <pqxx/pqxx>
#include <random>


// Class to handle PostgreSQL database interactions
class postgres
{
public:
    postgres()
        // Initialize the database connection with credentials
        : m_Cxn("dbname=imagedb user=root password=password hostaddr=127.0.0.1 port=5432")
    {
        // Check if the connection is open, print error message if not
        if (!m_Cxn.is_open())
        {
            printf("Cannot connect to postgres!\n");
            return;
        }

        printf("Postgres connected to %s!\n", m_Cxn.dbname());

        // Initialize the database
        initDatabase();
    }
    
    ~postgres()
    {
        // Close the database connection if it's open
        if(m_Cxn.is_open())
            m_Cxn.disconnect();
    }

    // Function to check if a table exists in the database
    bool existTable(pqxx::nontransaction& work, const char* tablename)
    {
        std::stringstream ss;
        ss << "SELECT EXISTS(SELECT FROM pg_type WHERE typname='" << tablename << "')";
        return work.exec(ss.str().c_str())[0][0].as<bool>();
    }

    // Function to retrieve materials from the database
    Material* getMaterials()
    {
        pqxx::nontransaction work(m_Cxn);

        pqxx::result query = work.exec("SELECT * FROM material");
        if(query.empty())
            return 0;
        
        Material* materials = (Material*) malloc(query.size() * sizeof(Material));

        // Loop over query results and populate materials
        for(int i = 0; i < query.size(); ++i)
        {
            pqxx::row row = query[i];
            materials[i].color         = readVec3(row[1].c_str());
            materials[i].roughness     = row[2].as<float>();
            materials[i].reflection    = row[3].as<float>();
            materials[i].refraction    = row[4].as<float>();
            materials[i].emissionColor = readVec3(row[5].c_str());
            materials[i].glowStrength  = row[6].as<float>();
        }

        return materials;
    }

    // Function to retrieve lights from the database for a given scene
    Light* getLights(int sceneID)
    {
        pqxx::nontransaction work(m_Cxn);

        std::stringstream ss;
        ss << "SELECT * FROM directional_light WHERE scene_id=" << sceneID;

        pqxx::result query = work.exec(ss.str().c_str());
        if(query.empty())
            return 0;
        
        Light** l_light = new Light*[query.size()];

        // Loop over query results and create directional lights
        for(int i = 0; i < query.size(); ++i)
        {
            pqxx::row row = query[i];
            DirectionalLight* light = new DirectionalLight(readVec3(row[1].c_str()));
            l_light[i] = light;
        }

        return new LightsList(l_light, query.size());
    }

    // Function to retrieve world objects from the database for a given scene
    Hittable* getWorld(int sceneID)
    {
        pqxx::nontransaction work(m_Cxn);

        std::stringstream ss;
        ss << "SELECT * FROM sphere WHERE scene_id=" << sceneID;

        pqxx::result query = work.exec(ss.str().c_str());
        if(query.empty())
            return 0;
        
        Hittable** l_world = new Hittable*[query.size()];

        // Loop over query results and create spheres
        for(int i = 0; i < query.size(); ++i)
        {
            pqxx::row row = query[i];
            Sphere* sphere = new Sphere(readVec3(row[1].c_str()), row[2].as<float>(), row[3].as<int>());
            l_world[i] = sphere;
        }

        return new HittablesList(l_world, query.size());
    }

    //genera casualmente una scena
    void generateRandomScene()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f); //range per coordinate x, y, z

        for (int i = 0; i < NUM_SPHERES; ++i)
        {
            glm::vec3 position(dis(gen), dis(gen), dis(gen));
            float radius = dis(gen);
            glm::vec3 color(dis(gen), dis(gen), dis(gen));

            // Aggiungi la sfera generata casualmente alla scena nel database
            addSphereToScene(position, radius, color);
        }
    }

private:

    // Function to initialize the database
    void initDatabase()
    {
        pqxx::nontransaction work(m_Cxn);

        // Create custom type VEC3 if it does not exist
        if(!existTable(work, "vec3"))
            work.exec("CREATE TYPE VEC3 AS ( \
                        x REAL, \
                        y REAL, \
                        z REAL  \
                    );");
        
        // Create scene table and populate it with default values if it does not exist
        if(!existTable(work, "scene"))
        {
            work.exec("CREATE TABLE IF NOT EXISTS scene ( \
                        id   INT         PRIMARY KEY NOT NULL, \
                        name VARCHAR(25)             NOT NULL  \
                    );");
            work.exec("INSERT INTO scene (id, name) VALUES \
                    (0, 'default')");
        }

        // Create directional_light table and populate it with default values if it does not exist
        if(!existTable(work, "directional_light"))
        {
            work.exec("CREATE TABLE IF NOT EXISTS directional_light ( \
                        id        INT  PRIMARY KEY NOT NULL, \
                        direction VEC3             NOT NULL, \
                        scene_id  INT  REFERENCES scene(id)  \
                    );");
            work.exec("INSERT INTO directional_light (id, direction, scene_id) VALUES \
                    (0, '(-0.25, -0.75, 0.45)'::VEC3, 0)");
        }

        // Create material table and populate it with default values if it does not exist
        if(!existTable(work, "material"))
        {
            work.exec("CREATE TABLE IF NOT EXISTS material ( \
                        id             INT  PRIMARY KEY  NOT NULL, \
                        color          VEC3              NOT NULL, \
                        roughness      REAL              NOT NULL, \
                        reflection     REAL              NOT NULL, \
                        refraction     REAL              NOT NULL, \
                        emission_color VEC3              NOT NULL, \
                        glow_strength  REAL              NOT NULL  \
                    );");
            work.exec("INSERT INTO material (id, color, roughness, reflection, refraction, emission_color, glow_strength) VALUES \
                    (0, '(0.8, 0.8, 0.0)'::VEC3, 0.0,  0.0,  0.0 , '(0.0, 0.0, 0.0)'::VEC3, 0.0), \
                    (1, '(0.8, 0.2, 0.1)'::VEC3, 0.08, 0.02, 0.0 , '(1.0, 0.0, 0.0)'::VEC3, 4.5), \
                    (2, '(0.8, 0.8, 0.8)'::VEC3, 0.9,  0.75, 0.0 , '(0.0, 0.0, 0.0)'::VEC3, 0.0), \
                    (3, '(0.0, 0.0, 0.0)'::VEC3, 0.0,  0.0 , 1.85, '(0.0, 0.0, 0.0)'::VEC3, 0.0)");
        }

        // Create sphere table and populate it with default values if it does not exist
        if(!existTable(work, "sphere"))
        {
            work.exec("CREATE TABLE IF NOT EXISTS sphere ( \
                        id          INT  PRIMARY KEY NOT NULL,    \
                        position    VEC3             NOT NULL,    \
                        radius      REAL             NOT NULL,    \
                        material_id INT  REFERENCES material(id), \
                        scene_id    INT  REFERENCES scene(id)     \
                    );");
            work.exec("INSERT INTO sphere (id, position, radius, material_id, scene_id) VALUES \
                    (0, '( 0.0, -1000.0, -4.0)'::VEC3, 1000.0, 0, 0), \
                    (1, '( 0.0,  1.0,    -4.0)'::VEC3, 1.0,    1, 0), \
                    (2, '(-3.0,  1.0,    -4.0)'::VEC3, 1.0,    2, 0), \
                    (3, '( 3.0,  1.0,    -4.0)'::VEC3, 1.0,    3, 0)");
        }
    }

	// Function to parse a VEC3 type from string | DEFAULT: (0.0, 0.0, 0.0)
	glm::vec3 readVec3(const char* str)
	{
		std::string s(str);
		int begin = 1, end = s.find(',');

		glm::vec3 v{ 0.0f, 0.0f, 0.0f };

		if(end == std::string::npos) { printf("Invalid string!\n"); return v; }

		v.x = (float) std::atof(s.substr(begin, end).c_str());
		begin = end + 1;
		end = s.find(',', begin);

		if(end == std::string::npos) { printf("Invalid string!\n"); return v; }

		v.y = (float) std::atof(s.substr(begin, end).c_str());
		begin = end + 1;
		end = s.find(')', begin);

		if(end == std::string::npos) { printf("Invalid string!\n"); return v; }

		v.z = (float) std::atof(s.substr(begin, end).c_str());

		return v;
	}

    //aggiunge una sfera alla scena nel database
    void addSphereToScene(const glm::vec3& position, float radius, const glm::vec3& color)
    {
        std::stringstream ss;
        ss << "INSERT INTO sphere (position, radius, material_id, scene_id) VALUES ('("
           << position.x << ", " << position.y << ", " << position.z << ")', "
           << radius << ", " << getRandomMaterialID() << ", " << SCENE_ID << ")";

        pqxx::nontransaction work(m_Cxn);
        work.exec(ss.str().c_str());
    }

    //ritorna un ID casuale per il materiale delle sfere
    int getRandomMaterialID()
    {
        //TODO
        return 0;
    }

private:
	pqxx::connection m_Cxn;
    static const int SCENE_ID = 0;
    static const int NUM_SPHERES = 10;
};
