#pragma once

#include "material.h"

#include <pqxx/pqxx>

class postgres
{
public:
	postgres()
		: m_Cxn("dbname=imagedb user=root password=password hostaddr=127.0.0.1 port=5432")
	{
		if (!m_Cxn.is_open())
		{
			printf("Cannot connect to postgres!\n");
			return;
		}

		printf("Postgres connected to %s!\n", m_Cxn.dbname());

		initDatabase();
	}

	~postgres()
	{
		if(m_Cxn.is_open())
			m_Cxn.disconnect();
	}

	bool existTable(pqxx::nontransaction& work, const char* tablename)
	{
		std::stringstream ss;
		ss << "SELECT EXISTS(SELECT FROM pg_type WHERE typname='" << tablename << "')";
		return work.exec(ss.str().c_str())[0][0].as<bool>();
	}

	Material* getMaterials(int sceneID)
	{
		pqxx::nontransaction work(m_Cxn);

		pqxx::result query = work.exec("SELECT FROM material");
		if(query.empty())
			return 0;
		
		Material* materials = (Material*) malloc(query.size() * sizeof(Material));
		for(int i = 0; i < query.size(); ++i)
		{
			pqxx::row row = query[i];
			materials[i].color         = row[1].as<glm::vec3>();
			materials[i].roughness     = row[2].as<float>();
			materials[i].reflection    = row[3].as<float>();
			materials[i].refraction    = row[4].as<float>();
			materials[i].emissionColor = row[5].as<glm::vec3>();
			materials[i].glowStrength  = row[6].as<float>();
		}

		return materials;
	}

private:
	void initDatabase()
	{
		pqxx::nontransaction work(m_Cxn);

		if(!existTable(work, "vec3"))
			work.exec("CREATE TYPE VEC3 AS ( \
						x REAL, \
						y REAL, \
						z REAL  \
					);");
		
		if(!existTable(work, "scene"))
		{
			work.exec("CREATE TABLE IF NOT EXISTS scene ( \
						id   INT         PRIMARY KEY NOT NULL, \
						name VARCHAR(25)             NOT NULL  \
					);");
			work.exec("INSERT INTO scene (id, name) VALUES \
					(0, 'default')");
		}

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

private:
	pqxx::connection m_Cxn;
};
