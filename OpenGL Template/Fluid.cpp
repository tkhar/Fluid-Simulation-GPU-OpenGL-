#include "Fluid.h"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "fluid_util.h"

using namespace std;

#define pres_dens_coef 10.0f // pressure-density coefficient
#define rest_density 10.0f // rest density parameter
#define gX 0.0f// gravity X
#define gY -1.0f// gravity Y
#define viscocity_coeff 10.0f // viscocity coefficient
#define stiffness_coeff 100.0f // stiffness for environmental respeonse
#define pi 3.14159

/////////////////////////////////// Constructor methods /////////////////////////////////////

Fluid::Fluid(unsigned int _nx, unsigned int _ny, double m, double _dx, double _dy, double _h)
{
	nx = _nx;
	ny = _ny;
	dx = _dx;
	dy = _dy;
	h = _h;
	Nx = (int)((nx * dx + 2 * 0.21 * nx * dx) / h) + 1;
	Ny = (int)((ny * dy + 2 * 0.21 * ny * dy) / h) + 1;
	// Initialize arrays of particles
	mass = new double[nx * ny];
	positionX = new double[nx * ny];
	positionY = new double[nx * ny];
	velocityX = new double[nx * ny];
	velocityY = new double[nx * ny];
	accX = new double[nx * ny];
	accY = new double[nx * ny];
	density = new double[nx * ny];
	pressure = new double[nx * ny];
	radius = new double[nx * ny];
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			mass[j * nx + i] = m;
			positionX[j * nx + i] = i * dx - 0.1f;
			positionY[j * nx + i] = j * dy - 0.1f;
			velocityX[j * nx + i] = 0.0f;
			velocityY[j * nx + i] = 0.0f;
			density[j * nx + i] = 1.0f;
			pressure[j * nx + i] = 0.0f;
			radius[j * nx + i] = 0.1f;
		}
	}
	// initialize spatial hashtables
	Hashtable = new int* [Nx * Ny];

	// initialize particle neighborhoods array
	Particle_Neighborhood = new int* [nx * ny];
	max_size_of_particle_neighborhood = 10 * max_size_of_value_list;

	// particle positions with glm vectors
	particlePositions = new glm::vec3[nx*ny];
	generateGLMParticles();

	// utility initialization
	utility = fluid_util(nx, ny, dx, dy, h);
}

void Fluid::generateGLMParticles()
{
	// fill up particle array for rendering
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			int particle_idx = j * nx + i;
			particlePositions[particle_idx] = glm::vec3(positionX[particle_idx],
														positionY[particle_idx], 0.0f);
		}
	}
}

void Fluid::spatialHashTableInit()
{
	// Initialize spatial hashtable 
	// Call function to populate this array
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			Hashtable[j * Nx + i] = new int[max_size_of_value_list];
			for (int k = 0; k < max_size_of_value_list; k++)
			{
				Hashtable[j * Nx + i][k] = -1;
			}
		}
	}
}

/////////////////////////////// End of constructor methods /////////////////////////////


///////////////////////////////// OTHER PUBLIC METHODS /////////////////////////////////

void Fluid::initializeSpatialHashtable()
{
	// For each particle, find which cell it belongs to
	// and insert its index there!
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			int cx = (int)((positionX[j * nx + i] + 0.21 * nx * dx - 0.1f) / h);
			int cy = (int)((positionY[j * nx + i] + 0.21 * ny * dy - 0.1f) / h);
			int c = cy * Nx + cx;
			int k = 0;
			while (k < max_size_of_value_list && Hashtable[c][k] != -1)
			{
				k++;
			}
			if (k < max_size_of_value_list)
			{
				Hashtable[c][k] = j * nx + i;
			}
			else
			{
				printf("We have to increase the max_size_of_value_list\n");
			}
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
}

void Fluid::testSpatialHashtable()
{
	printf("Spatial hashtable (populated): ");
	for (int i = 0; i < max_size_of_value_list; i++)
	{
		cout << Hashtable[(Nx * Ny) / 2][i] << " ";
	}
	cout << endl;
}

void Fluid::particleNeighborhoodsInit()
{
	// Initialize particle neighborhood array
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			Particle_Neighborhood[j * nx + i] = new int[max_size_of_particle_neighborhood];
			for (int k = 0; k < max_size_of_particle_neighborhood; k++)
			{
				Particle_Neighborhood[j * nx + i][k] = -1;
			}
		}
	}
}

glm::vec3 Fluid::getPosition(unsigned int idx)
{
	return particlePositions[idx];
}

void Fluid::neighborSearch()
{
	// neighbor search
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			int cx = (int)((positionX[j * nx + i] + 0.21 * nx * dx - 0.1f) / h);
			int cy = (int)((positionY[j * nx + i] + 0.21 * ny * dy - 0.1f) / h);
			int neighborhood_idx = 0;
			for (int k = 0; k < 9; k++)
			{
				int cell_x = cx + utility.neighborsX[k], cell_y = cy + utility.neighborsY[k];
				int hash_idx = cell_y * Nx + cell_x;
				if (hash_idx >= 0 && hash_idx < Nx * Ny)
				{
					int p = 0;
					while (p < max_size_of_value_list && Hashtable[hash_idx][p] != -1)
					{

						if (utility.dist(positionX[Hashtable[hash_idx][p]],
										 positionY[Hashtable[hash_idx][p]],
										 positionX[j * nx + i], positionY[j * nx + i]) <= h)
						{
							if (neighborhood_idx >= max_size_of_particle_neighborhood)
							{
								printf("Consider expanding the maximum size of a particle neighbohood.\n");
							}
							else
							{
								Particle_Neighborhood[j * nx + i][neighborhood_idx] = Hashtable[hash_idx][p];
								neighborhood_idx++;
							}
						}
						p++;
					}
				}
			}
		}
	}
}

void Fluid::testNeighborSearch()
{
	printf("Particle neighborhood of %d: ", (nx * ny / 2));
	for (int k = 0; k < max_size_of_particle_neighborhood / 10; k++)
	{
		printf("%d ", Particle_Neighborhood[(nx * ny) / 2][k]);
	}
	printf("\n");
}

void Fluid::updateForcesAndAcceleration()
{
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			int particle_idx = j * nx + i;
			// calculate density
			updateDensity(particle_idx);
			// calculate pressure
			updatePressure(particle_idx);
			 // update body force (aka gravity)
			vector<double> bodyForce(2,0.0f);
			updateBodyForce(particle_idx, bodyForce);
			// update pressure and viscocity forces
			vector<double> pressureViscocityForces(2,0.0f);
			updatePressureViscocityForces(particle_idx, pressureViscocityForces);
			// update penalty force
			vector<double> penaltyForce(2,0.0f);
			updatePenaltyForce(particle_idx, penaltyForce);
			// update acceleration      
			accX[particle_idx] = (bodyForce[0] + pressureViscocityForces[0] + penaltyForce[0]) /
																			density[particle_idx];
			accY[particle_idx] = (bodyForce[1] + pressureViscocityForces[1] + penaltyForce[1]) /
																			density[particle_idx];
		}
	}
}

void Fluid::eulerIntegration(double dt)
{
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			int particle_idx = j * nx + i;
			velocityX[particle_idx] += accX[particle_idx] * dt;
			velocityY[particle_idx] += accY[particle_idx] * dt;

			positionX[particle_idx] += velocityX[particle_idx] * dt;
			positionY[particle_idx] += velocityY[particle_idx] * dt;
		}
	}
}

void Fluid::freeAndReallocateHashtable()
{
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			free(Hashtable[j * Nx + i]);
		}
	}
	free(Hashtable); Hashtable = NULL;
	Hashtable = new int* [Nx * Ny];
}

void Fluid::getContainerPosition(int idx, glm::vec3& pos, int num_points)
{
	// creates circle
	double r = max(nx * dx * 0.71f, ny * dy * 0.71f);
	double cx = nx * dx * 0.5 - 0.1;
	double cy = ny * dy * 0.5 - 0.1;
	float theta = idx * (2 * pi * (1/num_points)) - pi;
	float x = r * cosf(theta);
	float y = r * sinf(theta);

	pos.x = cx+x; pos.y = cy+y; pos.z=0.0f;
}

//////////////////////////////// END OF PUBLIC METHODS ////////////////////////////////////

////////////////////////////////////// PRIVATE METHODS ////////////////////////////////////

void Fluid::updateDensity(unsigned int particle_idx)
{
	density[particle_idx] = 0.0f;
	for (int k=0; Particle_Neighborhood[particle_idx][k] != -1 &&
									k < max_size_of_particle_neighborhood; k++)
	{
		density[particle_idx] +=
			mass[particle_idx] *
			utility.Ws(positionX[particle_idx] - positionX[Particle_Neighborhood[particle_idx][k]],
				positionY[particle_idx] - positionY[Particle_Neighborhood[particle_idx][k]]);
	}
	if (density[particle_idx] == 0)
	{
		//printf("PROBLEM: zero density!\n");
	}
}

void Fluid::updatePressure(unsigned int particle_idx)
{
	pressure[particle_idx] = pres_dens_coef * (density[particle_idx] - rest_density);
}

void Fluid::updateBodyForce(unsigned int particle_idx, vector<double>& bodyForce)
{
	double body_forceX = density[particle_idx] * gX;
	double body_forceY = density[particle_idx] * gY;
	bodyForce[0] = body_forceX;
	bodyForce[1] = body_forceY;
}

void Fluid::updatePressureViscocityForces(unsigned int particle_idx, vector<double>& pressureViscocityForces) {
	double pressureForceX = 0.0f, pressureForceY = 0.0f;
	double viscocityForceX = 0.0f, viscocityForceY = 0.0f;
	for (int k = 0; Particle_Neighborhood[particle_idx][k] != -1 &&
		k < max_size_of_particle_neighborhood; k++)
	{
		// calculate pressure force
		vector<double> gradWsv; gradWsv.push_back(0.0f); gradWsv.push_back(0.0f);
		utility.gradWs(positionX[particle_idx] -
			positionX[Particle_Neighborhood[particle_idx][k]],
			positionY[particle_idx] -
			positionY[Particle_Neighborhood[particle_idx][k]], gradWsv);
		pressureForceX += (((pressure[particle_idx] +
			pressure[Particle_Neighborhood[particle_idx][k]]) / 2.0f) *
			(mass[Particle_Neighborhood[particle_idx][k]] /
				density[Particle_Neighborhood[particle_idx][k]]) *
			gradWsv[0] * (-1.0f));
		pressureForceY += (((pressure[particle_idx] +
			pressure[Particle_Neighborhood[particle_idx][k]]) / 2) *
			(mass[Particle_Neighborhood[particle_idx][k]] /
				density[Particle_Neighborhood[particle_idx][k]]) *
			gradWsv[1] * (-1.0f));
		// calculate viscocity force
		double laplacianWvv = utility.laplacianWv(positionX[particle_idx] -
			positionX[Particle_Neighborhood[particle_idx][k]],
			positionY[particle_idx] -
			positionY[Particle_Neighborhood[particle_idx][k]]);
		viscocityForceX += (viscocity_coeff * (-1.0f) * (velocityX[particle_idx] -
			velocityX[Particle_Neighborhood[particle_idx][k]])
			* (mass[Particle_Neighborhood[particle_idx][k]] /
				density[Particle_Neighborhood[particle_idx][k]])
			* laplacianWvv);
		viscocityForceY += (viscocity_coeff * (-1.0f) * (velocityY[particle_idx] -
			velocityY[Particle_Neighborhood[particle_idx][k]])
			* (mass[Particle_Neighborhood[particle_idx][k]] /
				density[Particle_Neighborhood[particle_idx][k]])
			* laplacianWvv);
	}
	pressureViscocityForces[0] = pressureForceX + viscocityForceX;
	pressureViscocityForces[1] = pressureForceY + viscocityForceY;
}

void Fluid::updatePenaltyForce(unsigned int particle_idx, vector<double>& penaltyForce) {
	double penaltyForceX = 0.0f, penaltyForceY = 0.0f;
	// calculate penalty force
	vector<double> PhiOuterNormalv;
	PhiOuterNormalv.push_back(0.0f); PhiOuterNormalv.push_back(0.0f);
	utility.PhiOuterNormal(positionX[particle_idx], positionY[particle_idx], PhiOuterNormalv);
	if (utility.PhiOuter(positionX[particle_idx], positionY[particle_idx]) < radius[particle_idx])
	{
		penaltyForceX = stiffness_coeff * PhiOuterNormalv[0]
			* (radius[particle_idx] -
				utility.PhiOuter(positionX[particle_idx],
					positionY[particle_idx]))
			* density[particle_idx];
		penaltyForceY = stiffness_coeff * PhiOuterNormalv[1]
			* (radius[particle_idx] -
				utility.PhiOuter(positionX[particle_idx],
					positionY[particle_idx]))
			* density[particle_idx];
	}
	// PHI-INNER GOES HERE

	penaltyForce[0] = penaltyForceX;
	penaltyForce[1] = penaltyForceY;
}


///////////////////////////////// END OF PRIVATE METHODS ///////////////////////////////