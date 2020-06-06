#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include "fluid_util.h"

class Fluid
{
private:
    double *positionX, *positionY;
    double *velocityX, *velocityY;
    double *accX, *accY;
    double *density, *pressure, *radius, *mass;

    unsigned int nx, ny;
    unsigned int Nx, Ny;
    double dx, dy;
    double h;

    unsigned int max_size_of_value_list = 50;
    int **Hashtable;

    unsigned int max_size_of_particle_neighborhood;
    int **Particle_Neighborhood;

    glm::vec3 *particlePositions;
    
    fluid_util utility;

    void updateDensity(unsigned int);
    void updatePressure(unsigned int);
    void updateBodyForce(unsigned int, vector<double>& );
    void updatePressureViscocityForces(unsigned int, vector<double>& );
    void updatePenaltyForce(unsigned int, vector<double>& );
    
public:
    Fluid(unsigned int _nx, unsigned int _ny, double m, double _dx, double _dy, double _h);
    void spatialHashTableInit();
    void initializeSpatialHashtable();
    void testSpatialHashtable();
    void particleNeighborhoodsInit();
    void generateGLMParticles();
    glm::vec3 getPosition(unsigned int idx);
    void neighborSearch();
    void testNeighborSearch();
    void updateForcesAndAcceleration();
    void eulerIntegration(double dt);
    void freeAndReallocateHashtable();
    void getContainerPosition(int, glm::vec3&, int);
};