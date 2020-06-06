#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include "shader.h"
#include "Camera.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace std;

// Constants to be used
///////////////////////////////////////////////////////
#define nx 20 // number of particles in x dimension
#define ny 20 // number of particles in y dimension
#define dx 0.3f // initial spacing between particles 
#define dy 0.3f // initial spacing between particles

#define h 0.3f // SPH radius
#define m 0.3f // mass
#define pres_dens_coef 10.0f // pressure-density coefficient
#define rest_density 10.0f // rest density parameter
#define gX 0.0f// gravity X
#define gY -1.0f// gravity Y
#define viscocity_coeff 10.0f // viscocity coefficient
#define stiffness_coeff 100.0f // stiffness for environmental respeonse
#define num_of_timesteps 219 // number of timesteps
#define dt 0.02f // timestep duration
#define pi 3.14159 // pi

// settings
const unsigned int SCR_WIDTH = 2400;
const unsigned int SCR_HEIGHT = 1600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// to access the 9 neighbors of a cell in spatial hashing
int neighborsX[9] = {0,1,-1,0,0,1,-1,1,-1};
int neighborsY[9] = {0,0,0,1,-1,1,-1,-1,1};

#define B 15 // number of blocks (on one dimension)

///////////////////////////////////////////////////////

// function declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);


__host__ void initialize_spatial_hashtable_flat ( double *positionX, double *positionY,
                                                  int *Hashtable, int Nx, int Ny,
                                                  int max_size_of_value_list);
__global__ void euler_integration_kernel(double *accX, double *accY,
                                                    double *positionX, double *positionY,
                                                    double *velocityX, double *velocityY,
                                                    int *Hashtable,
                                                    int max_size_of_value_list,
                                                    int Nx, int Ny);
__global__ void force_updates_kernel( double *mass, double *radius, 
                                      double *positionX, double *positionY,
                                      double *velocityX, double *velocityY,
                                      double *accX, double *accY,
                                      double *density, double *pressure,
                                      int *Particle_Neighborhood,
                                      int max_size_of_particle_neighborhood);
__global__ void density_pressure_update_kernel (double *mass,
                                        double *positionX, double *positionY,
                                        double *density, double *pressure, 
                                        int *Particle_Neighborhood,
                                        int max_size_of_particle_neighborhood);
__global__ void neighbor_search_kernel( int *Hashtable, int Nx, int Ny,
                                          double *positionX, double *positionY, 
                                          int *Particle_Neighborhood,
                                          int max_size_of_value_list,
                                          int max_size_of_particle_neighborhood);
__device__ void PhiInnerNormal_d(double x, double y, double *PhiInnerNormalv);
__host__ __device__ double PhiInner(double x, double y);
__device__ void PhiOuterNormal_d (double x, double y, double *PhiOuterNormalv);
__host__ __device__ double PhiOuter(double x, double y);
__host__ __device__ double laplacianWv (double x, double y);
__host__ __device__ double Wv (double x, double y);
__device__ void gradWs_d (double x, double y, double *gradWsv);
__host__ __device__ double Ws (double x, double y);
__host__ __device__ double dist(double x1, double y1, double x2, double y2);

int main() {
  // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader program
    // ------------------------------------
    Shader particleShader("fluidSimulatorShader.vs", "fluidSimulatorShader.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
      -0.5f, -0.5f, 0.0f, // left  
      0.5f, -0.5f, 0.0f, // right 
      0.0f,  0.5f, 0.0f  // top
    };

    unsigned int VBOs[2], VAOs[2];
    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs);

    glBindVertexArray(VAOs[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

// Initialize particles array on CPU
  //  Transfer this memory to GPU
  //////////////////////////////////////////
  double *mass = new double[nx * ny];
  double *positionX = new double[nx * ny];
  double *positionY = new double[nx * ny];
  double *velocityX = new double[nx * ny];
  double *velocityY = new double[nx * ny];
  double *accX = new double[nx * ny];
  double *accY = new double[nx * ny];
	double *density = new double[nx * ny];
	double *pressure = new double[nx * ny];
	double *radius = new double[nx * ny]; 
  for(int j=0; j<ny; j++)
  {
    for (int i=0; i<ny; i++)
    {
      mass[j*nx+i] = m;
      positionX[j*nx+i] = i*dx;
      positionY[j*nx+i] = j*dy;
      velocityX[j*nx+i] = 0.0f;
      velocityY[j*nx+i] = 0.0f;
      accX[j*nx+i] = 0.0f;
      accY[j*nx+i] = 0.0f;
      density[j*nx+i] = 1.0f;
      pressure[j*nx+i] = 0.0f;
      radius[j*nx+i] = 0.1f; 
    }
  }
    cout << "Initial position: " << positionX[(ny/2)*nx + (nx/2)] << ' ' << positionY[(ny/2)*nx + (nx/2)] << endl;
  double *mass_d= 0; cudaMalloc((void **)&mass_d, nx*ny*sizeof(double));
  double *positionX_d = 0; cudaMalloc((void **)&positionX_d, nx*ny*sizeof(double));
  double *positionY_d = 0; cudaMalloc((void **)&positionY_d, nx*ny*sizeof(double));
  double *radius_d = 0; cudaMalloc((void **)&radius_d, nx*ny*sizeof(double));
  double *velocityX_d = 0; cudaMalloc((void **)&velocityX_d, nx*ny*sizeof(double));
  double *velocityY_d = 0; cudaMalloc((void **)&velocityY_d, nx*ny*sizeof(double));
  double *accX_d = 0; cudaMalloc((void **)&accX_d, nx*ny*sizeof(double));
  double *accY_d = 0; cudaMalloc((void **)&accY_d, nx*ny*sizeof(double));
  double *density_d = 0; cudaMalloc((void **)&density_d, nx*ny*sizeof(double));
  double *pressure_d = 0; cudaMalloc((void **)&pressure_d, nx*ny*sizeof(double));
  cudaMemcpy(mass_d, mass, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(positionX_d, positionX, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(positionY_d, positionY, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(radius_d, radius, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(velocityX_d, velocityX, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(velocityY_d, velocityY, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(accX_d, accX, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(accY_d, accY, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(density_d, density, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(pressure_d, pressure, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
  /////////////////////////////////////////

// Initialize spatial hashtable on GPU
  // Call non-kernel function to populate this array
  /////////////////////////////////////////////////////////////////////////////////////////////
  int Nx = (int)((nx * dx+2*2.21*nx*dx)/h)+1;
  int Ny = (int)((ny * dy+2*2.21*ny*dy)/h)+1;
  int max_size_of_value_list = 200;
  int *Hashtable = new int [Nx*Ny*max_size_of_value_list];
  #ifdef DEBUG
    cout << "Nx = " << Nx << ", Ny = " << Ny << endl;
    cout << "max_size_of_value_list = " << max_size_of_value_list << endl;
  #endif
  for(int j=0; j<Ny; j++)
  {
    for(int i=0; i<Nx; i++)
    {
      for (int k=0; k<max_size_of_value_list; k++)
      {
        Hashtable[(j*Nx+i)*max_size_of_value_list+k] = -1;
      }
    }
  }
  #ifdef DEBUG
    cout << "Spatial hashtable (initialized): ";
    for(int i=0; i<max_size_of_value_list; i++)
    {
      cout << Hashtable[((Nx*Ny)/2)*max_size_of_value_list+i] << " ";
    }
    cout << endl;
  #endif
  initialize_spatial_hashtable_flat(positionX, positionY, 
                                    Hashtable, Nx, Ny, max_size_of_value_list);
  #ifdef DEBUG
    cout << "Spatial hashtable (populated): ";
    for(int i=0; i<max_size_of_value_list; i++)
    {
      cout << Hashtable[((Nx*Ny)/2)*max_size_of_value_list+i] << " ";
    }
    cout << endl;
  #endif
  int *Hashtable_d=0; cudaMalloc((void **)&Hashtable_d, Nx*Ny*max_size_of_value_list*sizeof(int));
  cudaMemcpy(Hashtable_d, Hashtable, Nx*Ny*max_size_of_value_list*sizeof(int),
                                      cudaMemcpyHostToDevice);
  #ifdef DEBUG
    cudaMemcpy(Hashtable, Hashtable_d, Nx*Ny*max_size_of_value_list*sizeof(int),
                                      cudaMemcpyDeviceToHost);
    cout << "Checking if hashtable populated correctly on GPU: ";
    for(int i=0; i<max_size_of_value_list; i++)
      {
        cout << Hashtable[((Nx*Ny)/2)*max_size_of_value_list+i] << " ";
      }
      cout << endl;
  #endif
  /////////////////////////////////////////////////////////////////////////////////////////////

// Initialize particle neighborhood array on GPU
  ////////////////////////////////////////////////////
  int max_size_of_particle_neighborhood = 10*max_size_of_value_list;
  int *Particle_Neighborhood = new int[nx * ny * max_size_of_particle_neighborhood];
  for(int j=0; j<ny; j++)
  {
    for(int i=0; i<nx; i++)
    {
      for(int k=0; k<max_size_of_particle_neighborhood; k++)
      {
        Particle_Neighborhood[(j*nx + i)*max_size_of_particle_neighborhood+k] = -1;
      }
    }
  }
  int *Particle_Neighborhood_d = 0;
  cudaMalloc((void **)&Particle_Neighborhood_d,
                  nx*ny*max_size_of_particle_neighborhood*sizeof(int));
  cudaMemcpy(Particle_Neighborhood_d, Particle_Neighborhood, 
                  nx*ny*max_size_of_particle_neighborhood*sizeof(int), cudaMemcpyHostToDevice);
  #ifdef DEBUG
    cudaMemcpy(Particle_Neighborhood, Particle_Neighborhood_d, 
      nx*ny*max_size_of_particle_neighborhood*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Checking if particle neighborhoods initialized correctly on gpu: ");
    for(int i=0; i<max_size_of_particle_neighborhood/10; i++)
      {
        cout << Particle_Neighborhood[((nx*ny)/2)*max_size_of_particle_neighborhood+i] << " ";
      }
      cout << endl;
  #endif
  /////////////////////////////////////////////////////
  cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
  cudaEventRecord(start);
  
// main timestep loop
  while (!glfwWindowShouldClose(window))
  {
    ///////////////////////////////// rendering /////////////////////////////////

    // input
    processInput(window);

    // render background
    // ------------------------------------------------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //////////////// render particles ///////////////////////

    // activate shader
    particleShader.use();

    // pass projection matrix to shader (note that in this case it could change every frame)
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    particleShader.setMat4("projection", projection);

    // camera/view transformation
    glm::mat4 view = camera.GetViewMatrix();
    particleShader.setMat4("view", view);

    // render boxes
    glBindVertexArray(VAOs[0]);
    for (unsigned int i = 0; i < nx * ny; i++)
    {
        // calculate the model matrix for each object and pass it to shader before drawing
        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::translate(model, glm::vec3(positionX[i], positionY[i], 0.0f));
        model = glm::scale(model, glm::vec3(0.05, 0.05, 1));
        particleShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    //////////////// done rendering particles /////////////////////////

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
// for debugging
  #ifdef DEBUG
    cout << "BEFORE" << endl;
    if(t==num_of_timesteps/2) {
      cudaMemcpy(density, density_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(pressure, pressure_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
      cout << "density: " << density[(nx*ny)/2] << endl;
      cout << "pressure: " << pressure[(nx*ny)/2] << endl;
    }
    cudaMemcpy(positionX, positionX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(positionY, positionY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocityX, velocityX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocityY, velocityY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(accX, accX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(accY, accY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cout << "acceleration: " << accX[(nx*ny)/2] << " " << accY[(nx*ny)/2] << endl;
    cout << "position: " << positionX[(nx*ny)/2] << " " << positionY[(nx*ny)/2] << endl;
    cout << "velocity: " << velocityX[(nx*ny)/2] << " " << velocityY[(nx*ny)/2] << endl;
  #endif
// Call kernel for neighbor search
    //////////////////////////////////////////
    neighbor_search_kernel<<<dim3(B,B), dim3(nx/B, ny/B)>>>(Hashtable_d, Nx, Ny,
                                                              positionX_d, positionY_d,
                                                              Particle_Neighborhood_d,
                                                              max_size_of_value_list,
                                                              max_size_of_particle_neighborhood);
    #ifdef DEBUG
      if(t==num_of_timesteps/2)
      {
        cudaMemcpy(Particle_Neighborhood, Particle_Neighborhood_d, 
          nx*ny*max_size_of_particle_neighborhood*sizeof(int), cudaMemcpyDeviceToHost);
        printf("Checking if particle neighborhoods found correctly on gpu: ");
        for(int i=0; i<max_size_of_particle_neighborhood/10; i++)
        {
          cout << Particle_Neighborhood[((nx*ny)/2)*max_size_of_particle_neighborhood+i] << " ";
        }
        cout << endl;
      }
    #endif
    //////////////////////////////////////////

// Call kernel for density + pressure updates
    ////////////////////////////////////////////////////////////////////////////////////////////
    density_pressure_update_kernel<<<dim3(B,B), dim3(nx/B, ny/B)>>> (mass_d, positionX_d,         
                                                                     positionY_d,
                                                                     density_d, pressure_d,
                                                                     Particle_Neighborhood_d,
                                                                 max_size_of_particle_neighborhood);
    ////////////////////////////////////////////////////////////////////////////////////////////


// Call kernel for force + acceleration updates
    /////////////////////////////////////////////////////
    force_updates_kernel<<<dim3(B,B), dim3(nx/B, ny/B)>>> (mass_d, radius_d, 
                                                           positionX_d, positionY_d,
                                                           velocityX_d, velocityY_d,
                                                           accX_d, accY_d,
                                                           density_d, pressure_d,
                                                           Particle_Neighborhood_d,
                                                           max_size_of_particle_neighborhood);
    /////////////////////////////////////////////////////

// Call Euler integration + spatial hashtable update kernel
    /////////////////////////////////////////////////////////////////
    euler_integration_kernel<<<dim3(B,B), dim3(nx/B, ny/B)>>> (accX_d, accY_d,
                                                                positionX_d, positionY_d,
                                                                velocityX_d, velocityY_d,
                                                                Hashtable_d,
                                                                max_size_of_value_list,
                                                                Nx, Ny);
    /////////////////////////////////////////////////////////////////
// Update spatial hashtable due to shifts that may have happened before
    cudaMemcpy(positionX, positionX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(positionY, positionY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    // easy and simple
    free(Hashtable); Hashtable = NULL; // free some memory
    #ifdef DEBUG
      printf("Finished freeing old hashtable\n");
    #endif
    Hashtable = new int[Nx*Ny*max_size_of_value_list];
    for(int j=0; j<Ny; j++)
    {
      for(int i=0; i<Nx; i++)
      {
        for (int k=0; k<max_size_of_value_list; k++)
        {
          Hashtable[(j*Nx+i)*max_size_of_value_list+k] = -1;
        }
      }
    }
    initialize_spatial_hashtable_flat(positionX, positionY, Hashtable, Nx, Ny, 
                                            max_size_of_value_list);
    cudaMemcpy(Hashtable_d, Hashtable, Nx*Ny*max_size_of_value_list*sizeof(int),
                                      cudaMemcpyHostToDevice);
// for debugging
    #ifdef DEBUG
    cout << "AFTER: " << endl;
    if(t==num_of_timesteps/2) {
      cudaMemcpy(density, density_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(pressure, pressure_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
      cout << "density: " << density[(nx*ny)/2] << endl;
      cout << "pressure: " << pressure[(nx*ny)/2] << endl;
    }
    cudaMemcpy(positionX, positionX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(positionY, positionY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocityX, velocityX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocityY, velocityY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(accX, accX_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(accY, accY_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    cout << "acceleration: " << accX[(nx*ny)/2] << " " << accY[(nx*ny)/2] << endl;
    cout << "position: " << positionX[(nx*ny)/2] << " " << positionY[(nx*ny)/2] << endl;
    cout << "velocity: " << velocityX[(nx*ny)/2] << " " << velocityY[(nx*ny)/2] << endl;
    #endif

  }
  cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
  cout << "Final position: " << positionX[(ny/2)*nx + (nx/2)] << ' ' <<
                                          positionY[(ny/2)*nx + (nx/2)] << endl;
// deletes
  delete mass;
  delete positionX;
  delete positionY;
  delete velocityX;
  delete velocityY;
  delete accX;
  delete accY;
  delete density;
  delete pressure;
  delete radius;
  delete Hashtable;
  delete Particle_Neighborhood;
  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(2, VAOs);
  glDeleteBuffers(2, VBOs);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
}

__host__ __device__ double dist(double x1, double y1, double x2, double y2)
{
  return sqrt(((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)));
}

__host__ __device__ double Ws (double x, double y){
// Spiky kernel function
  //////////////////////////////////////
    double r = sqrt(x*x + y*y);
    if( r>=0 && r<=h )
    {
      return 15.0 / (pi*pow(h,6)) * pow(h-r,3);
    }
		return 0.0f;

  //////////////////////////////////////
}

__device__ void gradWs_d (double x, double y, double *gradWsv)
{
// Spiky kernel gradient function
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double r = sqrt(x*x + y*y);
    if( r <= h && r > 0 )
    {
      gradWsv[0] = -45.0/(pi*pow(h,6))*pow(h-r,2)*x/r; 
      gradWsv[1] = -45.0/(pi*pow(h,6))*pow(h-r,2)*y/r;
    }
    else
    {
      gradWsv[0] = 0.0f;
      gradWsv[1] = 0.0f;
    }
  ////////////////////////////////////////////////////////////////////////////////////////////////
}

__host__ __device__ double Wv (double x, double y){
// Viscocity kernel function
  ////////////////////////////////////////
  double r = sqrt(x*x + y*y);
  if( r >= 0 && r <= h )
  {
    return 15.0/(2*pi*pow(h,3))*((-pow(r,3)/(2*pow(h,3))+r*r/(h*h)+h/(2*r)-1));
  }
  return 0.0f;
  ////////////////////////////////////////
}

__host__ __device__ double laplacianWv (double x, double y)
{
// Viscocity kernel laplacian function
  //////////////////////////////////////////////////////////////////////////////////////////////
  double r = sqrt(x*x + y*y);
  if( r <= h && r > 0 )
  {
    return 45.0/(pi*pow(h,6))*(h-r);
  }
	return 0.0f;
  ////////////////////////////////////////////////////////////////////////////////////////////
}

__host__ __device__ double PhiOuter(double x, double y)
{
// Phi function for outer container
  ///////////////////////////////////////////////////////////////////////////////////////////////
  double r = max(nx*dx*0.71f, ny*dy*0.71f);
  double cx = nx*dx*0.5-0.1;
  double cy = ny*dy*0.5-0.1;
  return r-sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
  /////////////////////////////////////////////////////////////////////////////////////////////////
}

__device__ void PhiOuterNormal_d (double x, double y, double *PhiOuterNormalv)
{
// Phi function for outer normal
  ///////////////////////////////////////////////////////////////////////////////////////////////
  double cx = nx*dx*0.5-0.1;
  double cy = ny*dy*0.5-0.1;
  PhiOuterNormalv[0] = (cx-x)/sqrt((cx-x)*(cx-x)+(cy-y)*(cy-y)); 
  PhiOuterNormalv[1] = (cy-y)/sqrt((cx-x)*(cx-x)+(cy-y)*(cy-y));
  ///////////////////////////////////////////////////////////////////////////////////////////////
}

__host__ __device__ double PhiInner(double x, double y)
{
// Phi function for inner object
  /////////////////////////////////////////////////////////////////////////////////////////////////
  double r = max(nx*dx*0.2f, ny*dy*0.2f);
  double cx = nx * dx * 0.5;
  double cy = ny * dy * 0.5;
  return r-sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
  ////////////////////////////////////////////////////////////////////////////////////////////////
}

__device__ void PhiInnerNormal_d(double x, double y, double *PhiInnerNormalv)
{
// Phi function for inner normal
  //////////////////////////////////////////////////////////////////////////////////////////////
  double cx = nx*dx*0.7;
  double cy = ny*dy*0.7;
  PhiInnerNormalv[0] = (x-cx)/sqrt((cx-x)*(cx-x)+(cy-y)*(cy-y)); 
  PhiInnerNormalv[1] = (y-cy)/sqrt((cx-x)*(cx-x)+(cy-y)*(cy-y)); 
  ////////////////////////////////////////////////////////////////////////////////////////////
}

__global__ void neighbor_search_kernel( int *Hashtable, int Nx, int Ny,
                                        double *positionX, double *positionY, 
                                        int *Particle_Neighborhood,
                                        int max_size_of_value_list,
                                        int max_size_of_particle_neighborhood)
{
	/*
		    For the current particle, locate its h-neighborhood
			  using the Hashtable.
			  Write results in array PN[]
  */
  /////////////////////////////////////////////////////////////
  int neighborsX_d[9] = {0,1,-1,0,0,1,-1,1,-1};
  int neighborsY_d[9] = {0,0,0,1,-1,1,-1,-1,1};
  int particle_id=(gridDim.x * blockDim.x) * (blockDim.y * blockIdx.y + threadIdx.y) + 
                                                          blockIdx.x * blockDim.x + threadIdx.x;
  int cx = (int)((positionX[particle_id] + 2.21*nx*dx)/h);
  int cy = (int)((positionY[particle_id] + 2.21*ny*dy)/h);
  int neighborhood_idx = 0;
  for(int k=0; k<9; k++)
  {
    int cell_x = cx+neighborsX_d[k], cell_y=cy+neighborsY_d[k];
    int hash_idx = cell_y*Nx + cell_x;
    if(hash_idx >= 0 && hash_idx < Nx*Ny)
    {
      int p=0;
      while(p<max_size_of_value_list && Hashtable[hash_idx*max_size_of_value_list+p] != -1)
      {
        if (dist (  positionX[Hashtable[hash_idx*max_size_of_value_list+p]], 
                    positionY[Hashtable[hash_idx*max_size_of_value_list+p]],
                    positionX[particle_id], positionY[particle_id]  ) <= h)
        {
          if(neighborhood_idx >= max_size_of_particle_neighborhood)
          {
            printf("Consider expanding the maximum size of a particle neighbohood.\n");
          }
          else 
          {
            Particle_Neighborhood[particle_id*max_size_of_particle_neighborhood+neighborhood_idx] 
                                  = Hashtable[hash_idx*max_size_of_value_list+p];
            neighborhood_idx++;
          }
        }
        p++;
      }
    }
  }
  /////////////////////////////////////////////////////////////
}

__global__ void density_pressure_update_kernel (double *mass,
                                                double *positionX, double *positionY,
                                                double *density, double *pressure, 
                                                int *Particle_Neighborhood,
                                                int max_size_of_particle_neighborhood)
{
	/*
		TODO: For this particle, update its density and pressure
			  using its neighboring particles.
  */
  /////////////////////////////////////////////////////////////
  int particle_idx=(gridDim.x * blockDim.x) * (blockDim.y * blockIdx.y + threadIdx.y) + 
                                            blockIdx.x * blockDim.x + threadIdx.x;
  // calculate density
  density[particle_idx]=0.0f;
  for(int k=0; Particle_Neighborhood[particle_idx*max_size_of_particle_neighborhood+k]!=-1 && 
              k<max_size_of_particle_neighborhood; k++)
  {
    density[particle_idx] += 
             mass[particle_idx] * 
             Ws(positionX[particle_idx]-
                positionX[Particle_Neighborhood[particle_idx*max_size_of_particle_neighborhood+k]],
                positionY[particle_idx]-
                positionY[Particle_Neighborhood[particle_idx*max_size_of_particle_neighborhood+k]]);
  }
  // calculate pressure
  pressure[particle_idx] = pres_dens_coef*(density[particle_idx]-rest_density);
  /////////////////////////////////////////////////////////////
}

__global__ void force_updates_kernel( double *mass, double *radius, 
                                      double *positionX, double *positionY,
                                      double *velocityX, double *velocityY,
                                      double *accX, double *accY,
                                      double *density, double *pressure,
                                      int *Particle_Neighborhood,
                                      int max_size_of_particle_neighborhood) 
{
// For this particle, update its acceleration by computing all 
// the forces that act on it.
  /////////////////////////////////////////////////////////////////////
  int particle_idx=(gridDim.x * blockDim.x) * (blockDim.y * blockIdx.y + threadIdx.y) + 
                                            blockIdx.x * blockDim.x + threadIdx.x;
  // update body force (aka gravity)
  double body_forceX=density[particle_idx]*gX;
  double body_forceY=density[particle_idx]*gY;
  double pressureForceX=0.0f, pressureForceY=0.0f;
  double viscocityForceX=0.0f, viscocityForceY=0.0f;
  for(int k=0; Particle_Neighborhood[particle_idx*max_size_of_particle_neighborhood+k]!=-1 && 
                        k<max_size_of_particle_neighborhood; k++)
  {
  int pn_idx = Particle_Neighborhood[particle_idx*max_size_of_particle_neighborhood+k];
  // calculate pressure force
    double gradWsv[2]; gradWsv[0] = gradWsv[1] = 0.0f;
    gradWs_d(positionX[particle_idx]-
            positionX[pn_idx],
            positionY[particle_idx]-
            positionY[pn_idx], gradWsv);
    pressureForceX += (((pressure[particle_idx]+
                                pressure[pn_idx])/2.0f) * 
                                (mass[pn_idx]/
                                density[pn_idx])*
                                gradWsv[0] * (-1.0f));
    pressureForceY += (((pressure[particle_idx]+
                                pressure[pn_idx])/2) * 
                                (mass[pn_idx]/
                                density[pn_idx])*
                                gradWsv[1] * (-1.0f));
  // calculate viscocity force
    double laplacianWvv = laplacianWv(positionX[particle_idx]-
                                    positionX[pn_idx],
                                    positionY[particle_idx]-
                                    positionY[pn_idx]);
    viscocityForceX += (viscocity_coeff * (-1.0f) * (velocityX[particle_idx]-
                                          velocityX[pn_idx])
                                      * (mass[pn_idx]/
                                          density[pn_idx])
                                      * laplacianWvv);
    viscocityForceY += (viscocity_coeff * (-1.0f) * (velocityY[particle_idx]-
                                          velocityY[pn_idx])
                                      * (mass[pn_idx]/
                                          density[pn_idx])
                                      * laplacianWvv);
  }  
  // calculate penalty force
  double penaltyForceX=0.0f, penaltyForceY=0.0f;
  double PhiOuterNormalv[2];
  PhiOuterNormalv[0] = PhiOuterNormalv[1] = 0.0f;
  PhiOuterNormal_d(positionX[particle_idx], positionY[particle_idx], PhiOuterNormalv);
  if (PhiOuter(positionX[particle_idx], positionY[particle_idx]) < radius[particle_idx])
  {
    penaltyForceX += stiffness_coeff * PhiOuterNormalv[0]
                                      * (radius[particle_idx] - 
                                        PhiOuter(positionX[particle_idx],
                                                      positionY[particle_idx]))
                                      * density[particle_idx];
    penaltyForceY += stiffness_coeff * PhiOuterNormalv[1]
                                      * (radius[particle_idx] - 
                                        PhiOuter(positionX[particle_idx],
                                                      positionY[particle_idx]))
                                      * density[particle_idx];
  }  
// update acceleration      
  accX[particle_idx] = (body_forceX+pressureForceX+viscocityForceX+penaltyForceX)/
                                                  density[particle_idx];
  accY[particle_idx]= (body_forceY+pressureForceY+viscocityForceY+penaltyForceY)/
                                                  density[particle_idx];
  /////////////////////////////////////////////////////////////////////
}

__global__ void euler_integration_kernel(double *accX, double *accY,
                                          double *positionX, double *positionY,
                                          double *velocityX, double *velocityY,
                                          int *Hashtable,
                                          int max_size_of_value_list,
                                          int Nx, int Ny)
{
// Update velocity and position of given particle.
  ////////////////////////////////////////////////////////////////////////////
  int particle_idx=(gridDim.x * blockDim.x) * (blockDim.y * blockIdx.y + threadIdx.y) + 
                                            blockIdx.x * blockDim.x + threadIdx.x;
  velocityX[particle_idx] += accX[particle_idx] * dt;
  velocityY[particle_idx] += accY[particle_idx] * dt;
  positionX[particle_idx] += velocityX[particle_idx] * dt;
  positionY[particle_idx] += velocityY[particle_idx] * dt;
}                                                  

__host__ void initialize_spatial_hashtable_flat ( double *positionX, double *positionY,
                                                  int *Hashtable, int Nx, int Ny,
                                                  int max_size_of_value_list)
{
  // For each particle, find which cell it belongs to
  //  and insert its index there!
  /////////////////////////////////////////////////////////////////////////////////////////////s
  for (int j=0; j<ny; j++)
  {
    for (int i=0; i<nx; i++)
    {
      int cx = (int)((positionX[j*nx + i] + 2.21*nx*dx)/h);
      int cy = (int)((positionY[j*nx + i] + 2.21*nx*dx)/h);
      int c = (cy*Nx + cx)*max_size_of_value_list;
      int k=0;
      while(k<max_size_of_value_list && Hashtable[c+k] != -1)
      {
        k++;
      } 
      if (k<max_size_of_value_list) 
      {
        Hashtable[c+k] = j*nx + i;
      }
      else 
      {
        // printf("We have to increase the max_size_of_value_list\n");
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, dt);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, dt);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, dt);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, dt);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}