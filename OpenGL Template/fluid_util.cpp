#include "fluid_util.h"
#include <algorithm>

#define pi 3.14

const int fluid_util::neighborsX[] = {0,1,-1,0,0,1,-1,1,-1};
const int fluid_util::neighborsY[] = {0,0,0,1,-1,1,-1,-1,1};

fluid_util::fluid_util(int _nx, int _ny, double _dx, double _dy, double _h)
{
    nx = _nx; ny = _ny;
    dx = _dx; dy = _dy;
    h = _h;
}

double fluid_util::dist(double x1, double y1, double x2, double y2)
{
    return sqrt(((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

double fluid_util::Ws(double x, double y) {
    // Spiky kernel function
      //////////////////////////////////////
    double r = sqrt(x * x + y * y);
    if (r >= 0 && r <= h)
    {
        return 15.0 / (pi * pow(h, 6)) * pow(h - r, 3);
    }
    return 0.0f;

    //////////////////////////////////////
}

void fluid_util::gradWs(double x, double y, vector<double>& gradWsv)
{
    // spiky kernel gradient function
      ////////////////////////////////////////////////////////////////////////////////////////////////
    gradWsv = vector<double>();
    double r = sqrt(x * x + y * y);
    if (r <= h && r > 0)
    {
        gradWsv.push_back(-45.0 / (pi * pow(h, 6)) * pow(h - r, 2) * x / r);
        gradWsv.push_back(-45.0 / (pi * pow(h, 6)) * pow(h - r, 2) * y / r);
    }
    else
    {
        gradWsv.push_back(0.0f);
        gradWsv.push_back(0.0f);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
}

double fluid_util::Wv(double x, double y) {
    // Viscocity kernel function
      ////////////////////////////////////////
    double r = sqrt(x * x + y * y);
    if (r >= 0 && r <= h)
    {
        return 15.0 / (2 * pi * pow(h, 3)) * ((-pow(r, 3) / (2 * pow(h, 3)) + r * r / (h * h) + h / (2 * r) - 1));
    }
    return 0.0f;
    ////////////////////////////////////////
}

double fluid_util::laplacianWv(double x, double y)
{
    // Viscocity kernel laplacian function
      //////////////////////////////////////////////////////////////////////////////////////////////
    double r = sqrt(x * x + y * y);
    if (r <= h && r > 0)
    {
        return 45.0 / (pi * pow(h, 6)) * (h - r);
    }
    return 0.0f;
    ////////////////////////////////////////////////////////////////////////////////////////////
}

double fluid_util::PhiOuter(double x, double y)
{
    // Phi function for outer container
      ///////////////////////////////////////////////////////////////////////////////////////////////
    double r = max(nx * dx * 0.71f, ny * dy * 0.71f);
    double cx = nx * dx * 0.5 - 0.1;
    double cy = ny * dy * 0.5 - 0.1;
    return r - sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
    /////////////////////////////////////////////////////////////////////////////////////////////////
}

void fluid_util::PhiOuterNormal(double x, double y, vector<double>& PhiOuterNormalv)
{
    // Phi function for outer normal
      ///////////////////////////////////////////////////////////////////////////////////////////////
    PhiOuterNormalv = vector<double>();
    double cx = nx * dx * 0.5 - 0.1;
    double cy = ny * dy * 0.5 - 0.1;
    PhiOuterNormalv.push_back((cx - x) / sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y)));
    PhiOuterNormalv.push_back((cy - y) / sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y)));
    ///////////////////////////////////////////////////////////////////////////////////////////////
}

double fluid_util::PhiInner(double x, double y)
{
    // Phi function for inner object
      /////////////////////////////////////////////////////////////////////////////////////////////////
    double r = max(nx * dx * 0.2f, ny * dy * 0.2f);
    double cx = nx * dx * 0.5;
    double cy = ny * dy * 0.5;
    return r - sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
    ////////////////////////////////////////////////////////////////////////////////////////////////
}

void fluid_util::PhiInnerNormal(double x, double y, vector<double>& PhiInnerNormalv)
{
    // Phi function for inner normal
      //////////////////////////////////////////////////////////////////////////////////////////////
    PhiInnerNormalv = vector<double>();
    double cx = nx * dx * 0.7;
    double cy = ny * dy * 0.7;
    PhiInnerNormalv.push_back((x - cx) / sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y)));
    PhiInnerNormalv.push_back((y - cy) / sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y)));
    ////////////////////////////////////////////////////////////////////////////////////////////
}
