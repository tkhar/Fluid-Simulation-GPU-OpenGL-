#pragma once
#include<vector>

using namespace std;

class fluid_util 
{
public:
	static const int neighborsX[];
	static const int neighborsY[];

	fluid_util(): nx(5),ny(5),dx(0.35f),dy(0.35f),h(0.8f){};
	fluid_util(int _nx, int _ny, double _dx, double _dy, double _h);
	double dist(double x1, double y1, double x2, double y2);
	double Ws(double x, double y);
	void gradWs(double x, double y, vector<double>& gradWsv);
	double Wv(double x, double y);
	double laplacianWv(double x, double y);
	double PhiOuter(double x, double y);
	void PhiOuterNormal(double x, double y, vector<double>& PhiOuterNormalv);
	double PhiInner(double x, double y);
	void PhiInnerNormal(double x, double y, vector<double>& PhiInnerNormalv);

private:
	unsigned int nx, ny;
	double h;
	double dx,dy;
};