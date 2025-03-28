#include <Engine/MeshEdit/Simulate.h>


using namespace Ubpa;

using namespace std;
using namespace Eigen;


void Simulate::Clear() {
	this->positions.clear();
	this->velocity.clear();
}

bool Simulate::Init() {
	//Clear();


	this->velocity.resize(positions.size());
	for (int i = 0; i < positions.size(); i++)
	{

		for (int j = 0; j < 3; j++)
		{
			this->velocity[i][j] = 0;
		}
	}

	this->force.resize(positions.size());
	for (int i = 0; i < positions.size(); i++)
	{

		this->force[i][0] = 0;
		this->force[i][1] = 4.5;
		this->force[i][2] = 9.8;

	}



	this->prelength.resize(edgelist.size() / 2);
	for (int i = 0; i < edgelist.size() / 2; i++)
	{
		size_t x1 = edgelist[2 * i];
		size_t x2 = edgelist[2 * i + 1];
		//vecf3 d = positions[x2] - positions[x1];
		double length = pointf3::distance(positions[x2], positions[x1]);
		prelength[i] = length;
		printf("i=%d x1=%d x2=%d\n", i, int(x1), int(x2));
		std::ostream_iterator<float> it(std::cout, " ");
		std::copy(positions[x1].begin(), positions[x1].end(), it);
		printf("\n");
		std::ostream_iterator<float> iter(std::cout, " ");
		std::copy(positions[x2].begin(), positions[x2].end(), iter);
		printf("\n");
	}
	size_t n = positions.size();
	size_t s = edgelist.size() / 2;




	g_delta.resize(3 * n, 3 * n);

	xn.resize(3 * n, 1);
	x_na.resize(3 * n, 1);
	ya.resize(3 * n, 1);
	y.resize(3 * n, 1);
	g.resize(3 * n, 1);
	vn.resize(3 * n, 1);
	fext.resize(3 * n, 1);
	fint.resize(3 * n, 1);


	J.resize(3 * n, 3 * s);
	D.resize(3 * s, 1);
	size_t f = fixed_id.size();
	K.resize(3 * (n - f), 3 * n);
	K.setZero();
	_b.resize(3 * n, 1);
	_b.setZero();

	int size = this->positions.size();
	int springsize = this->edgelist.size() / 2;
	//Accel
	//Found M+h2*L
	M_h2L.resize(3 * n, 3 * n);
	vector<Triplet<double>> tripletlist;
	for (size_t i = 0; i < springsize; i++)
	{
		auto x1 = edgelist[2 * i];
		auto x2 = edgelist[2 * i + 1];
		for (int j = 0; j < 3; j++)
		{
			tripletlist.push_back(Triplet<double>(3 * x1 + j, 3 * x1 + j, h * h * stiff));
			tripletlist.push_back(Triplet<double>(3 * x1 + j, 3 * x2 + j, -h * h * stiff));
			tripletlist.push_back(Triplet<double>(3 * x2 + j, 3 * x1 + j, -h * h * stiff));
			tripletlist.push_back(Triplet<double>(3 * x2 + j, 3 * x2 + j, h * h * stiff));
		}
	}
	for (size_t i = 0; i < 3 * size; i++)
	{
		tripletlist.push_back(Triplet<double>(i, i, M));
	}
	M_h2L.setFromTriplets(tripletlist.begin(), tripletlist.end());
	//Found J
	J.resize(3 * size, 3 * springsize);
	vector<Triplet<double>> tripletlist2;
	for (size_t i = 0; i < springsize; i++)
	{
		auto x1 = edgelist[2 * i];
		auto x2 = edgelist[2 * i + 1];
		for (int j = 0; j < 3; j++)
		{
			tripletlist2.push_back(Triplet<double>(3 * x1 + j, 3 * i + j, stiff));
			tripletlist2.push_back(Triplet<double>(3 * x2 + j, 3 * i + j, -stiff));
		}
	}
	J.setFromTriplets(tripletlist2.begin(), tripletlist2.end());
	//Found K & b
	b.resize(3 * size, 1);
	b.setZero();

	vector<Triplet<double>> tripletlist3;
	int m_size = size - fixed_id.size();
	K.resize(3 * m_size, 3 * size);
	int j = 0;
	for (size_t i = 0; i < size; i++)
	{
		if (!Isfixed(i))
		{
			for (int k = 0; k < 3; k++)
			{
				tripletlist3.push_back(Triplet<double>(3 * j + k, 3 * i + k, 1.0));
			}
			j++;
		}
		else
		{
			for (int k = 0; k < 3; k++)
			{
				b(3 * i + k, 0) = positions[i][k];
			}
		}
	}
	K.setFromTriplets(tripletlist3.begin(), tripletlist3.end());
	Kt = K.transpose();
	//Coff Matrix A
	if (fixed_id.size() > 0)
	{
		A = K * M_h2L * Kt;
	}
	else
	{
		A = M_h2L;
	}
	A.makeCompressed();
	A_LU.compute(A);

	return true;
}

bool Simulate::Run() {
	if (mode == 0) {

		SimulateOnce();

	}
	else if (mode == 1) {
		//printf("00\n");
		
		SimulateOnceEuler();
		printf("Euler\n");
	}
	else if (mode == 2) {
		
		SimulateOnceFast();
		printf("Fast\n");
	}


	// half-edge structure -> triangle mesh

	return true;
}

void Ubpa::Simulate::SetLeftFix()
{
	//固定网格x坐标最小点
	fixed_id.clear();
	double x = 100000;
	this->isFixed.resize(positions.size());

	for (int i = 0; i < positions.size(); i++)
	{
		isFixed[i] = 0;
	}

	for (int i = 0; i < positions.size(); i++)
	{
		if (positions[i][0] < x)
		{
			x = positions[i][0];
		}
	}

	for (int i = 0; i < positions.size(); i++)
	{

		if (abs(positions[i][0] - x) < 1e-5)
		{
			fixed_id.push_back(i);
			isFixed[i] = 1;
		}
	}

	Init();
	printf("left\n");
}

void Ubpa::Simulate::SetUpFix()
{
	//固定网格x坐标最小点
	fixed_id.clear();
	double y = -100000;

	this->isFixed.resize(positions.size());

	for (int i = 0; i < positions.size(); i++)
	{
		isFixed[i] = 0;
	}

	for (int i = 0; i < positions.size(); i++)
	{
		if (positions[i][1] > y)
		{
			y = positions[i][1];
		}
	}

	for (int i = 0; i < positions.size(); i++)
	{

		if (abs(positions[i][1] - y) < 1e-5)
		{
			fixed_id.push_back(i);
			isFixed[i] = 1;
		}
	}

	Init();
	// K = K.transpose();
	printf("up\n");
}


void Simulate::SimulateOnce() {
	
	//cout << "WARNING::Simulate::SimulateOnce:" << endl;
//		<< "\t" << "not implemented" << endl;



	for (int i = 0; i < positions.size(); i++)
	{
		this->force[i][0] = 0;
		this->force[i][1] = -9.8;		// reset the initial force of each mass point be its gravity 
		this->force[i][2] = 0;
	}
	for (int i = 0; i < edgelist.size() / 2; i++) {
		size_t x1 = edgelist[2 * i];
		size_t x2 = edgelist[2 * i + 1];
		vecf3 d = positions[x2] - positions[x1];
		double length = (positions[x1] - positions[x2]).norm();

		vecf3 fspring = stiff * (length / prelength[i] - 1) * (d / length);


		vecf3 air_f1 = air_cof * velocity[x1];
		vecf3 air_f2 = air_cof * velocity[x2];
		force[x1] = force[x1] + fspring - air_f1;
		force[x2] = force[x2] - fspring - air_f2;

	}

	for (int i = 0; i < positions.size(); i++)
	{
		if (!isFixed[i]) {
			this->velocity[i] += h * force[i] / m;

			this->positions[i][0] += h * this->velocity[i][0];
			this->positions[i][1] += h * this->velocity[i][1];
			this->positions[i][2] += h * this->velocity[i][2];
		}
	}



}

void Simulate::SetMethodEuler() {
	this->mode = 0;
}

void Simulate::SetMethodImpEuler() {
	this->mode = 1;
}

void Simulate::SimulateOnceEuler() {

	xn.setZero();
	y.setZero();
	g.setZero();
	vn.setZero();
	fext.setZero();
	fint.setZero();

	for (int i = 0; i < positions.size(); i++) {
		for (int j = 0; j < 3; j++) {
			xn(3 * i + j, 0) = positions[i][j];
		}
	}
	for (int i = 0; i < positions.size(); i++) {
		for (int j = 0; j < 3; j++) {
			vn(3 * i + j, 0) = velocity[i][j];
		}
	}
	for (int i = 0; i < positions.size(); i++) {

		fext(3 * i, 0) = 0;
		fext(3 * i + 1, 0) = -9.8 / 1.414;
		fext(3 * i + 2, 0) = 0;

	}

	for (int i = 0; i < positions.size(); i++) {
		if (isFixed[i]) {
			for (int j = 0; j < 3; j++) {
				y(3 * i + j) = positions[i][j];
			}
		}
		else {
			for (int j = 0; j < 3; j++) {
				y(3 * i + j) = positions[i][j] + h * velocity[i][j] + h * h * (1 / m) * fext(3 * i + j);
			}
		}
	}
	xn = y;
	fint.setZero();
	int cnt = 0;
	double merror = 0;
	//printf("11\n");

	do {
		//y = xn + h * vn + h * h * (1 / m) * fext;
		g_delta.setZero();
		for (int i = 0; i < edgelist.size() / 2; i++) {
			size_t x1 = edgelist[2 * i];
			size_t x2 = edgelist[2 * i + 1];
			vecf3 d;
			for (int j = 0; j < 3; j++) {
				d[j] = xn(3 * x2 + j, 0) - xn(3 * x1 + j, 0);
			}
			//vecf3 d = positions[x2] - positions[x1];
			double length = d.norm();

			vecf3 fspring = stiff * (length / prelength[i] - 1) * d;//fix pre /length
			for (int j = 0; j < 3; j++) {
				fint(3 * x1 + j, 0) += fspring[j];
				fint(3 * x2 + j, 0) -= fspring[j];
			}

		}
		//printf("12\n");
		g = m * (xn - y) - h * h * fint;
		//printf("13\n");
		for (int i = 0; i < positions.size(); i++) {
			if (isFixed[i]) {
				for (int j = 0; j < 3; j++) {
					g(3 * i + j, 0) = 0;
				}
			}
		}
		//printf("14\n");
		Eigen::MatrixXd fint_d;
		fint_d.resize(3, 3);
		fint_d.setZero();
		Eigen::MatrixXd I;
		I.setIdentity(3, 3);
		std::vector<Eigen::Triplet<double> > coefficients;
		for (int i = 0; i < edgelist.size() / 2; i++) {
			size_t x1 = edgelist[2 * i];
			size_t x2 = edgelist[2 * i + 1];
			//vecf3 r = positions[x1] - positions[x2];
			Eigen::MatrixXd r;
			r.resize(3, 1);
			r.setZero();
			for (int j = 0; j < 3; j++) {
				//r(j) = positions[x1][j] - positions[x2][j];
				r(j) = xn(3 * x1 + j, 0) - xn(3 * x2 + j, 0);
			}
			double rnorm = r.norm();
			fint_d = stiff * (prelength[i] / rnorm - 1) * I - stiff * prelength[i] * (1 / rnorm) * (1 / rnorm) * (1 / rnorm) * r * r.transpose();
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					if (!isFixed[x1]) {
						coefficients.push_back(Eigen::Triplet<double>(3 * x1 + j, 3 * x1 + k, -h * h * fint_d(j, k)));
						coefficients.push_back(Eigen::Triplet<double>(3 * x1 + j, 3 * x2 + k, h * h * fint_d(j, k)));
					}
					if (!isFixed[x2]) {
						coefficients.push_back(Eigen::Triplet<double>(3 * x2 + j, 3 * x1 + k, h * h * fint_d(j, k)));
						coefficients.push_back(Eigen::Triplet<double>(3 * x2 + j, 3 * x2 + k, -h * h * fint_d(j, k)));
					}
				}
			}

		}
		for (int i = 0; i < 3 * positions.size(); i++) {
			coefficients.push_back(Eigen::Triplet<double>(i, i, m));
		}
		//printf("0\n");
		g_delta.setFromTriplets(coefficients.begin(), coefficients.end());
		g_delta.makeCompressed();
		//printf("1\n");
		gd_lu.compute(g_delta);
		//printf("2\n");
		Eigen::MatrixXd solu;
		solu = gd_lu.solve(g);
		//printf("3\n");
		xn = xn - solu;

		for (int i = 0; i < positions.size() * 3; i++) {
			if (merror < fabs(solu(i))) {
				merror = fabs(solu(i));
			}
		}
		cnt++;
	} while (merror > 10e-6 && cnt <= 3);
	//printf("4\n");
	for (int i = 0; i < positions.size(); i++)
	{
		if (!isFixed[i]) {
			for (int j = 0; j < 3; j++) {
				this->velocity[i][j] = (xn(3 * i + j) - positions[i][j]) / h;

				this->positions[i][j] = xn(3 * i + j);

			}
		}
	}
}


void Simulate::SimulateOnceFast() {

	for (int i = 0; i < positions.size(); i++) {

		fext(3 * i, 0) = 0;
		fext(3 * i + 1, 0) = -9.8 / 1.414;
		fext(3 * i + 2, 0) = 0;

	}
	if (DEBUG) cout << "K:" << endl << MatrixXd(K) << endl << endl;
	size_t size = positions.size();
	size_t springsize = edgelist.size() / 2;
	//Get x_n & y
	for (size_t i = 0; i < size; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			x_na(3 * i + j, 0) = positions[i][j];
			ya(3 * i + j, 0) = positions[i][j] + h * velocity[i][j] + h * h / M * fext(3 * i + j);
		}
	}
	if (DEBUG) cout << "x_na:" << endl << MatrixXd(x_na) << endl << endl;
	if (DEBUG) cout << "ya:" << endl << MatrixXd(ya) << endl << endl;

	//Get d
	for (size_t i = 0; i < springsize; i++)
	{
		auto x1 = edgelist[2 * i];
		auto x2 = edgelist[2 * i + 1];
		auto p1 = positions[x1];
		auto p2 = positions[x2];
		double dist = (p1 - p2).norm();
		for (int j = 0; j < 3; j++)
		{
			D(3 * i + j, 0) = prelength[i] / dist * (p1[j] - p2[j]);
		}
	}
	if (DEBUG) cout << "D:" << endl << MatrixXd(D) << endl << endl;

	//Iteration part
	double maxerror = 0;
	int cnt = 0;
	do//
	{
		//Update x
		xr = K * (h * h * J * D + M * ya - M_h2L * b);
		xa = A_LU.solve(xr);

		size_t j = 0;
		for (size_t i = 0; i < size; i++)
		{
			if (!Isfixed(i))
			{
				for (int k = 0; k < 3; k++)
				{
					x_na(3 * i + k, 0) = xa(3 * j + k, 0);
				}
				j++;
			}
		}
		if (DEBUG) cout << "xr:" << endl << MatrixXd(xr) << endl << endl;
		if (DEBUG) cout << "xa:" << endl << MatrixXd(xa) << endl << endl;
		if (DEBUG) cout << "x_na:" << endl << MatrixXd(x_na) << endl << endl;
		if (DEBUG) cout << "D:" << endl << MatrixXd(D) << endl << endl;

		//Update d
		for (size_t i = 0; i < springsize; i++)
		{
			auto x1 = edgelist[2 * i];
			auto x2 = edgelist[2 * i + 1];
			pointf3 p1, p2;
			for (int k = 0; k < 3; k++)
			{
				p1[k] = x_na(3 * x1 + k, 0);
				p2[k] = x_na(3 * x2 + k, 0);
			}
			double dist = (p1 - p2).norm();
			for (int j = 0; j < 3; j++)
			{
				D(3 * i + j, 0) = prelength[i] / dist * (p1[j] - p2[j]);
			}
		}
		cnt++;
	} while (cnt <= 10);

	//Set Vel
	for (int i = 0; i < positions.size(); i++)
	{
		if (!Isfixed(i))
		{
			for (int j = 0; j < 3; j++)
			{
				velocity[i][j] = (x_na(3 * i + j, 0) - positions[i][j]) / h;
				positions[i][j] = x_na(3 * i + j, 0);
			}
		}
	}

}

bool Simulate::Isfixed(size_t x)
{
	for (int i = 0; i < fixed_id.size(); i++)
	{
		if (x == fixed_id[i])
		{
			return(true);
		}
	}
	return(false);
}