#pragma once

#include <Basic/HeapObj.h>
//#include <Engine/Primitive/MassSpring.h>
#include <UGM/UGM>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

namespace Ubpa {
	class Simulate : public HeapObj {
	public:
		Simulate(const std::vector<pointf3>& plist,
			const std::vector<unsigned>& elist) {
			edgelist = elist;
			this->positions.resize(plist.size());
			for (int i = 0; i < plist.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					this->positions[i][j] = plist[i][j];
				}
			}
		};
	public:
		static const Ptr<Simulate> New(const std::vector<pointf3>& plist,
			const std::vector<unsigned>& elist) {
			return Ubpa::New<Simulate>(plist, elist);
		}
	public:
		// clear cache data
		void Clear();

		// init cache data (eg. half-edge structure) for Run()
		bool Init();
		//bool Init();

		// call it after Init()
		bool Run();

		const std::vector<pointf3>& GetPositions() const { return positions; };

		const float GetStiff() { return stiff; };
		void SetStiff(float k) { stiff = k; Init(); };
		const float GetTimeStep() { return h; };
		void SetTimeStep(float k) { h = k; Init(); };
		std::vector<unsigned>& GetFix() { return this->fixed_id; };
		void SetFix(const std::vector<unsigned>& f) { this->fixed_id = f; Init(); };
		const std::vector<vecf3>& GetVelocity() { return velocity; };
		//void SetVelocity(const std::vector<pointf3>& v) { velocity = v; };
		const float GetAirCof() { return air_cof; };
		void SetAirCof(float k) { air_cof = k; Init(); };

		void SetLeftFix();
		void SetUpFix();
		void SetMethodEuler();
		void SetMethodImpEuler();
		void SetMethodFast() { this->mode = 2; }
		void SetNormal() { this->DEBUG = false; }
		void SetDebug() { this->DEBUG = true; }
		bool Isfixed(size_t x);
		bool DEBUG ;


	private:
		// kernel part of the algorithm
		void SimulateOnce();
		void SimulateOnceEuler();
		void SimulateOnceFast();

	private:
		float h ;  //步长 0.03f
		float stiff;
		std::vector<unsigned> fixed_id;  //fixed point id


		//mesh data
		std::vector<unsigned> edgelist;


		//simulation data
		std::vector<pointf3> positions;
		std::vector<vecf3> velocity;

		std::vector<vecf3> force;

		std::vector<double> prelength;

		std::vector<int> isFixed;

		double acc_g = 9.8;
		float m = 1.0; //mass
		float M = 1;
		float air_cof;
		int mode;

		Eigen::SparseMatrix<double> g_delta;



		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > gd_lu;




		Eigen::MatrixXd xn;
		Eigen::MatrixXd y;
		Eigen::MatrixXd g;
		Eigen::MatrixXd vn;
		Eigen::MatrixXd fext;
		Eigen::MatrixXd fint;






		Eigen::SparseMatrix<double> mhl;
		Eigen::SparseMatrix<double> left;
		Eigen::SparseMatrix<double> K;
		Eigen::SparseMatrix<double> Kt;
		Eigen::SparseMatrix<double> J;
		//Accel method
		Eigen::SparseMatrix<double> A;//系数矩阵
		Eigen::SparseMatrix<double> M_h2L;//M+h*h*L
		Eigen::SparseLU<Eigen::SparseMatrix<double>> A_LU;
		Eigen::MatrixXd b;//x-K*Kt*x

		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > left_cls;

		Eigen::MatrixXd D;

		Eigen::MatrixXd _b;
		Eigen::MatrixXd right;
		Eigen::MatrixXd x_na;//迭代数据x(全部结点)
		Eigen::MatrixXd ya;
		Eigen::MatrixXd xa;//迭代数据x(自由结点)
		Eigen::MatrixXd xr;//迭代数据x(自由结点)


	};
}
