#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Geometry>
#include <iostream>

Eigen::MatrixXd riccati_solver(Eigen::MatrixXd A,
                               Eigen::MatrixXd B,
                               Eigen::MatrixXd Q,
                               Eigen::MatrixXd R) {
  int const n = (int)A.cols();
  auto H      = Eigen::MatrixXd(2 * n, 2 * n);
  H << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();

  Eigen::EigenSolver<Eigen::MatrixXd> s(H);

  int j   = 0;
  auto S1 = Eigen::MatrixXcd(n, n);
  auto S2 = Eigen::MatrixXcd(n, n);
  for (int i = 0; i < 2 * n; i++) {
    if (s.eigenvalues()[i].real() < 0) {
      S1.col(j) = s.eigenvectors().block(0, i, n, 1);
      S2.col(j) = s.eigenvectors().block(n, i, n, 1);
      j++;
    }
  }

  auto P = S2 * S1.inverse();

  return P.real();
}

Eigen::MatrixXd lqr(Eigen::MatrixXd A,
                    Eigen::MatrixXd B,
                    Eigen::MatrixXd Q,
                    Eigen::MatrixXd R) {
  auto P = riccati_solver(A, B, Q, R);
  return R.inverse() * B.transpose() * P;
}

int main(void) {
  Eigen::Matrix<double, 2, 2> A;
  Eigen::Matrix<double, 2, 1> B;
  Eigen::Matrix<double, 2, 2> Q;
  Eigen::Matrix<double, 1, 1> R;
  A << 0, 1, -10, -1;
  B << 0, 1;
  Q << 300, 0, 0, 60;
  R << 1;

  auto K = lqr(A, B, Q, R);
  std::cout << K << std::endl;
  /*
  P: [170 10; 10 8]
  K: [10 8]
  */
}
