#include "SystemModel.hpp"
#include <kalman/ExtendedKalmanFilter.hpp>

using namespace KalmanExamples;

using T = double;
using State = Lie::State<T>;
// No control
using Control = Kalman::Vector<T, 0>;
using SystemModel = Lie::SystemModel<T>;

int main(int argc, char *argv[])
{
  State x;
  x = State::Identity();
  std::cout << x.matrix().transpose() << std::endl;

  // Extended Kalman Filter
  Kalman::ExtendedKalmanFilter<State> ekf;

  SystemModel sys;
  sys.velocity = State::Zero();
  sys.velocity(4) = .1;
  std::cout << "System velocity: " << sys.velocity.matrix().transpose() << std::endl;

  // Init filters with the true system state
  ekf.init(x);



  Control u;
  State x_ = sys.f(x, u);
  std::cout << "New State: " << x_.matrix().transpose() << std::endl;

  auto x_ekf = ekf.predict(sys, u);
  std::cout << "x_ekf: " << x_ekf.matrix().transpose() << std::endl;

  return 0;
}
