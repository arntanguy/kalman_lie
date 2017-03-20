#include "SystemModel.hpp"
#include "LieMeasurementModel.hpp"
#include <kalman/ExtendedKalmanFilter.hpp>

#include <random>
#include <chrono>
#include <fstream>

using namespace KalmanExamples;

using T = double;
using State = Lie::State<T>;
// No control
using Control = Kalman::Vector<T, 0>;
using SystemModel = Lie::SystemModel<T>;
using LieMeasurementModel = Lie::LieMeasurementModel<T>;
using LieMeasurement = Lie::LieMeasurement<T>;

struct Noise
{
  // Random number generation (for noise simulation)
  std::default_random_engine generator;
  std::normal_distribution<T> noise;

  // Standard-Deviation of noise added to all state vector components during state transition
  T systemNoise = 0.1;
  // Standard-Deviation of noise added to all measurement vector components
  T measurementNoise = 0.025;

  Noise() : noise(0,1)
  {
    generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
  }

  LieMeasurement addNoise(const Eigen::VectorXd& m)
  {
    LieMeasurement r;
    for (size_t i = 0; i < m.rows(); ++i) {
      r(i) = m(i) + measurementNoise * noise(generator);
    }
    return r;
  }
};

struct CSVWriter
{
  std::ofstream csv;

  CSVWriter(const std::string& path="data.csv")
  {
    csv.open(path);
  }

  std::ofstream& operator()()
  {
    return csv;
  }

  void write(const std::vector<Eigen::VectorXd>& data)
  {
    for(const auto& x : data)
    {
      for(size_t i=0; i<x.rows()-1; i++)
      {
        csv << x(i) << ";";
      }
      csv << x(x.size()-1);
    }
    csv << std::endl;
  }
};

int main(int argc, char *argv[])
{
  Noise noise;
  CSVWriter csv("data.csv");

  // Pure predictor without measurement update (ie groundtruth)
  Kalman::ExtendedKalmanFilter<State> predictor;

  // Extended Kalman Filter
  Kalman::ExtendedKalmanFilter<State> ekf;
  SystemModel sys;
  LieMeasurementModel measurement;

  // Initalize system model
  sys.velocity = State::Zero();
  sys.velocity(4) = .1;
  std::cout << "System velocity: " << sys.velocity.matrix().transpose() << std::endl;


  // Init filters with the true system state
  State x;
  x = State::Identity();
  predictor.init(x);
  ekf.init(x);

  std::cout << "Initial State: " << x.matrix().transpose() << std::endl;

  csv() << "#x_pred;x_mes;x_ekf" << std::endl;
  for(int i=0; i<5; i++)
  {
    LieMeasurement x_mes;
    // x: sensor state
    // h: maps sensor to state
    x_mes = measurement.h(noise.addNoise(x));
    // x_mes = measurement.h(x);

    Control u;
    x = sys.f(x, u);
    std::cout << "New State: " << x.matrix().transpose() << std::endl;

    ekf.update(measurement, x_mes);

    // Predict state for current time-step using the filters
    auto x_pred = predictor.predict(sys, u);
    auto x_ekf = ekf.predict(sys, u);
    std::cout << "x_pred: " << x_pred.matrix().transpose() << std::endl;
    std::cout << "x_mes: " << x_mes.matrix().transpose() << std::endl;
    std::cout << "x_ekf: " << x_ekf.matrix().transpose() << std::endl;
    csv.write({x_pred, x_mes, x_ekf});
  }

  return 0;
}
