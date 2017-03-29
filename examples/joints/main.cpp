#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman_joints/SystemModel.hpp>
#include <kalman_joints/JointPositionMeasurementModel.hpp>

#include <chrono>
#include <fstream>
#include <random>

using T = double;
// Number of joints
constexpr size_t N = 2;
// State size
constexpr size_t Ns = N*2;
using State = Joints::State<T>;
using SystemModel = Joints::SystemModel<T>;
using JointMeasurement = Joints::JointMeasurement<T>;
using JointPositionMeasurementModel = Joints::JointPositionMeasurementModel<T>;

struct Noise
{
    // Random number generation (for noise simulation)
    std::default_random_engine generator;
    std::normal_distribution<T> noise;

    // Standard-Deviation of noise added to all state vector components during state transition
    T systemNoise = 0.;
    // Standard-Deviation of noise added to all pos_measurement vector components
    T measurementNoise = 0.015; // 0.005;

    Noise() : noise(0, 1)
    {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    Eigen::VectorXd addNoise(const Eigen::VectorXd& m)
    {
        Eigen::VectorXd r(m.rows());
        for (size_t i = 0; i < m.rows(); ++i)
        {
            r(i) = m(i) + measurementNoise * noise(generator);
        }
        return r;
    }
};

int main(int argc, char* argv[])
{
    // Extended Kalman Filter
    Kalman::ExtendedKalmanFilter<State> ekf;
    SystemModel sys(N);
    JointPositionMeasurementModel pos_measurement(N);
    Noise noise;

    State x_init(Ns);
    x_init << .1, .2, 1., 1.;
    std::cout << "state: " << x_init.transpose() << std::endl;
    std::cout << "state q: " << x_init.q().transpose() << std::endl;
    std::cout << "state q_dot: " << x_init.q_dot().transpose() << std::endl;
    std::cout << std::endl;

    State ekf_init(Ns);
    ekf_init << 0., 0., 0., 0.;
    std::cout << "ekf init: " << ekf_init.transpose() << std::endl;
    ekf.init(ekf_init);


    // Generate a trajectory
    std::vector<State> traj(20);
    traj[0] = x_init;
    for (size_t i = 1; i < traj.size(); ++i)
    {
        traj[i].resize(Ns);
        traj[i].q(traj[i - 1].q() + x_init.q_dot());
        traj[i].q_dot(x_init.q_dot());
    }

    std::cout << std::endl;
    std::cout << "Simulating a noisy trajectory" << std::endl;
    JointMeasurement x_mes(N);
    for (int i = 1; i < traj.size(); i++)
    {
      State x_ref = traj[i];

      ekf.predict(sys, 1);
      State x_ekf = ekf.getState();
      std::cout << "Pred State: " << x_ekf.q().transpose() << ", " << x_ekf.q_dot().transpose() << std::endl;
      std::cout << "Ref: " << x_ref.q().transpose() << ", " << x_ref.q_dot().transpose() << std::endl;

      if (i % 3 == 0)
      {
          std::cout << "UPDATING POSITION" << std::endl;
          x_mes = noise.addNoise(x_ref.q());
          ekf.update(pos_measurement, x_mes);
      }
    }

    State x_future;
    sys.predict(ekf.getState(), 5, x_future);
    std::cout << "Future (dt=5): " << x_future.transpose() << std::endl;
    return 0;
}
