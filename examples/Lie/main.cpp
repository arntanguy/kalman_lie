#include <kalman/ExtendedKalmanFilter.hpp>
#include "LiePositionMeasurementModel.hpp"
#include "LieVelocityMeasurementModel.hpp"
#include "SystemModel.hpp"

#include <chrono>
#include <fstream>
#include <random>

using T = double;
using State = Lie::State<T>;
using SE3 = State::SE3;
using Tangent = State::Tangent;
// No control
using Control = Kalman::Vector<T, 0>;
using SystemModel = Lie::SystemModel<T>;
using LiePositionMeasurementModel = Lie::LiePositionMeasurementModel<T>;
using LieVelocityMeasurementModel = Lie::LieVelocityMeasurementModel<T>;
using LieMeasurement = Lie::LieMeasurement<T>;

struct Noise
{
    // Random number generation (for noise simulation)
    std::default_random_engine generator;
    std::normal_distribution<T> noise;

    // Standard-Deviation of noise added to all state vector components during state transition
    T systemNoise = 0.1;
    // Standard-Deviation of noise added to all pos_measurement vector components
    T measurementNoise = 0.015;

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

struct CSVWriter
{
    std::ofstream csv;

    CSVWriter(const std::string& path = "data.csv")
    {
        csv.open(path);
    }

    std::ofstream& operator()()
    {
        return csv;
    }

    void write(const std::vector<Eigen::VectorXd>& data)
    {
        for (size_t j = 0; j < data.size(); j++)
        {
            auto& x = data[j];
            for (size_t i = 0; i < x.rows() - 1; i++)
            {
                csv << x(i) << ";";
            }
            csv << x(x.size() - 1);
            if (j < data.size() - 1)
                csv << ";";
        }
        csv << std::endl;
    }
};

int main(int argc, char* argv[])
{
    Noise noise;
    CSVWriter csv("data.csv");

    // Extended Kalman Filter
    Kalman::ExtendedKalmanFilter<State> ekf;
    SystemModel sys;
    LiePositionMeasurementModel pos_measurement;
    LieVelocityMeasurementModel velocity_measurement;

    Tangent x_init;
    Tangent v_init;
    v_init(0) = .1;
    v_init(1) = .2;
    v_init(4) = .2;

    // Generate a trajectory:
    // - From an initial position and velocity
    Tangent v = v_init;
    // Generate a trajectory
    std::vector<Tangent> traj(10);
    traj[0] = x_init;
    for (size_t i = 1; i < traj.size(); ++i)
    {
        traj[i] = SE3::log(SE3::exp(traj[i - 1]) * SE3::exp(v_init));
    }

    // Init filters with the true system state
    State x;
    x = State::Zero();
    // Init with actual velocity
    x.v = v_init;
    x.x = x_init;

    ekf.init(x);
    std::cout << "Initial State: " << x.x.matrix().transpose() << ", " << x.v.matrix().transpose() << std::endl;

    csv() << "#x_pred;x_mes;x_ekf" << std::endl;
    csv.write({traj[0], traj[0], traj[0]});
    velocity_measurement.addPosition(traj[0]);

    // No control
    Control u;

    LieMeasurement x_mes;
    LieMeasurement v_mes;
    for (int i = 1; i < traj.size(); i++)
    {
        Tangent x_ref = traj[i];

        // Use reference velocity for testing
        // velocity_measurement.addPosition(x_mes);
        velocity_measurement.addPosition(x_ref);

        // Simulate system (constant velocity model)
        // x = sys.f(x, u);
        // std::cout << "New State: " << x.x.transpose() << ", " << x.v.transpose() << std::endl;

        // Predict state for current time-step using the filters
        ekf.predict(sys, 1.);
        State x_ekf = ekf.getState();
        std::cout << "Pred State: " << x_ekf.x.transpose() << ", " << x_ekf.v.transpose() << std::endl;

        // Sensor update every 3 iteration, predict the rest of the time
        // if (i % 2 == 0)
        // {
        //     std::cout << "UPDATING VELOCITY" << std::endl;
        //     v_mes = velocity_measurement.v;
        //     ekf.update(velocity_measurement, v_mes);
        // }

        if (i % 3 == 0)
        {
            std::cout << "UPDATING POSITION" << std::endl;
            x_mes = noise.addNoise(x_ref);
            ekf.update(pos_measurement, x_mes);
        }

        auto cov = pos_measurement.getCovariance();
        std::cout << "covariance:\n"
                  << ekf.getCovariance().matrix() << std::endl;

        std::cout << "x_pred: " << x_ref.transpose() << std::endl;
        std::cout << "x_mes: " << x_mes.transpose() << std::endl;
        std::cout << "x_ekf: " << x_ekf.x.transpose() << std::endl;
        csv.write({x_ref, x_mes, x_ekf.x});
    }

    State x_future;
    sys.predict(ekf.getState(), 5, x_future);
    std::cout << "Future (dt=5): " << x_future.x.transpose() << std::endl;

    return 0;
}
