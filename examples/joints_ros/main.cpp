#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman_joints/SystemModel.hpp>
#include <kalman_joints/JointPositionMeasurementModel.hpp>

#include <chrono>
#include <fstream>
#include <random>
#include <thread>
#include <iostream>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/JointState.h>

using T = double;
constexpr size_t N = 2;
using State = Joints::State<T>;
using SystemModel = Joints::SystemModel<T>;
using JointMeasurement = Joints::JointMeasurement<T>;
using JointPositionMeasurementModel = Joints::JointPositionMeasurementModel<T>;

struct ROSManager
{
   protected:
    std::shared_ptr<ros::NodeHandle> nh;
    ros::Subscriber joint_sub;
    ros::Publisher joint_pub;
    std::vector<std::string> joint_names;

    double rate;

    bool m_spin = true;
    std::thread m_spin_th;

    void spinner_thread()
    {
        // std::unique_lock<std::mutex> lock(spinMutex);
        // spinCV.wait(lock);
        ros::Rate rt(rate);
        while (m_spin && ros::ok())
        {
            rt.sleep();
            ros::spinOnce();
        }
    }

   protected:
    Kalman::ExtendedKalmanFilter<State> ekf;
    std::unique_ptr<SystemModel> sys;
    std::unique_ptr<JointPositionMeasurementModel> pos_measurement;

   public:
    ROSManager(std::shared_ptr<ros::NodeHandle> nh, double rate = 100)
        : nh(nh), rate(rate)
    {
        joint_sub = nh->subscribe("/joint_states", 1, &ROSManager::jointsCallback, this);
        joint_pub = nh->advertise<sensor_msgs::JointState>("/kalman/joint_states", 1);
        m_spin_th = std::thread(std::bind(&ROSManager::spinner_thread, this));
    }

    ~ROSManager()
    {
        m_spin = false;
        m_spin_th.join();
    }

    void jointsCallback(const sensor_msgs::JointState& msg)
    {
      auto size = msg.name.size();
      Eigen::VectorXd joints(size);
      for (size_t i = 0; i < size; ++i) {
        joints(i) = msg.position[i];
      }

      // initialize on first joint callback
      if(!sys)
      {
        joint_names = msg.name;

        sys.reset(new SystemModel(size));
        pos_measurement.reset(new JointPositionMeasurementModel(size));
        State s(size*2);
        // set the joint positions
        s.q(joints);
        // initialize ekf
        ekf.init(s);
      }
      else
      {
        JointMeasurement mes = joints;
        ekf.update(*pos_measurement, mes);
      }
      // XXX dt
      ekf.predict(*sys, 1.);
      std::cout << "ekf q:\n " << ekf.getState().q().transpose() << std::endl;
      std::cout << "ekf q_dot:\n " << ekf.getState().q_dot().transpose() << std::endl;

      publish_ekf();
    }

    void publish_ekf()
    {
      sensor_msgs::JointState msg;
      msg.name = joint_names;
      msg.position.resize(joint_names.size());
      msg.velocity.resize(joint_names.size());
      const auto &q = ekf.getState().q();
      const auto &q_dot = ekf.getState().q_dot();
      for (size_t i = 0; i < joint_names.size(); ++i) {
        msg.position[i] = q(i);
        msg.velocity[i] = q_dot(i);
      }
      msg.header.stamp = ros::Time::now();
      joint_pub.publish(msg);
    }
};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "lie_ros");
    auto nh = std::make_shared<ros::NodeHandle>();
    ROSManager tf_ekf(nh, 100);

    ros::spin();

    return 0;
}
