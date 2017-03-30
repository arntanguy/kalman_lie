#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman_lie/LiePositionMeasurementModel.hpp>
#include <kalman_lie/SystemModel.hpp>

#include <chrono>
#include <fstream>
#include <random>
#include <thread>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

using T = double;
using State = Lie::State<T>;
using SE3 = State::SE3;
using Tangent = State::Tangent;
// No control
using Control = Kalman::Vector<T, 0>;
using SystemModel = Lie::SystemModel<T>;
using LiePositionMeasurementModel = Lie::LiePositionMeasurementModel<T>;
using LieMeasurement = Lie::LieMeasurement<T>;

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

struct TFManager
{
   protected:
    std::shared_ptr<ros::NodeHandle> nh;
    std::unique_ptr<tf2_ros::TransformListener> tf_l;
    tf2_ros::Buffer tfBuffer;
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
            spinner_action();
            rt.sleep();
        }
    }

   public:
    TFManager(std::shared_ptr<ros::NodeHandle> nh, double rate = 100)
        : nh(nh), rate(rate)
    {
        tf_l = std::unique_ptr<tf2_ros::TransformListener>(new tf2_ros::TransformListener(tfBuffer));
        m_spin_th = std::thread(std::bind(&TFManager::spinner_thread, this));
    }

    ~TFManager()
    {
        m_spin = false;
        m_spin_th.join();
    }

    virtual void spinner_action() = 0;
};

struct TFKalman : public TFManager
{
   private:
    Kalman::ExtendedKalmanFilter<State> ekf;
    SystemModel sys;
    LiePositionMeasurementModel pos_measurement;

    tf2_ros::TransformBroadcaster br;

   public:
    TFKalman(std::shared_ptr<ros::NodeHandle> nh, double rate = 100) : TFManager(nh, rate)
    {
        Eigen::Affine3d pose;
        bool init = get_pose(pose);
        State x_init = State::Zero();
        if (init)
        {
            x_init.x = State::SE3::log(State::SE3(pose.matrix()));
        }
        ekf.init(x_init);
        std::cout << "Initial State: " << x_init.x.matrix().transpose() << ", " << x_init.v.matrix().transpose() << std::endl;
    }

    bool get_pose(Eigen::Affine3d& pose, float wait = 3.)
    {
        // Get TF
        geometry_msgs::TransformStamped transformStamped;
        bool got_pose = false;
        try
        {
            ros::Time now = ros::Time::now();
            // XXX hardcoded
            transformStamped = tfBuffer.lookupTransform("map", "camera_link", ros::Time(0), ros::Duration(wait));
            pose = tf2::transformToEigen(transformStamped);
            got_pose = true;
        }
        catch (tf2::TransformException& ex)
        {
            pose = Eigen::Affine3d::Identity();
            ROS_WARN("Could NOT find SLAM transform: %s", ex.what());
            got_pose = false;
        }
        return got_pose;
    }

    //! Called by ros spinner thread
    virtual void spinner_action() override
    {
        Eigen::Affine3d AX_SW_xtion;
        bool got_pose = get_pose(AX_SW_xtion);
        if (got_pose)
        {
            // Add to Kalman
            State::SE3 T(AX_SW_xtion.matrix());
            LieMeasurement x_mes = State::SE3::log(T);
            ekf.update(pos_measurement, x_mes);
        }

        // XXX use appropiate dt here
        auto dt = 1.;
        ekf.predict(sys, dt);
        std::cout << "ekf speed: " << ekf.getState().v.transpose() << std::endl;

        // Publish
        publish_ekf();
    }

    void publish_ekf()
    {
        geometry_msgs::TransformStamped tr;
        State x = ekf.getState();
        Eigen::Affine3d X(State::SE3::exp(x.x).matrix());
        tr = tf2::eigenToTransform(X);
        tr.header.frame_id = "map";
        tr.child_frame_id = "ekf";
        br.sendTransform(tr);

        State x_future;
        sys.predict(ekf.getState(), 1., x_future);
        geometry_msgs::TransformStamped tr_future;
        Eigen::Affine3d X_future(State::SE3::exp(x_future.x).matrix());
        tr_future = tf2::eigenToTransform(X_future);
        tr_future.header.frame_id = "map";
        tr_future.child_frame_id = "ekf_future";
        br.sendTransform(tr_future);
    }
};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "lie_ros");
    auto nh = std::make_shared<ros::NodeHandle>();
    TFKalman tf_ekf(nh, 100);

    ros::spin();

    return 0;
}
