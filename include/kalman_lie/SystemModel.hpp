#pragma once

#include <kalman/LinearizedSystemModel.hpp>
#include <sophus/se3.hpp>
#include <unsupported/Eigen/AutoDiff>
#include <kalman_lie/LieTypes.hpp>
#include <kalman_lie/NumericalDiff.hpp>

namespace Lie
{
/**
 * @brief System model for constant-velocity 6D pose model
 *
 * This is the system model, defining how our pose moves from one
 * time-step to the next, i.e. how the system state evolves over time.
 *
 * This model assumes a constant velocity, that is between measurement updates,
 * the velocity is kept as the last known velocity.
 *
 * @param T Numeric scalar type
 */
template <typename T>
class SystemModel : public Kalman::LinearizedSystemModel<State<T>>
{
   public:
    //! State type shortcut definition
    using S = State<T>;

    //! No control
    using C = Kalman::Vector<T, 0>;

    using StateJacobian = Kalman::Jacobian<T, S>;

    // Wraps the system model cost function in an automatic differentiation functor
    // XXX should be computed manually
    template <typename Parent>
    struct LieFunctor_ : Functor<T>
    {
        Parent* m;

        LieFunctor_(Parent* m) : m(m)
        {
        }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
        {
            // Cost function results (with no control)
            fvec = m->f(S(x), C());
            return 0;
        }

        int inputs() const { return S::dim(); }  // We are differentiating wrt inputs() variables
        int values() const { return S::dim(); }  // One observation each time
    };
    using LieFunctor = LieFunctor_<SystemModel<T>>;
    // Perform numerical differentiation
    NumericalDiffFunctor<LieFunctor> num_diff;

    SystemModel() : num_diff(this)
    {
    }

    // new interface
    void predict(const S& x, const double dt, S& out)
    {
        std::cout << "SystemModel::predict" << std::endl;
        // Is dt multiplication done like this correct?
        out.x = S::SE3::log(S::SE3::exp(x.x) * S::SE3::exp(dt * x.v));
        out.v = x.v;
    }

    StateJacobian getJacobian(const S& x, const double dt)
    {
        std::cout << "SystemModel::getJacobian" << std::endl;
        StateJacobian J;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(this->F.rows(), this->F.cols());
        jac.setZero();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xx(S::dim(), 1);
        xx << x.x.matrix(), x.v.matrix();

        // Compute numerical jacobian
        // F = df/dx (Jacobian of state transition w.r.t. the state)
        num_diff.df(xx, jac);
        J = jac;
        return J;
    }

    /**
     * @brief Definition of (non-linear) state transition function
     * This function defines how the system state is propagated through time,
     * i.e. it defines in which state \f$\hat{x}_{k+1}\f$ is system is expected to
     * be in time-step \f$k+1\f$ given the current state \f$x_k\f$ in step \f$k\f$ and
     * the system control input \f$u\f$.
     *
     * TODO: Give the ability to pass a user-defined timestep
     *
     * @param [in] x The system state in current time-step
     * @param [in] u The control vector input (**unused**)
     * @returns The (predicted) system state in the next time-step
     */
    DEPRECATED S f(const S& x, const C& /*u*/) const
    {
        //! Predicted state vector after transition
        S x_;

        //! Predict given current pose and velocity
        // Constant velocity model: update position based on last known velocity
        x_.x = S::SE3::log(S::SE3::exp(x.x) * S::SE3::exp(x.v));
        // Update velocity directly based on measurements
        x_.v = x.v;
        return x_;
    }

   protected:
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-lineear state transition function \f$f(x,u)\f$ around the
     * current state \f$x\f
     *
     * @note This is only needed when implementing a LinearizedSystemModel,
     *       for usage with an ExtendedKalmanFilter or SquareRootExtendedKalmanFilter.
     *       When using a fully non-linear filter such as the UnscentedKalmanFilter
     *       or its square-root form then this is not needed.
     *
     * @param x The current system state around which to linearize
     * @param u The current system control input
     */
    DEPRECATED void updateJacobians(const S& x, const C& /*u*/)
    {
        // Return transitioned state vector
        // H = dh/dx (Jacobian of measurement function w.r.t. the state)
        // this->H.setZero();
        // // 6x12 jacobian for the pose measurement update
        // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(this->H.rows(), this->H.cols());
        // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xx = x.x;
        // num_diff.df(xx, jac);
        // this->H = jac;
    }
};

}  // namespace Robot
