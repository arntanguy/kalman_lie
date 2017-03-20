#pragma once

#include <kalman/LinearizedSystemModel.hpp>
#include <sophus/se3.hpp>
#include <unsupported/Eigen/AutoDiff>
#include <kalman_lie/NumericalDiff.hpp>
#include <kalman_lie/LieTypes.hpp>

namespace KalmanExamples
{
namespace Lie
{
/**
 * @brief System state vector-type for a 6DOF Pose
 *
 * This is a system state for a 6D Pose expressed in Lie algebra,
 * with a constant velocity model
 *
 * @param T
 */
template <typename T>
class State : public Sophus::SE3<T>::Tangent
{
   public:
    LIE_KALMAN_VECTOR(State, T)

    // //! X-position
    // static constexpr size_t X = 0;
    // //! Y-Position
    // static constexpr size_t Y = 1;
    // //! Orientation
    // static constexpr size_t THETA = 2;

    // T x()       const { return (*this)[ X ]; }
    // T y()       const { return (*this)[ Y ]; }
    // T theta()   const { return (*this)[ THETA ]; }

    // T& x()      { return (*this)[ X ]; }
    // T& y()      { return (*this)[ Y ]; }
    // T& theta()  { return (*this)[ THETA ]; }
};

/**
 * @brief System control-input in se3
 *
 * @param T Numeric scalar type
 */
template <typename T>
class Control : public Sophus::SE3<T>::Tangent
{
   public:
    LIE_KALMAN_VECTOR(Control, T)

    // //! Velocity
    // static constexpr size_t V = 0;
    // //! Angular Rate (Orientation-change)
    // static constexpr size_t DTHETA = 1;

    // T v()       const { return (*this)[ V ]; }
    // T dtheta()  const { return (*this)[ DTHETA ]; }

    // T& v()      { return (*this)[ V ]; }
    // T& dtheta() { return (*this)[ DTHETA ]; }
};

/**
 * @brief System model for a simple planar 3DOF robot
 *
 * This is the system model defining how our robot moves from one
 * time-step to the next, i.e. how the system state evolves over time.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template <typename T>
class SystemModel : public Kalman::LinearizedSystemModel<State<T>>
{
   public:
    //! State type shortcut definition
    typedef State<T> S;

    //! No control
    typedef Kalman::Vector<T, 0> C;

    S velocity;

    // Wraps the system model cost function in an automatic differentiation functor
    template <typename Parent>
    struct LieFunctor_ : Functor<T>
    {
        Parent* m;

        LieFunctor_(Parent* m) : m(m)
        {
        }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
        {
            C c;
            // Cost function
            fvec = m->f(x, c);
            return 0;
        }

        int inputs() const { return 6; }  // There are two parameters of the model
        int values() const { return 6; }  // The number of observations
    };
    using LieFunctor = LieFunctor_<SystemModel<T>>;

    NumericalDiffFunctor<LieFunctor> num_diff;

    SystemModel() : num_diff(this)
    {
    }

    /**
     * @brief Definition of (non-linear) state transition function
     * This function defines how the system state is propagated through time,
     * i.e. it defines in which state \f$\hat{x}_{k+1}\f$ is system is expected to
     * be in time-step \f$k+1\f$ given the current state \f$x_k\f$ in step \f$k\f$ and
     * the system control input \f$u\f$.
     *
     * @param [in] x The system state in current time-step
     * @param [in] u The control vector input
     * @returns The (predicted) system state in the next time-step
     */
    // XXX todo pass timestep
    S f(const S& x, const C& u) const
    {
        //! Predicted state vector after transition
        S x_;

        //! Predict given current pose and velocity
        // XXX equation for constant velocity model
        x_ = S::Group::log(S::Group::exp(x) * S::Group::exp(velocity));

        // Return transitioned state vector
        return x_;
    }

   protected:
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-lineRowsAtCompileTimear state transition function \f$f(x,u)\f$ around the
     * current state \f$x\f   typedef typename Functor::Scalar Scalar;
     *
     * @note This is only needed when implementing a LinearizedSystemModel,
     *       for usage with an ExtendedKalmanFilter or SquareRootExtendedKalmanFilter.
     *       When using a fully non-linear filter such as the UnscentedKalmanFilter
     *       or its square-root form then this is not needed.
     *
     * @param x The current system state around which to linearize
     * @param u The current system control input
     */
    void updateJacobians(const S& x, const C& u)
    {
        // F = df/dx (Jacobian of state transition w.r.t. the state)
        this->F.setZero();

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(6, 6);
        num_diff.df(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(x), jac);
        // std::cout << "jacobian: " << jac.matrix() << std::endl;
        this->F = jac;
    }
};

}  // namespace Robot
}  // namespace KalmanExamples
