#pragma once
#include <kalman/LinearizedMeasurementModel.hpp>
#include "LieTypes.hpp"
#include "NumericalDiff.hpp"

namespace Lie
{
/**
 * @brief Measurement model for measuring a 6D pose
 *
 *  This is a measurement model for measuring a 6DoF pose from direct
 *  observations (SLAM, ...)
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template <typename T, template <class> class CovarianceBase = Kalman::StandardBase>
class LiePositionMeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, LieMeasurement<T>, CovarianceBase>
{
   public:
    //! State type shortcut definition
    using S = State<T>;

    //! Measurement type shortcut definition
    using M = LieMeasurement<T>;

    using MeasurementJacobian = Kalman::Jacobian<T, S, M>;

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
            S s;
            s.x = x;
            M o;
            // Cost function
            m->measure(s, o);
            fvec = o;
            return 0;
        }

        int inputs() const { return 6; }  // There are two parameters of the model
        int values() const { return 6; }  // The number of observations
    };
    using LieFunctor = LieFunctor_<LiePositionMeasurementModel<T>>;
    NumericalDiffFunctor<LieFunctor> num_diff;

    LiePositionMeasurementModel() : num_diff(this)
    {
        // Setup noise jacobian. As this one is static, we can define it once
        // and do not need to update it dynamically
        this->V.setIdentity();
    }

    /**
     * @brief Definition of (possibly non-linear) measurement function
     *
     * This function maps the system state to the measurement that is expected
     * to be received from the sensor assuming the system is currently in the
     * estimated state.
     * Here we are only interested in the pose measurement (position+orientation)
     *
     * @param [in] x The system state in current time-step
     * @param [out] out The (predicted) pose measurement for the system state
     */
    void measure(const S& x, M& out)
    {
        std::cout << "LiePositionMeasurementModel::measure" << std::endl;
        out = x.x;
    }

    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-linear measurement function \f$h(x)\f$ around the
     * current state \f$x\f$.
     *
     * @note This is only needed when implementing a LinearizedSystemModel,
     *       for usage with an ExtendedKalmanFilter or SquareRootExtendedKalmanFilter.
     *       When using a fully non-linear filter such as the UnscentedKalmanFilter
     *       or its square-root form then this is not needed.
     *
     * @param x The current system state around which to linearize
     * @returns J The measurement jacobian
     */
    MeasurementJacobian getJacobian(const S& x)
    {
        // H = dh/dx (Jacobian of measurement function w.r.t. the state)
        MeasurementJacobian J;
        // 6x12 jacobian for the pose measurement update
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(this->H.rows(), this->H.cols());
        jac.setZero();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xx = x.x;
        num_diff.df(xx, jac);
        J = jac;
        return J;
    }

    DEPRECATED M h(const S& x) const override
    {
    }

   protected:
    DEPRECATED void updateJacobians(const S& x)
    {
    }
};

}  // namespace Robot
