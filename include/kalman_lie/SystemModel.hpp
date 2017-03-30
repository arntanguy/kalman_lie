#pragma once

#include <kalman/LinearizedSystemModel.hpp>
#include <sophus/se3.hpp>
#include <unsupported/Eigen/AutoDiff>
#include <kalman_lie/LieTypes.hpp>
#include <kalman_utils/NumericalDiff.hpp>

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
        double dt;

        LieFunctor_(Parent* m, const double dt) : m(m), dt(dt)
        {
        }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
        {
            // Cost function results (with no control)
            S out;
            m->predict(S(x), dt, out);
            fvec = out;
            return 0;
        }

        int inputs() const { return S::dim(); }  // We are differentiating wrt inputs() variables
        int values() const { return S::dim(); }  // One observation each time
    };
    using LieFunctor = LieFunctor_<SystemModel<T>>;

    SystemModel()
    {
    }

    // new interface
    void predict(const S& x, const double dt, S& out)
    {
        // std::cout << "SystemModel::predict" << std::endl;
        out.x = S::SE3::log(S::SE3::exp(dt * x.v) * S::SE3::exp(x.x));
        out.v = x.v;
    }

    StateJacobian getJacobian(const S& x, const double dt)
    {
        // std::cout << "SystemModel::getJacobian" << std::endl;
        StateJacobian J;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(this->F.rows(), this->F.cols());
        jac.setZero();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xx(S::dim(), 1);
        xx << x.x.matrix(), x.v.matrix();

        // Compute numerical jacobian
        // F = df/dx (Jacobian of state transition w.r.t. the state)

        // Perform numerical differentiation
        NumericalDiffFunctor<LieFunctor> num_diff(this, dt);
        num_diff.df(xx, jac);
        J = jac;
        // std::cout << "SystemModel jacobian:\n" << J << std::endl;
        return J;
    }

    DEPRECATED S f(const S& /*x*/, const C& /*u*/) const override
    {
      return S::Zero();
    }
};

}  // namespace Robot
