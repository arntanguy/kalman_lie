#pragma once

#include <iostream>
#include <kalman/LinearizedSystemModel.hpp>
#include <kalman_joints/NumericalDiff.hpp>
#include <kalman_joints/JointTypes.hpp>

namespace Joints
{
/**
 * @brief System model for joints position and velocity
 *
 * This is the system model, defining how each joint moves from one
 * time-step to the next, i.e. how the system state evolves over time.
 *
 * This model assumes a constant velocity, that is between measurement updates,
 * the velocity is kept as the last known velocity.
 *
 * @param T Numeric scalar type
 * @param N Number of joints dof
 */
template <typename T, unsigned int N>
class SystemModel : public Kalman::LinearizedSystemModel<State<T, N>>
{
   public:
    //! State type shortcut definition
    using S = State<T, N>;
    using StateJacobian = Kalman::Jacobian<T, S>;
    //! No control
    using C = Kalman::Vector<T, 0>;

    // Wraps the system model cost function in an automatic differentiation functor
    // XXX should be computed manually
    template <typename Parent>
    struct Functor_ : Functor<T>
    {
        Parent* m;
        double dt;

        Functor_(Parent* m, const double dt) : m(m), dt(dt)
        {
        }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
        {
            // Cost function results (with no control)
            S fvec_;
            m->predict(x, dt, fvec_);
            fvec = fvec_;
            return 0;
        }

        int inputs() const { return S::dim(); }
        int values() const { return S::dim(); }
    };
    using Functor = Functor_<SystemModel<T, N>>;

    SystemModel()
    {
    }

    void predict(const S& x, const double dt, S& out)
    {
        using M = Eigen::Matrix<T, S::dim(), S::dim()>;
        M Fk = M::Identity();
        for (int i = 0; i < N; ++i)
        {
            Fk(i, N + i) = dt;
        }

        // System model
        // xk+1 = xk+ xk_dot*dt
        out = Fk * x;
    }

    StateJacobian getJacobian(const S& x, const double dt)
    {
        // std::cout << "SystemModel::getJacobian" << std::endl;
        StateJacobian J;
        // std::cout << "J size: " << J.rows() << ", " << J.cols() << std::endl;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(J.rows(), J.cols());
        jac.setZero();
        // std::cout << "jac: " << jac << std::endl;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xx(S::dim(), 1);
        xx << x;
        // std::cout << "xx: " << xx << std::endl;

        // Compute numerical jacobian
        // F = df/dx (Jacobian of state transition w.r.t. the state)
        // Perform numerical differentiation
        NumericalDiffFunctor<Functor> num_diff(this, dt);
        num_diff.df(xx, jac);
        // std::cout << "num_diff" << std::endl;
        J = jac;
        // std::cout << "SystemModel jacobian:\n"
        //           << J << std::endl;
        return J;
    }

    DEPRECATED S f(const S& x, const C& /*u*/) const
    {
    }

   protected:
    DEPRECATED void updateJacobians(const S& x, const C& /*u*/)
    {
    }
};

}  // namespace Robot
