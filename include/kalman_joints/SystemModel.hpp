#pragma once

#include <iostream>
#include <kalman/LinearizedSystemModel.hpp>
#include <kalman_utils/NumericalDiff.hpp>
#include <kalman_joints/JointTypes.hpp>

namespace Joints
{
template<typename T>
using Control = typename Kalman::Vector<T, Eigen::Dynamic>;

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
 */
template <typename T>
class SystemModel : public Kalman::LinearizedSystemModel<State<T>, Control<T>>
{
   public:
    //! State type shortcut definition
    using S = State<T>;
    using StateJacobian = Kalman::Jacobian<T, S>;
    //! No control
    using C = Control<T>;

    //! Number of joints
    size_t n_joints;

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

        int inputs() const { return m->n_joints; }
        int values() const { return m->n_joints; }
    };
    using Functor = Functor_<SystemModel<T>>;

    using M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    M Fk(const double dt) const
    {
      M Fk_;
      Fk_ = M::Identity(n_joints*2, n_joints*2);
      for (int i = 0; i < n_joints; ++i)
      {
        Fk_(i, n_joints + i) = dt;
      }
      return Fk_;
    }

    SystemModel(const size_t& n_joints) : n_joints(n_joints)
    {
        this->P.resize(n_joints*2, n_joints*2);
        this->P.setIdentity();
    }

    void predict(const S& x, const double dt, S& out)
    {
        // System model
        // xk+1 = xk+ xk_dot*dt
        out = Fk(dt) * x;
    }

    void predict(const S& x, const C& u, const double dt, S& out)
    {
        // System model
        // xk+1 = xk+ xk_dot*dt
        out = Fk(dt) * x + u;
    }

    StateJacobian getJacobian(const S& x, const double dt)
    {
        // std::cout << "SystemModel::getJacobian" << std::endl;
        StateJacobian J;
        // std::cout << "J size: " <crashtest< J.rows() << ", " << J.cols() << std::endl;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jac(x.dim(), x.dim());
        jac.setZero();
        // std::cout << "jac: " << jac << std::endl;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xx(x.dim(), 1);
        xx << x;
        // std::cout << "xx: " << xx << std::endl;

        // Compute numerical jacobian
        // F = df/dx (Jacobian of state transition w.r.t. the state)
        // Perform numerical differentiation
        NumericalDiffFunctor<Functor> num_diff(this, dt);
        num_diff.df(xx, jac);
        // std::cout << "num_diff" << std::endl;
        J = jac;
        std::cout << "SystemModel jacobian (no control):\n"
                  << J << std::endl;
        return J;
    }

    StateJacobian getJacobian(const S& x, const C& u, const double dt)
    {
        // std::cout << "SystemModel::getJacobian" << std::endl;
        StateJacobian J;
        J.resize(x.dim(), x.dim());
        J = Fk(dt);
        std::cout << "SystemModel Jacobian (with control):\n" << J << std::endl;
        return J;
    }

    DEPRECATED S f(const S& x, const C& /*u*/) const override
    {
      return x;
    }
};

}  // namespace Robot
