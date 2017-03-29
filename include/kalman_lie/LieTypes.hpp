#pragma once

#include <sophus/se3.hpp>

// Lie Type Definition
#define LIE_KALMAN_VECTOR(NAME, T)                                   \
    typedef typename Sophus::SE3<T>::Tangent Base;                   \
    typedef typename Sophus::SE3<T> Group;                           \
    using typename Base::Scalar;                                     \
    using Base::RowsAtCompileTime;                                   \
    using Base::ColsAtCompileTime;                                   \
    using Base::SizeAtCompileTime;                                   \
    NAME(void) : Base() {}                                           \
    template <typename OtherDerived>                                 \
    NAME(const Eigen::MatrixBase<OtherDerived>& other) : Base(other) \
    {                                                                \
    }                                                                \
                                                                     \
    template <typename OtherDerived>                                 \
    NAME& operator=(const Eigen::MatrixBase<OtherDerived>& other)    \
    {                                                                \
        this->Base::operator=(other);                                \
        return *this;                                                \
    }

namespace Lie
{
/**
 * @brief Measurement vector measuring the robot position
 *
 * @param T Numeric scalar type
 */
template <typename T>
class LieMeasurement : public Sophus::SE3<T>::Tangent
{
   public:
    LIE_KALMAN_VECTOR(LieMeasurement, T)
};

/**
 * @brief System state vector-type for a 6DOF Pose
 *
 * This is a system state for a 6D Pose expressed in Lie algebra,
 * with a constant velocity model
 *
 * @param T
 */
template <typename _Scalar, int _Rows = 12, int _Cols = 1>
class State
{
   public:
    using Scalar = _Scalar;
    enum
    {
        RowsAtCompileTime = _Rows,
        ColsAtCompileTime = _Cols
    };

    using SE3 = typename Sophus::SE3<Scalar>;
    using Tangent = typename SE3::Tangent;
    using Position = Tangent;
    using Velocity = Tangent;
    Tangent x;
    Tangent v;

    State()
    {
        x.setZero();
        v.setZero();
    }

    State(const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>& x)
    {
        assert(x.rows() == dim() && "State construction failed: the parameter vector does not have the correct number of state variables");
        this->x = x.block(0, 0, 6, 1);
        this->v = x.block(6, 0, 6, 1);
    }

    operator Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>() const
    {
        Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> res(dim(), 1);
        res << x, v;
        return res;
    }

    /**
     * @brief Total number of state variables
     */
    constexpr int static dim()
    {
        return RowsAtCompileTime;
    }

    constexpr size_t rows() const
    {
      return RowsAtCompileTime;
    }

    static State Zero()
    {
        State t;
        t.x = Tangent::Zero();
        t.v = Tangent::Zero();
        return t;
    }

    // XXX should be adding a state, not a matrix
    State& operator+=(const Eigen::Matrix<_Scalar, _Rows, _Cols>& rhs)
    {
        // XXX should this be addition?
        x += rhs.block(0, 0, 6, 1);
        v += rhs.block(6, 0, 6, 1);
        // x = SE3::log(SE3::exp(rhs.block(0, 0, 6, 1)) * SE3::exp(x));
        // v = SE3::log(SE3::exp(rhs.block(6, 0, 6, 1)) * SE3::exp(v));
        // std::cout << "x: " << x.transpose() << std::endl;
        // std::cout << "v: " << v.transpose() << std::endl;
        return *this;
    }
};

} /* Lie */
