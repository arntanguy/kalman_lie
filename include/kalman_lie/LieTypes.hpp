#pragma once

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
template<typename T>
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
    enum {
      RowsAtCompileTime = _Rows,
      ColsAtCompileTime = _Cols
    };

    using SE3 = typename Sophus::SE3<Scalar>;
    using Tangent = typename SE3::Tangent;
    using Position = Tangent;
    using Velocity = Tangent;
    Tangent x;
    Tangent v;

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
      // std::cout << "rhs size: " << rhs.rows() << ", " << rhs.cols() << std::endl;
      x = SE3::log(SE3::exp(x) * SE3::exp(rhs.block(0,0,6,1)));
      v = SE3::log(SE3::exp(v) * SE3::exp(rhs.block(5,0,6,1)));
      return *this;
    }
};


} /* Lie */

