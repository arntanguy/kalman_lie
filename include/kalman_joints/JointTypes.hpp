#pragma once
#include <kalman/Matrix.hpp>

namespace Joints
{

/**
 * @brief System state vector-type for 1DoF linear joints (revolute)
 *
 * @param T Numeric scalar type
 * @param N Number of joints
 */
template <typename T, unsigned int N>
class State : public Kalman::Vector<T, 2 * N>
{
   public:
    KALMAN_VECTOR(State, T, 2 * N)

      /**
       * @brief Set joint positions
       * @param val Joint values
       */
    void q(const Kalman::Vector<T, N>& val)
    {
      this->block(0, 0, N, 1) = val;
    }

      /**
       * @brief Get joint positions
       * @return[q] vector of joint positions
       */
    Kalman::Vector<T, N> q() const
    {
      return this->block(0, 0, N, 1);
    }

    /**
     * @brief Set joint velocities
     * @param val Joint velocities
     */
    void q_dot(const Kalman::Vector<T, N>&  val)
    {
      this->block(N, 0, N, 1) = val;
    }

    /**
     * @brief Get joint velocities
     * @return[q_dot] joint velocities
     */
    Kalman::Vector<T, N> q_dot() const
    {
        return this->block(N, 0, N, 1);
    }

    /**
     * @brief Number of elements in the state
     */
    constexpr static size_t dim()
    {
        return 2 * N;
    }
};

template <typename T, unsigned int N>
class JointMeasurement : public Kalman::Vector<T, N>
{
 public:
  KALMAN_VECTOR(JointMeasurement, T, N);
};

} /* Joints */
