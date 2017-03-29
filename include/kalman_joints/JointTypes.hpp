#pragma once
#include <kalman/Matrix.hpp>

namespace Joints
{
/**
 * @brief System state vector-type for 1DoF linear joints (revolute)
 *
 * @param N Number of joints
 */
template <typename T>
class State : public Kalman::Vector<T, Kalman::Dynamic>
{
   public:
    KALMAN_VECTOR(State, T, Kalman::Dynamic)

    State(const size_t N)
    {
        this->resize(N);
    }

    /**
       * @brief Set joint positions
       * @param val Joint values
       */
    void q(const Base& val)
    {
        this->block(0, 0, q_dim(), 1) = val;
    }

    /**
       * @brief Get joint positions
       * @return[q] vector of joint positions
       */
    Base q() const
    {
        return this->block(0, 0, q_dim(), 1);
    }

    /**
     * @brief Set joint velocities
     * @param val Joint velocities
     */
    void q_dot(const Base& val)
    {
        this->block(q_dim(), 0, q_dim(), 1) = val;
    }

    /**
     * @brief Get joint velocities
     * @return[q_dot] joint velocities
     */
    Base q_dot() const
    {
        return this->block(q_dim(), 0, q_dim(), 1);
    }

    size_t q_dim() const
    {
        return dim() / 2;
    }

    /**
     * @brief Number of elements in the state
     */
    size_t dim() const
    {
        return this->rows();
    }
};

template <typename T>
class JointMeasurement : public Kalman::Vector<T, Kalman::Dynamic>
{
   public:
    KALMAN_VECTOR(JointMeasurement, T, Kalman::Dynamic);
    JointMeasurement(const size_t N)
    {
        this->resize(N);
    }
};

} /* Joints */
