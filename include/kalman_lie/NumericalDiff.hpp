#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/NumericalDiff>

/**
 * @brief Generic Functor for Eigen AutoDiff
 *
 * @tparam _Scalar
 */
template <typename _Scalar>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
    int inputs() const { return m_inputs; }  // number of degree of freedom (= 2*nb_vertices)
    int values() const { return m_values; }  // number of energy terms (= nb_vertices + nb_edges)
    // you should define that in the subclass :
    //    void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

/**
 * @brief Eigen::NumericalDiff adds a df function to your functor
 * that numerically computes the jacobian.
 */
template <class Functor>
struct NumericalDiffFunctor : Eigen::NumericalDiff<Functor>
{
    // Perfect forwarding of constructor arguments
    template <typename... Args>
    NumericalDiffFunctor(Args&&... args)
        : Eigen::NumericalDiff<Functor>(std::forward<Args>(args)...)
    {
    }
};
