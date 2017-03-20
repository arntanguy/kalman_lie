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
