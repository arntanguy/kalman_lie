KalmanLie
=========

Overview
--------

This repository contains an implementation of Lie-algebra kalman filters.

So far, we have to following implementations:
- 6D Pose estimation with constant velocity model

Dependencies
------------

This package relies heavily on:
- [kalman](ihttps://github.com/mherb/kalman) A c++11 kalman filtering framework
- [Sophus](https://github.com/strasdat/Sophus) for Lie algebra support.


6D Pose estimation filter with constant velocity model
--------------

For this model, the state is composed of the 6D pose and velocity in the lie algebra *se(3)*.

The system model is as follow

![System Model Equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bs%7D%20%3D%20%5B%5Cmathbf%7Bx%7D%20%2C%20%5Cmathbf%7B%5Cdot%7Bx%7D%7D%5D%5C%5C%20%5Cmathbf%7Bx%7D_%7Bk&plus;1%7D%20%3D%20log%28e%5E%7B%5Cmathbf%7Bx_k%7D%7D%20*%20e%5E%7B%5Cdot%7B%5Cmathbf%7Bx_k%7D%7D*dt%7D%29%5C%5C%20%5Cmathbf%7B%5Cdot%7Bx%7D_%7Bk&plus;1%7D%7D%20%3D%20%5Cmathbf%7B%5Cdot%7Bx%7D_%7Bk%7D%7D)

