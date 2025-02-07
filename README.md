# GTSAM Pose4DoF Package

> **Caution:** This package is currently under validation; please refrain from using it at this time. 


### How to build & run

It just can be used as a ROS package now! 

```
cd ~/catkin_ws/src
git clone git@github.mit.edu:SPARK/gtsam_4dof.git
cd ..
catkin build pose4dof
```

### Notes About TEST

In 

* `TEST(Pose4DoFTest, Pose4DoFSLAMWithGaussNewton)` 
* `TEST(Pose4DoFTest, Pose4DoFSLAMWithLevenbergMarquardt)` 
 
in `test/pose4dof_unit_tests.cpp`, it might be a natural phenomenon, but once `stdev_yaw` becomes too large, the unit tests do not pass.
