project(lie_ros)

find_package(catkin COMPONENTS
  roscpp
  tf
  tf_conversions
  REQUIRED)

catkin_package(CATKIN_DEPENDS roscpp tf tf_conversions)

include_directories(${catkin_INCLUDE_DIRS})
ADD_EXECUTABLE(lie_ros main.cpp)
TARGET_LINK_LIBRARIES(lie_ros ${catkin_LIBRARIES})
