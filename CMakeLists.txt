cmake_minimum_required(VERSION 3.0.2)
project(robots_moviles)
cmake_policy(SET CMP0054 NEW)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf
  gazebo_ros
)

find_package(gazebo REQUIRED)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf gazebo_ros
  DEPENDS GAZEBO
)

link_directories(${GAZEBO_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch models rviz worlds 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  src/robots_moviles/topology_library.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/exploration.py
  scripts/topological_navigation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)