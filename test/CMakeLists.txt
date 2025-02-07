find_package(GTest REQUIRED)

include(GoogleTest)
enable_testing()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(
  test_${PROJECT_NAME} pose4dof_unit_tests.cpp ../src/Pose4DoF.cpp)
target_link_libraries(test_${PROJECT_NAME} GTest::gtest_main gtsam Eigen3::Eigen)

# if we don't want to set main file
# gtest_add_tests(TARGET test_${PROJECT_NAME})
