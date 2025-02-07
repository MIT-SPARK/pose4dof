/**
 * @file   pose4dof_unit_tests.cpp
 * @brief  main file for unit tests
 * @author Hyungtae Lim
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pose4dof/Pose4DoF.h>

using namespace gtsam;

template <typename Factor, typename V1, typename V2>
void evaluateFactor(const Factor& factor, const V1& v1, const V2& v2,
                    double tol, double delta = 1.0e-5) {
  gtsam::Matrix H1_actual, H2_actual;
#if GTSAM_VERSION_MAJOR <= 4 && GTSAM_VERSION_MINOR < 3
  const auto actual = factor.evaluateError(v1, v2, H1_actual, H2_actual);
#else
  const auto actual = factor.evaluateError(v1, v2, &H1_actual, &H2_actual);
#endif

  const auto H1_expected = gtsam::numericalDerivative21<gtsam::Vector, V1, V2>(
      [&](const auto& v1, const auto& v2) {
        return factor.evaluateError(v1, v2);
      },
      v1, v2, delta);

  const auto H2_expected = gtsam::numericalDerivative22<gtsam::Vector, V1, V2>(
      [&](const auto& v1, const auto& v2) {
        return factor.evaluateError(v1, v2);
      },
      v1, v2, delta);

  EXPECT_TRUE(gtsam::assert_equal(H1_expected, H1_actual, tol));
  EXPECT_TRUE(gtsam::assert_equal(H2_expected, H2_actual, tol));
}

/* ************************************************************************* */
// Test constructors and initialization
TEST(Pose4DoFTest, ConstructorInitialization) {
  // Test default constructor with Pose2 and z
  Pose2 pose2(1.0, 2.0, M_PI_4);  // x, y, theta
  Pose4DoF pose4dof(pose2, 3.0);  // Pose2, z
  EXPECT_EQ(pose4dof.x(), 1.0);
  EXPECT_EQ(pose4dof.y(), 2.0);
  EXPECT_EQ(pose4dof.z(), 3.0);
  EXPECT_EQ(pose4dof.theta(), M_PI_4);
  EXPECT_EQ(pose4dof.pitch(), 0.0);
  EXPECT_EQ(pose4dof.roll(), 0.0);

  // Test constructor with all values (x, y, z, yaw, pitch, roll)
  Pose4DoF pose4dof_full(1.0, 2.0, 3.0, M_PI_4, 0.1, 0.2);
  EXPECT_EQ(pose4dof_full.x(), 1.0);
  EXPECT_EQ(pose4dof_full.y(), 2.0);
  EXPECT_EQ(pose4dof_full.z(), 3.0);
  EXPECT_EQ(pose4dof_full.theta(), M_PI_4);
  EXPECT_EQ(pose4dof_full.pitch(), 0.1);
  EXPECT_EQ(pose4dof_full.roll(), 0.2);
}

/* ************************************************************************* */
// Test translation and rotation functions
TEST(Pose4DoFTest, TranslationRotation) {
  Pose4DoF pose4dof(1.0, 2.0, 3.0, M_PI_4);
  Point3 translation = pose4dof.translation();
  EXPECT_EQ(translation.x(), 1.0);
  EXPECT_EQ(translation.y(), 2.0);
  EXPECT_EQ(translation.z(), 3.0);

  Rot3 rotation = pose4dof.rotation();
  auto ypr = rotation.ypr();
  EXPECT_EQ(ypr(0), M_PI_4);  // Yaw
  EXPECT_EQ(ypr(1), 0.0);     // Pitch
  EXPECT_EQ(ypr(2), 0.0);     // Roll
}

/* ************************************************************************* */
// Test pose composition
TEST(Pose4DoFTest, PoseComposition) {
  Pose4DoF pose1(1.0, 0.0, 0.0, M_PI_4);
  Pose4DoF pose2(0.0, 1.0, 1.0, -M_PI_4);

  Pose4DoF composedPose = pose1.compose(pose2);

  EXPECT_NEAR(composedPose.x(), 1.0 - sqrt(2.0) / 2.0, 1e-5);
  EXPECT_NEAR(composedPose.y(), sqrt(2.0) / 2.0, 1e-5);
  EXPECT_NEAR(composedPose.z(), 1.0, 1e-5);
  EXPECT_NEAR(composedPose.theta(), 0.0, 1e-5);
}

/* ************************************************************************* */
// Test inverse function
TEST(Pose4DoFTest, PoseInverse) {
  double x = 1.0;
  double y = 2.0;
  double z = 3.0;
  double yaw = M_PI_4;
  Pose4DoF pose(x, y, z, yaw);
  Pose4DoF inversePose = pose.inverse();

  // -R^T * t
  double transformed_x = -cos(yaw) * x - sin(yaw) * y;
  double transformed_y = sin(yaw) * x - cos(yaw) * y;
  double transformed_z = -z;

  EXPECT_EQ(inversePose.x(), transformed_x);
  EXPECT_EQ(inversePose.y(), transformed_y);
  EXPECT_EQ(inversePose.z(), transformed_z);
}

/* ************************************************************************* */
// Test between function
TEST(Pose4DoFTest, PoseBetween) {
  Pose4DoF pose1(1.0, 2.5, 3.0, M_PI_4);
  Pose4DoF pose2(2.0, 3.0, 4.0, M_PI_4);

  Pose4DoF betweenPose = pose1.between(pose2);

  double rel_x = 2.0 - 1.0;
  double rel_y = 3.0 - 2.5;

  // R^T * (t2 - t1)
  double final_x = cos(M_PI_4) * rel_x + sin(M_PI_4) * rel_y;
  double final_y = -sin(M_PI_4) * rel_x + cos(M_PI_4) * rel_y;
  double final_z = 1.0;

  EXPECT_NEAR(betweenPose.x(), final_x, 1e-5);
  EXPECT_NEAR(betweenPose.y(), final_y, 1e-5);
  EXPECT_NEAR(betweenPose.z(), final_z, 1e-5);
}

/* ************************************************************************* */
// Test transformTo function
TEST(Pose4DoFTest, TransformToPoint) {
  Pose4DoF pose(1.0, 2.0, 3.0, M_PI_4);
  Point3 point(2.0, 3.0, 4.0);

  Point3 transformedPoint = pose.transformTo(point);

  EXPECT_NEAR(transformedPoint.x(), sqrt(2.0), 1e-5);
  EXPECT_NEAR(transformedPoint.y(), 0.0, 1e-5);
  EXPECT_NEAR(transformedPoint.z(), 1.0, 1e-5);
}

/* ************************************************************************* */
// Test numerical derivative
TEST(Pose4DoFTest, NumericalDerivative) {
  for (int i = 0; i < 10000; ++i) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> random_angle_noise(-M_PI / 2,
                                                              M_PI / 2);

    // Define two Pose2 instances
    Pose4DoF pose1(1.0, 1.0, 0.0, random_angle_noise(gen),
                   random_angle_noise(gen), random_angle_noise(gen));
    Pose4DoF pose2(2.0, 3.0, 0.0, random_angle_noise(gen),
                   random_angle_noise(gen), random_angle_noise(gen));

    static const gtsam::SharedNoiseModel& noise =
        gtsam::noiseModel::Isotropic::Variance(3, 1e-3);
    const auto factor =
        BetweenFactor<Pose4DoF>(4, 5, pose1.between(pose2), noise);

    evaluateFactor(factor, pose1, pose2, 1.0e-5);
  }
}

/* ************************************************************************* */
// Test Pose 4DoF SLAM Exammple
TEST(Pose4DoFTest, Pose4DoFSLAMWithGaussNewton) {
  int total_num_iter = 10000;
  for (int i = 0; i < total_num_iter; ++i) {
    std::random_device rd;
    std::mt19937 gen(rd());
    double stdev_yaw = M_PI / 6.0;
    std::uniform_real_distribution<double> random_angle_noise(-M_PI / 2,
                                                              M_PI / 2);
    std::uniform_real_distribution<double> yaw_noise(-stdev_yaw, stdev_yaw);

    NonlinearFactorGraph graph;

    auto prior_vec = Vector(4);
    prior_vec(0) = 0.3;
    prior_vec(1) = 0.3;
    prior_vec(2) = 0.3;
    prior_vec(3) = 0.1;
    noiseModel::Diagonal::shared_ptr priorNoise =
        noiseModel::Diagonal::Sigmas(prior_vec);
    graph.add(
        PriorFactor<Pose4DoF>(1, Pose4DoF(0.0, 0.0, 0.0, 0.0), priorNoise));

    // For simplicity, we will use the same noise model for odometry and loop
    // closures
    auto model_vec = Vector(4);
    model_vec(0) = 0.2;
    model_vec(1) = 0.2;
    model_vec(2) = 0.2;
    model_vec(3) = stdev_yaw;
    noiseModel::Diagonal::shared_ptr model =
        noiseModel::Diagonal::Sigmas(model_vec);

    gtsam::Pose4DoF p1 = gtsam::Pose4DoF(0.0, 0.0, 0.0, 0.0);
    gtsam::Pose4DoF p2 = gtsam::Pose4DoF(2.0, 0.0, 0.0, 0.0);
    gtsam::Pose4DoF p3 = gtsam::Pose4DoF(4.0, 0.0, 0.0, M_PI_2);
    gtsam::Pose4DoF p4 = gtsam::Pose4DoF(4.0, 2.0, 0.0, M_PI);
    gtsam::Pose4DoF p5 = gtsam::Pose4DoF(2.0, 2.0, 0.0, -M_PI_2);

    graph.add(BetweenFactor<Pose4DoF>(1, 2, p1.between(p2), model));
    graph.add(BetweenFactor<Pose4DoF>(2, 3, p2.between(p3), model));
    graph.add(BetweenFactor<Pose4DoF>(3, 4, p3.between(p4), model));
    graph.add(BetweenFactor<Pose4DoF>(4, 5, p4.between(p5), model));
    // Loop closure
    graph.add(
        BetweenFactor<Pose4DoF>(5, 2, Pose4DoF(2, 0.0, 0.0, M_PI_2), model));

    Values initialEstimate;
    // noisy angles are set to test robustness
    std::vector<double> yaws = {0.0 + yaw_noise(gen), 0.0 + yaw_noise(gen),
                                M_PI_2 + yaw_noise(gen), M_PI + yaw_noise(gen),
                                -M_PI_2 + yaw_noise(gen)};
    std::vector<double> pitches;
    std::vector<double> rolls;
    for (int i = 0; i < 5; ++i) {
      pitches.push_back(random_angle_noise(gen));
      rolls.push_back(random_angle_noise(gen));
    }

    initialEstimate.insert(
        1, Pose4DoF(0.6, 0.0, 0.0, yaws[0], pitches[0], rolls[0]));
    initialEstimate.insert(
        2, Pose4DoF(2.3, 0.1, 0.0, yaws[1], pitches[1], rolls[1]));
    initialEstimate.insert(
        3, Pose4DoF(4.1, 0.1, 0.0, yaws[2], pitches[2], rolls[2]));
    initialEstimate.insert(
        4, Pose4DoF(4.0, 2.0, 0.0, yaws[3], pitches[3], rolls[3]));
    initialEstimate.insert(
        5, Pose4DoF(2.1, 2.1, 0.0, yaws[4], pitches[4], rolls[4]));

    GaussNewtonParams parameters;
    parameters.relativeErrorTol = 1e-5;
    parameters.maxIterations = 100;

    GaussNewtonOptimizer optimizer(graph, initialEstimate, parameters);
    Values result = optimizer.optimize();

    // True pose values
    gtsam::Values expected_values;
    expected_values.insert(
        1, gtsam::Pose4DoF(0.0, 0.0, 0.0, 0.0, pitches[0], rolls[0]));
    expected_values.insert(
        2, gtsam::Pose4DoF(2.0, 0.0, 0.0, 0.0, pitches[1], rolls[1]));
    expected_values.insert(
        3, gtsam::Pose4DoF(4.0, 0.0, 0.0, M_PI_2, pitches[2], rolls[2]));
    expected_values.insert(
        4, gtsam::Pose4DoF(4.0, 2.0, 0.0, M_PI, pitches[3], rolls[3]));
    expected_values.insert(
        5, gtsam::Pose4DoF(2.0, 2.0, 0.0, -M_PI_2, pitches[4], rolls[4]));

    for (const auto& key_value : result) {
      auto key = key_value.key;
      auto optimized_pose = result.at<gtsam::Pose4DoF>(key);
      auto expected_pose = expected_values.at<gtsam::Pose4DoF>(key);
      EXPECT_TRUE(expected_pose.equals(optimized_pose, 1e-5));
    }
  }
}

/* ************************************************************************* */
TEST(Pose4DoFTest, Pose4DoFSLAMWithLevenbergMarquardt) {
  int total_num_iter = 10000;
  for (int i = 0; i < total_num_iter; ++i) {
    std::random_device rd;
    std::mt19937 gen(rd());
    double stdev_yaw = M_PI / 6.0;
    std::uniform_real_distribution<double> random_angle_noise(-M_PI / 2,
                                                              M_PI / 2);
    std::uniform_real_distribution<double> yaw_noise(-stdev_yaw, stdev_yaw);

    NonlinearFactorGraph graph;

    auto prior_vec = Vector(4);
    prior_vec(0) = 0.3;
    prior_vec(1) = 0.3;
    prior_vec(2) = 0.3;
    prior_vec(3) = 0.1;
    noiseModel::Diagonal::shared_ptr priorNoise =
        noiseModel::Diagonal::Sigmas(prior_vec);
    graph.add(
        PriorFactor<Pose4DoF>(1, Pose4DoF(0.0, 0.0, 0.0, 0.0), priorNoise));

    // For simplicity, we will use the same noise model for odometry and loop
    // closures
    auto model_vec = Vector(4);
    model_vec(0) = 0.2;
    model_vec(1) = 0.2;
    model_vec(2) = 0.2;
    model_vec(3) = stdev_yaw;
    noiseModel::Diagonal::shared_ptr model =
        noiseModel::Diagonal::Sigmas(model_vec);

    gtsam::Pose4DoF p1 = gtsam::Pose4DoF(0.0, 0.0, 0.0, 0.0);
    gtsam::Pose4DoF p2 = gtsam::Pose4DoF(2.0, 0.0, 0.0, 0.0);
    gtsam::Pose4DoF p3 = gtsam::Pose4DoF(4.0, 0.0, 0.0, M_PI_2);
    gtsam::Pose4DoF p4 = gtsam::Pose4DoF(4.0, 2.0, 0.0, M_PI);
    gtsam::Pose4DoF p5 = gtsam::Pose4DoF(2.0, 2.0, 0.0, -M_PI_2);

    graph.add(BetweenFactor<Pose4DoF>(1, 2, p1.between(p2), model));
    graph.add(BetweenFactor<Pose4DoF>(2, 3, p2.between(p3), model));
    graph.add(BetweenFactor<Pose4DoF>(3, 4, p3.between(p4), model));
    graph.add(BetweenFactor<Pose4DoF>(4, 5, p4.between(p5), model));
    // Loop closure
    graph.add(
        BetweenFactor<Pose4DoF>(5, 2, Pose4DoF(2, 0.0, 0.0, M_PI_2), model));

    Values initialEstimate;
    // noisy angles are set to test robustness
    std::vector<double> yaws = {0.0 + yaw_noise(gen), 0.0 + yaw_noise(gen),
                                M_PI_2 + yaw_noise(gen), M_PI + yaw_noise(gen),
                                -M_PI_2 + yaw_noise(gen)};
    std::vector<double> pitches;
    std::vector<double> rolls;
    for (int i = 0; i < 5; ++i) {
      pitches.push_back(random_angle_noise(gen));
      rolls.push_back(random_angle_noise(gen));
    }

    initialEstimate.insert(
        1, Pose4DoF(0.6, 0.0, 0.0, yaws[0], pitches[0], rolls[0]));
    initialEstimate.insert(
        2, Pose4DoF(2.3, 0.1, 0.0, yaws[1], pitches[1], rolls[1]));
    initialEstimate.insert(
        3, Pose4DoF(4.1, 0.1, 0.0, yaws[2], pitches[2], rolls[2]));
    initialEstimate.insert(
        4, Pose4DoF(4.0, 2.0, 0.0, yaws[3], pitches[3], rolls[3]));
    initialEstimate.insert(
        5, Pose4DoF(2.1, 2.1, 0.0, yaws[4], pitches[4], rolls[4]));

    LevenbergMarquardtParams parameters;
    parameters.relativeErrorTol = 1e-5;
    parameters.maxIterations = 100;

    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, parameters);
    Values result = optimizer.optimize();

    // True pose values
    gtsam::Values expected_values;
    expected_values.insert(
        1, gtsam::Pose4DoF(0.0, 0.0, 0.0, 0.0, pitches[0], rolls[0]));
    expected_values.insert(
        2, gtsam::Pose4DoF(2.0, 0.0, 0.0, 0.0, pitches[1], rolls[1]));
    expected_values.insert(
        3, gtsam::Pose4DoF(4.0, 0.0, 0.0, M_PI_2, pitches[2], rolls[2]));
    expected_values.insert(
        4, gtsam::Pose4DoF(4.0, 2.0, 0.0, M_PI, pitches[3], rolls[3]));
    expected_values.insert(
        5, gtsam::Pose4DoF(2.0, 2.0, 0.0, -M_PI_2, pitches[4], rolls[4]));

    for (const auto& key_value : result) {
      auto key = key_value.key;
      auto optimized_pose = result.at<gtsam::Pose4DoF>(key);
      auto expected_pose = expected_values.at<gtsam::Pose4DoF>(key);
      EXPECT_TRUE(expected_pose.equals(optimized_pose, 1e-5));
    }
  }
}

/* ************************************************************************* */
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
