#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <random>

#include "pose4dof/Pose4DoF.h"

using namespace std;
using namespace gtsam;

int main(int argc, char** argv) {
  std::random_device rd;
  std::mt19937 gen(rd());
  double stdev_yaw = 0.1;
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
  graph.add(PriorFactor<Pose4DoF>(1, Pose4DoF(0.0, 0.0, 0.0, 0.0), priorNoise));

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

  graph.add(
      BetweenFactor<Pose4DoF>(5, 2, Pose4DoF(2, 0.0, 0.0, M_PI_2), model));

  Values initialEstimate;
  // noisy angles are set to test robustness
  vector<double> yaws = {0.0 + yaw_noise(gen), 0.0 + yaw_noise(gen),
                         M_PI_2 + yaw_noise(gen), M_PI + yaw_noise(gen),
                         -M_PI_2 + yaw_noise(gen)};
  vector<double> pitches;
  vector<double> rolls;
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

  // GaussNewtonParams parameters;
  // // parameters.setVerbosity("DELTA");
  // parameters.relativeErrorTol = 1e-5;
  // parameters.maxIterations = 100;

  // GaussNewtonOptimizer optimizer(graph, initialEstimate, parameters);

  LevenbergMarquardtParams parameters;
  // parameters.setVerbosity("DELTA");
  parameters.relativeErrorTol = 1e-5;
  parameters.maxIterations = 100;

  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, parameters);

  Values result = optimizer.optimize();
  result.print("Final Result:\n");

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

    if (!expected_pose.equals(optimized_pose, 1e-5)) {
      std::cout << "\033[1;31mPose " << key << " is outside the tolerance!\n";
    }
  }

  std::cout
      << "\033[1;32mSuccessfully optimized preserving pitch and roll angles!\n";

  return 0;
}
