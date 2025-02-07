#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
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
  std::uniform_real_distribution<double> random_angle_noise(-M_PI / 2,
                                                            M_PI / 2);

  NonlinearFactorGraph graph;
  NonlinearFactorGraph graph_w_pose3;

  auto prior_vec = Vector(4);
  prior_vec << 0.3, 0.3, 0.3, 0.1;
  noiseModel::Diagonal::shared_ptr priorNoise =
      noiseModel::Diagonal::Sigmas(prior_vec);

  graph.add(PriorFactor<Pose4DoF>(1, Pose4DoF(Pose3()), priorNoise));
  graph_w_pose3.add(PriorFactor<Pose4DoF>(1, Pose4DoF(Pose3()), priorNoise));

  auto model_vec = Vector(4);
  model_vec << 0.2, 0.2, 0.2, 0.1;
  noiseModel::Diagonal::shared_ptr model =
      noiseModel::Diagonal::Sigmas(model_vec);

  vector<Pose3> pose3_trajectory = {
      Pose3(Rot3::Ypr(0.0, 0.1, 0.1), Point3(0.0, 0.0, 0.0)),
      Pose3(Rot3::Ypr(0.0, 0.2, 0.1), Point3(2.0, 0.0, 0.0)),
      Pose3(Rot3::Ypr(M_PI_2, 0.3, 0.1), Point3(4.0, 0.0, 0.0)),
      Pose3(Rot3::Ypr(M_PI, 0.4, 0.1), Point3(4.0, 2.0, 0.0)),
      Pose3(Rot3::Ypr(-M_PI_2, 0.5, 0.1), Point3(2.0, 2.0, 0.0))};

  for (size_t i = 0; i < pose3_trajectory.size() - 1; ++i) {
    // NOTE(hlim) This part is crucial
    Pose4DoF pose_from = Pose4DoF(pose3_trajectory[i]);
    Pose4DoF pose_to = Pose4DoF(pose3_trajectory[i + 1]);
    graph.add(BetweenFactor<Pose4DoF>(i + 1, i + 2, pose_from.between(pose_to),
                                      model));
    // NOTE(hlim) Note, this is wrong!
    graph_w_pose3.add(BetweenFactor<Pose4DoF>(
        i + 1, i + 2, pose3_trajectory[i].between(pose3_trajectory[i + 1]),
        model));
  }

  Values initialEstimate;
  for (size_t i = 0; i < pose3_trajectory.size(); ++i) {
    double p_random = random_angle_noise(gen);
    double y_random = random_angle_noise(gen);
    Pose3 noisy_pose = pose3_trajectory[i].compose(
        Pose3(Rot3::Ypr(0.1, p_random, y_random), Point3(0.1, 0.1, 0.1)));
    std::cout << noisy_pose.rotation().ypr().transpose() << "\n";
    initialEstimate.insert(i + 1, Pose4DoF(noisy_pose));
  }

  LevenbergMarquardtParams parameters;
  parameters.relativeErrorTol = 1e-5;
  parameters.maxIterations = 100;

  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, parameters);
  Values result = optimizer.optimize();
  result.print("Final Optimized Result:");

  // NOTE(hlim)
  // As seen in the results, directly using Pose3 for the between factor
  // causes instability in the z value!
  LevenbergMarquardtOptimizer optimizer2(graph_w_pose3, initialEstimate,
                                         parameters);
  Values result_w_pose3 = optimizer2.optimize();
  result_w_pose3.print("\033[1;33mFinal Optimized Result for comparison:");
  std::cout << "\033[0m\n";

  return 0;
}
