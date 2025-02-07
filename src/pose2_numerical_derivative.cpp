#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Dense>

using namespace gtsam;
using namespace std;

template <typename Factor, typename Error, typename V1, typename V2>
void evaluateFactor(const Factor& factor, const V1& v1, const V2& v2,
                    const Error& expected, double tol, double delta = 1.0e-5) {
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

  std::cout << gtsam::assert_equal(Vector3(0.0, 0.0, 0.0), actual, tol)
            << std::endl;
  std::cout << gtsam::assert_equal(H1_expected, H1_actual, tol) << std::endl;
  std::cout << gtsam::assert_equal(H2_expected, H2_actual, tol) << std::endl;
}

int main() {
  // Define two Pose2 instances
  Pose2 pose1(1.0, 1.0, 0);
  Pose2 pose2(2.0, 3.0, M_PI / 4);

  Pose2 rel_pose = Pose2(1.0, 2.0, M_PI / 4);
  Vector3 error;
  error << 1.0, 2.0, M_PI / 4;

  static const gtsam::SharedNoiseModel& noise =
      gtsam::noiseModel::Isotropic::Variance(3, 1e-3);
  const auto factor = BetweenFactor<Pose2>(1, 2, pose1.between(pose2), noise);

  evaluateFactor(factor, pose1, pose2, error, 1.0e-5);
  return 0;
}
