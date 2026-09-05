#include "GaussianProcessFactory.hpp"

#include "GaussianProcess.hpp"
#include "GPNeuralNetwork.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(GaussianProcessFactory, "gaussian_process", ({
    {"standard", std::make_shared<GaussianProcess>},
    {"csg", std::make_shared<GPSampleNodeCSG>},
}))


DEFINE_STRINGABLE_ENUM(MeanFunctionFactory, "mean", ({
    {"homogeneous", std::make_shared<HomogeneousMean>},
    {"spherical", std::make_shared<SphericalMean>},
    {"linear", std::make_shared<LinearMean>},
    {"tabulated", std::make_shared<TabulatedMean>},
    {"mesh", std::make_shared<MeshSdfMean>},
    {"procedural", std::make_shared<ProceduralMean>},
    {"neural", std::make_shared<NeuralMean>},
}))


DEFINE_STRINGABLE_ENUM(CovarianceFunctionFactory, "covariance", ({
    {"squared_exponential", std::make_shared<SquaredExponentialCovariance>},
    {"rational_quadratic", std::make_shared<RationalQuadraticCovariance>},
    {"matern", std::make_shared<MaternCovariance>},
    {"gabor_aniso", std::make_shared<GaborAnisotropicCovariance>},
    {"gabor_iso", std::make_shared<GaborIsotropicCovariance>},
    {"dot_product", std::make_shared<DotProductCovariance>},
    {"periodic", std::make_shared<PeriodicCovariance>},
    {"thin_plate", std::make_shared<ThinPlateCovariance>},
    {"grid_nonstationary", std::make_shared<GridNonstationaryCovariance>},
    {"mg_nonstationary", std::make_shared<MeanGradNonstationaryCovariance>},
    {"neural_nonstationary", std::make_shared<NeuralNonstationaryCovariance>},
    {"proc_nonstationary", std::make_shared<ProceduralNonstationaryCovariance>},
}))


DEFINE_STRINGABLE_ENUM(ProceduralScalarFactory, "procedural_scalar", ({
    {"constant", std::make_shared<ConstantScalar>},
    {"sdf", std::make_shared<ProceduralSdf>},
    {"noise", std::make_shared<ProceduralNoise>},
    {"regular_grid", std::make_shared<RegularGridScalar>},
}))

DEFINE_STRINGABLE_ENUM(ProceduralVectorFactory, "procedural_vector", ({
    {"constant", std::make_shared<ConstantVector>},
    {"regular_grid", std::make_shared<RegularGridVector>},
    {"noise", std::make_shared<ProceduralNoiseVec>},
}))

}
