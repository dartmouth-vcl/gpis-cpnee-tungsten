#ifndef FUNCTIONSPACEGAUSSIANPROCESSMEDIUM_HPP_
#define FUNCTIONSPACEGAUSSIANPROCESSMEDIUM_HPP_

#include "GaussianProcessMedium.hpp"

namespace Tungsten {

class GaussianProcess;

class FunctionSpaceGaussianProcessMedium : public GaussianProcessMedium
{
    int _samplePoints;
    double _stepSizeCov;
    double _stepSize;
    double _skipSpace;

    // Tricky way to assign different phase functions
    bool _surfVolPhaseSeparate;
    float _surfVolPhaseAmpThresh;

public:
    FunctionSpaceGaussianProcessMedium();
    FunctionSpaceGaussianProcessMedium(std::shared_ptr<GaussianProcess> gp, 
        std::vector<std::shared_ptr<PhaseFunction>> phase,
        float materialSigmaA, float materialSigmaS, float density, int samplePoints,
        GPCorrelationContext ctxt = GPCorrelationContext::RenewalPlus,
        GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete, 
        GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian,
        double stepSizeCov = 0, double stepSize = 0, double skipSpace = 0) :
            GaussianProcessMedium(gp, phase, materialSigmaA, materialSigmaS, density, ctxt, intersectMethod, normalSamplingMethod),
            _samplePoints(samplePoints), _stepSizeCov(stepSizeCov), _stepSize(stepSize), _skipSpace(skipSpace)
    {}

    FunctionSpaceGaussianProcessMedium(std::shared_ptr<GaussianProcess> gp,
        float materialSigmaA, float materialSigmaS, float density, int samplePoints,
        GPCorrelationContext ctxt = GPCorrelationContext::RenewalPlus,
        GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete,
        GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian,
        double stepSizeCov = 0, double stepSize = 0, double skipSpace = 0) :
        GaussianProcessMedium(gp, {nullptr}, materialSigmaA, materialSigmaS, density, ctxt, intersectMethod, normalSamplingMethod),
        _samplePoints(samplePoints), _stepSizeCov(stepSizeCov), _stepSize(stepSize), _skipSpace(skipSpace)
    {}

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& isect_p, const float& isect_t, MediumState& state, Vec3d& grad) const override;

    virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool firstIsectAlongRay) const override;

    /*virtual Vec3f transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface,
        bool endOnSurface, MediumSample * sample) const override;*/
};

}

#endif /* FUNCTIONSPACEGAUSSIANPROCESSMEDIUM_HPP_ */
