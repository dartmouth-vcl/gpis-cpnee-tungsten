#ifndef SPARSECONVOLUITONNOISEMEDIUM_HPP_
#define SPARSECONVOLUITONNOISEMEDIUM_HPP_

#include "GaussianProcessMedium.hpp"
#include "math/SparseConvolutionNoise.hpp"

namespace Tungsten {

    class GaussianProcess;

    struct GPContextSparseConvNoise : public GPContext {
        SparseConvolutionNoiseRealization noise;
        virtual void reset() override {
            // Don't reset the realization
        }
    };

    class SparseConvolutionNoiseMedium : public GaussianProcessMedium
    {
        float _rayMarchStepSize;
        uint _rayMarchMinStep;
        uint _globalSeed;
        float _impulseDensity;
        bool _useSingleRealization;
        bool _isotropicSpace3DSampling; // Switch between sampling 3D noise in isotropic vs world space
        bool _1DSampling; // 1D noise is sampled in isotropic space
        SparseConv1DSamplingScheme _1DsamplingScheme;
        // x: scale for UNI, y: scale for NEE, z: scale for CPNEE
        Vec3f _1DsamplingSchemeSampleCountScale;
        bool _correlationXY; // Consider the correlation along XY axes for the gradient, when conditioning on 3D normal

        // Tricky way to assign different phase functions
        bool _surfVolPhaseSeparate;
        float _surfVolPhaseAmpThresh;

    public:

        SparseConvolutionNoiseMedium();
        SparseConvolutionNoiseMedium(std::shared_ptr<GaussianProcess> gp, std::vector<std::shared_ptr<PhaseFunction>> phases,
                                         float materialSigmaA, float materialSigmaS, float density, int numBasisFunctions, bool useSingleRealization, float rayMarchStepSize) :
                GaussianProcessMedium(gp, phases, materialSigmaA, materialSigmaS, density), _rayMarchStepSize(rayMarchStepSize)
        {}

        static SparseConv1DSamplingScheme stringToSamplingScheme1D(const std::string& name);
        static Vec3f samplingSchemeToSampleCountScale(SparseConv1DSamplingScheme scheme, JsonPtr value);
        static std::string samplingScheme1DToString(SparseConv1DSamplingScheme val);

        virtual void fromJson(JsonPtr value, const Scene &scene) override;
        virtual rapidjson::Value toJson(Allocator &allocator) const override;

        virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& isect_p, const float& isect_t, MediumState& state, Vec3d& grad) const override;

        virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool) const override;
    };

}

#endif /* SPARSECONVOLUITONNOISEMEDIUM_HPP_ */
