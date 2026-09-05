#include "SparseConvolutionNoiseMedium.hpp"
#include "WeightSpaceGaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformPathSampler.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {
    SparseConvolutionNoiseMedium::SparseConvolutionNoiseMedium()
            : GaussianProcessMedium(
                    std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(),
                    std::make_shared<SquaredExponentialCovariance>()),
                    {}, 0.0f, 0.0f, 1.f),
              _rayMarchStepSize(0.01f),
              _rayMarchMinStep(8),
              _globalSeed(0u),
              _impulseDensity(3.0f),
              _useSingleRealization(false),
              _isotropicSpace3DSampling(false),
              _1DSampling(false),
              _1DsamplingScheme(SparseConv1DSamplingScheme::UNI),
              _1DsamplingSchemeSampleCountScale(1.f, 0.f, 0.f),
              _correlationXY(false),
              _surfVolPhaseSeparate(false),
              _surfVolPhaseAmpThresh(0.0)
    {
    }

    SparseConv1DSamplingScheme SparseConvolutionNoiseMedium::stringToSamplingScheme1D(const std::string& name)
    {
        if (name == "uni" || name == "UNI")
            return SparseConv1DSamplingScheme::UNI;
        else if (name == "nee" || name == "NEE")
            return SparseConv1DSamplingScheme::NEE;
        else if (name == "mis" || name == "MIS")
            return SparseConv1DSamplingScheme::MIS;
        else if (name == "cpnee" || name == "CPNEE")
            return SparseConv1DSamplingScheme::CPNEE;
        else if (name == "mis_3way" || name == "MIS_3WAY")
            return SparseConv1DSamplingScheme::MIS_3WAY;
        FAIL("Invalid sparse conv sampling scheme: '%s'", name);
    }

    Vec3f SparseConvolutionNoiseMedium::samplingSchemeToSampleCountScale(SparseConv1DSamplingScheme scheme, JsonPtr value)
    {
        Vec3f result;
        switch (scheme) {
            default:
            case SparseConv1DSamplingScheme::UNI:  result = Vec3f(1.f, 0.f, 0.f); break;
            case SparseConv1DSamplingScheme::NEE:  result = Vec3f(0.f, 1.f, 0.f); break;
            case SparseConv1DSamplingScheme::MIS:  result = Vec3f(0.5f, 0.5f, 0.f); break;
            case SparseConv1DSamplingScheme::CPNEE:  result = Vec3f(0.f, 0.f, 1.f); break;
            case SparseConv1DSamplingScheme::MIS_3WAY:  result = Vec3f(0.333333333f, 0.333333333f, 0.333333334f); break;
        }
        value.getField("1D_sampling_scheme_sample_count_scale", result); // Allow user manually set for scale. 
        return result;
    }

    std::string SparseConvolutionNoiseMedium::samplingScheme1DToString(SparseConv1DSamplingScheme val)
    {
        switch (val) {
            default:
            case SparseConv1DSamplingScheme::UNI:  return "uni";
            case SparseConv1DSamplingScheme::NEE:  return "nee";
            case SparseConv1DSamplingScheme::MIS:  return "mis";
            case SparseConv1DSamplingScheme::CPNEE:  return "cpnee";
            case SparseConv1DSamplingScheme::MIS_3WAY:  return "mis_3way";
        }
    }

    void SparseConvolutionNoiseMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("step_size", _rayMarchStepSize);
        value.getField("min_step", _rayMarchMinStep);
        value.getField("seed", _globalSeed);
        value.getField("impulse_density", _impulseDensity);
        value.getField("single_realization", _useSingleRealization);
        value.getField("isotropic_3D_sampling", _isotropicSpace3DSampling);
        value.getField("1D_sampling", _1DSampling);
        std::string scheme1DString = "uni";
        value.getField("1D_sampling_scheme", scheme1DString);
        _1DsamplingScheme = stringToSamplingScheme1D(scheme1DString);
        _1DsamplingSchemeSampleCountScale = samplingSchemeToSampleCountScale(_1DsamplingScheme, value);
        value.getField("1D_gradient_correlationXY", _correlationXY);
        value.getField("surf_vol_phase_separate", _surfVolPhaseSeparate);
        value.getField("surf_vol_phase_amp_thresh", _surfVolPhaseAmpThresh);
    }

    rapidjson::Value SparseConvolutionNoiseMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
                           "type", "sparse_conv_noise",
                           "step_size", _rayMarchStepSize,
                           "min_step", _rayMarchMinStep,
                           "seed", _globalSeed,
                           "impulse_density", _impulseDensity,
                           "single_realization", _useSingleRealization,
                           "isotropic_3D_sampling", _isotropicSpace3DSampling,
                           "1D_sampling", _1DSampling,
                           "1D_sampling_scheme", samplingScheme1DToString(_1DsamplingScheme),
                           "1D_sampling_scheme_sample_count_scale", _1DsamplingSchemeSampleCountScale,
                           "1D_gradient_correlationXY", _correlationXY,
                           "surf_vol_phase_separate", _surfVolPhaseSeparate,
                           "surf_vol_phase_amp_thresh", _surfVolPhaseAmpThresh,
        };
    }

    bool SparseConvolutionNoiseMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray,
                                                      const Vec3d& isect_p, const float& isect_t, MediumState& state, Vec3d& grad) const {
        auto gp = _gp->get_gaussian_process();
        UniformSampler uniformSampler;
        GPContextSparseConvNoise& ctxt = *(GPContextSparseConvNoise*)state.gpContext.get();
        grad = Vec3d(ctxt.noise.evaluateGradient(Vec3f(isect_p), isect_t, ray.dir(), state.info, uniformSampler));
        return true;
    }

    bool SparseConvolutionNoiseMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool) const {
        auto gp = _gp->get_gaussian_process();
        SparseConvolutionNoiseRealization noise(gp, _ctxt, _globalSeed, _impulseDensity, _useSingleRealization, _isotropicSpace3DSampling, _1DSampling, 
            _1DsamplingScheme, _1DsamplingSchemeSampleCountScale, _correlationXY, _surfVolPhaseSeparate, _surfVolPhaseAmpThresh);

        auto rd = vec_conv<Vec3d>(ray.dir());
        float step_size = (ray.farT() - ray.nearT()) / (float) _rayMarchMinStep;
        if (_rayMarchStepSize < step_size)
            step_size = _rayMarchStepSize;

        int GPId;
        UniformSampler uniformSampler;

        auto p = vec_conv<Vec3d>(ray.pos());
        t = ray.nearT();

        if (!state.firstScatter) {
            double targetVal = state.lastVal;
            Vec3d targetGrad = state.lastAniso;
            // Perform conditioning at the start of the secondary ray
            // Supported memory models: Renewal, Renewal+
            noise.conditioning(Vec3f(p), ray.dir(), targetVal, Vec3f(targetGrad), state.info, uniformSampler);
        }

        double f0 = noise.evaluateValue(Vec3f(p + t * rd), t, ray.dir(), state.info, uniformSampler, GPId);
        int sign0 = f0 < 0 ? -1 : 1;

        double pf = f0;
        t = ray.nearT() + step_size * sampler.next1D();
        int step = 0;
        // Perform ray marching (with equal step size) to find the first zero-crossing
        while (t < ray.farT()) {
            step ++;
            auto p_c = p + t * rd;
            double f_c = noise.evaluateValue(Vec3f(p_c), t, ray.dir(), state.info, uniformSampler, GPId);
            int signc = f_c < 0 ? -1 : 1;

            if (!state.firstScatter && step == 1) {
                sign0 = signc;
            }
            else if (signc != sign0) {
                // Estimate the exact intersect location
                double intp_factor = pf / (pf - f_c);
                double t_test, t_test_prev;
                int sign_test;
                t_test_prev = lerp(t - step_size, t, intp_factor);
                do {
                    t_test = lerp(t - step_size, t, intp_factor); // intp_factor changes in each loop
                    double f_test = noise.evaluateValue(Vec3f(p + t_test * rd), t_test, ray.dir(), state.info, uniformSampler, GPId);
                    sign_test = f_test < 0 ? -1 : 1;
                    if (sign_test == sign0) {
                        break;
                    }
                    intp_factor *= 0.9;
                    if (intp_factor <= 0.01) {
                        t_test_prev = t_test = 0;
                        break;
                    }
                    t_test_prev = t_test;
                } while(true);

                t = t_test_prev;
                state.lastGPId = GPId;
                auto samplingScheme_Scale_Pair = noise.samplingScheme(Vec3f(p + t * rd));
                state.sparseConv1DSamplingScheme = samplingScheme_Scale_Pair.first;
                state.sparseConv1DSamplingSchemeSampleCountScale = samplingScheme_Scale_Pair.second;
                auto ctxt = std::make_shared<GPContextSparseConvNoise>();
                ctxt->noise = noise;
                state.gpContext = ctxt;
                state.lastVal = 0.0;
                return true;
            }

            pf = f_c;
            t += step_size;
        }

        t = ray.farT();
        state.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
        state.sparseConv1DSamplingSchemeSampleCountScale = Vec3f(1.f, 0.f, 0.f);
        auto ctxt = std::make_shared<GPContextSparseConvNoise>();
        ctxt->noise = noise;
        state.gpContext = ctxt;
        state.lastVal = noise.evaluateValue(Vec3f(p + t * rd), t, ray.dir(), state.info, uniformSampler, GPId);
        return false;
    }
}