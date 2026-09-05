#include "WeightSpaceGaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformPathSampler.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

namespace Tungsten {


    WeightSpaceGaussianProcessMedium::WeightSpaceGaussianProcessMedium()
        : GaussianProcessMedium(
            std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>()),
            {},
            0.0f, 0.0f, 1.f),
        _numBasisFunctions(300),
        _useSingleRealization(false),
        _rayMarchStepSize(0.01f),
        _rayMarchMinStep(8),
        _globalSeed(0u)
    {
    }

    void WeightSpaceGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("basis_functions", _numBasisFunctions);
        value.getField("single_realization", _useSingleRealization);
        value.getField("step_size", _rayMarchStepSize);
        value.getField("min_step", _rayMarchMinStep);
        value.getField("seed", _globalSeed);

        if (_useSingleRealization) {
            auto basisSampler = UniformPathSampler(0xdeadbeef);
            auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
            WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, _numBasisFunctions, Vec4u(0u), Vec3d(0.), false, 3);

            _globalReal = WeightSpaceRealization::sample(std::make_shared<WeightSpaceBasis>(basis), gp, Vec4u(0u), _globalSeed);
        }
    }

    rapidjson::Value WeightSpaceGaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "weight_space_gaussian_process",
            "basis_functions", _numBasisFunctions,
            "single_realization", _useSingleRealization,
            "step_size", _rayMarchStepSize,
            "min_step", _rayMarchMinStep,
            "seed", _globalSeed,
        };
    }

    bool WeightSpaceGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& isect_p, const float& isect_t, MediumState& state, Vec3d& grad) const {
        if(_intersectMethod == GPIntersectMethod::Mean || !state.gpContext) {
            // auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
            auto gp = _gp->get_gaussian_process();
            auto ctxt = std::make_shared<GPContextWeightSpace>();

            if (_useSingleRealization) {
                ctxt->real = _globalReal;
            }
            else {
                Vec4u pixel_sample_segment = state.info.pixelSampleSegment;
                if (_ctxt == GPCorrelationContext::Global)
                    pixel_sample_segment.w() = 0;
                WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, _numBasisFunctions, pixel_sample_segment, Vec3d(), false, 3);
                ctxt->real = WeightSpaceRealization::sample(std::make_shared<WeightSpaceBasis>(basis), gp, pixel_sample_segment, _globalSeed);
            }

            state.gpContext = ctxt;
        }

        GPContextWeightSpace& ctxt = *(GPContextWeightSpace*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch(_normalSamplingMethod) {
            case GPNormalSamplingMethod::FiniteDifferences:
            {
                float eps = 0.0001f;
                std::array<Vec3d, 6> gradPs{
                        isect_p + Vec3d(eps, 0.f, 0.f),
                        isect_p + Vec3d(0.f, eps, 0.f),
                        isect_p + Vec3d(0.f, 0.f, eps),
                        isect_p - Vec3d(eps, 0.f, 0.f),
                        isect_p - Vec3d(0.f, eps, 0.f),
                        isect_p - Vec3d(0.f, 0.f, eps),
                };

                std::array<double, 6> gradVs;
                int GPIdTmp;
                for(int i = 0; i < 6; i++) {
                    gradVs[i] = ctxt.real.evaluate(gradPs[i], GPIdTmp);
                }

                grad = Vec3d{
                    gradVs[0] - gradVs[3],
                    gradVs[1] - gradVs[4],
                    gradVs[2] - gradVs[5],
                } / (2 * eps);

                break;
            }
            case GPNormalSamplingMethod::ConditionedGaussian:
            {
                grad = ctxt.real.evaluateGradient(isect_p);
                break;
            }
            case GPNormalSamplingMethod::Beckmann:
            {
                auto deriv = Derivative::First;
                Vec3d normal = Vec3d(
                    _gp->mean(&isect_p, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                    _gp->mean(&isect_p, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                    _gp->mean(&isect_p, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

                TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

                Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
                float alpha = _gp->compute_beckmann_roughness(isect_p);
                BeckmannNDF ndf(0, alpha, alpha);

                grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
                break;
            }
            case GPNormalSamplingMethod::GGX:
            {
                auto deriv = Derivative::First;
                Vec3d normal = Vec3d(
                    _gp->mean(&isect_p, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                    _gp->mean(&isect_p, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                    _gp->mean(&isect_p, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

                TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

                Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
                float alpha = _gp->compute_beckmann_roughness(isect_p);
                GGXNDF ndf(0, alpha, alpha);
                grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
                break;
            }
        }


        return true;
    }

    bool WeightSpaceGaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool) const {
        if(_ctxt != GPCorrelationContext::Global || !state.gpContext) {
            auto gp = _gp->get_gaussian_process();
            auto ctxt = std::make_shared<GPContextWeightSpace>();

            if (_useSingleRealization) {
                ctxt->real = _globalReal;
            }
            else {
                Vec4u pixel_sample_segment = state.info.pixelSampleSegment;
                if (_ctxt == GPCorrelationContext::Global)
                    pixel_sample_segment.w() = 0;
                WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, _numBasisFunctions, pixel_sample_segment, Vec3d(), false, 3);
                ctxt->real = WeightSpaceRealization::sample(std::make_shared<WeightSpaceBasis>(basis), gp, pixel_sample_segment, _globalSeed);
            }

            state.gpContext = ctxt;
        }

        GPContextWeightSpace& ctxt = *(GPContextWeightSpace*)state.gpContext.get();
        const WeightSpaceRealization& real = ctxt.real;

        double farT = ray.farT();
        auto rd = vec_conv<Vec3d>(ray.dir());

        int GPId;

        if (_rayMarchStepSize == 0) {
            const double sig_0 = (farT - ray.nearT()) * 0.1f;
            const double delta = 0.01;
            const double np = 1.5;
            const double nm = 0.5;

            t = 0;
            double sig = sig_0;

            auto p = vec_conv<Vec3d>(ray.pos()) + (t + ray.nearT()) * rd;
            double f0 = real.evaluate(p, GPId);

            int sign0 = f0 < 0 ? -1 : 1;

            for (int i = 0; i < 2048 * 4; i++) {
                auto p_c = p + (t + ray.nearT() + delta) * rd;
                double f_c = real.evaluate(p_c, GPId);
                int signc = f_c < 0 ? -1 : 1;

                if (signc != sign0) {
                    t += ray.nearT();
                    state.lastGPId = GPId;
                    return true;
                }

                auto c = p + (t + ray.nearT() + sig * 0.5) * rd;
                auto v = sig * 0.5 * rd;

                double nsig;
                if (real.rangeBound(c, { v }) != RangeBound::Unknown) {
                    nsig = sig;
                    sig = sig * np;
                }
                else {
                    nsig = 0;
                    sig = sig * nm;
                }

                t += max(nsig, delta);

                if (t + ray.nearT() >= farT) {
                    t += ray.nearT();
                    return false;
                }
            }

            std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
            t = ray.farT();
            return false;
        }
        else {
            float step_size = (ray.farT() - ray.nearT()) / (float) _rayMarchMinStep;
            if (_rayMarchStepSize < step_size)
                step_size = _rayMarchStepSize;

            auto p = vec_conv<Vec3d>(ray.pos());
            t = ray.nearT();
            double f0 = real.evaluate(p + t * rd, GPId);
            int sign0 = f0 < 0 ? -1 : 1;

            double pf = f0;
            t = ray.nearT() + step_size * sampler.next1D();
            int step = 0;
            while (t < ray.farT()) {
                step ++;
                auto p_c = p + t * rd;
                double f_c = real.evaluate(p_c, GPId);
                int signc = f_c < 0 ? -1 : 1;

                if (!state.firstScatter && step == 1) {
                    sign0 = signc;
                }
                else if (signc != sign0) {
                    // Note: estimate the exact intersect location
                    double intp_factor = pf / (pf - f_c);
                    double t_test, t_test_prev;
                    int sign_test;
                    t_test_prev = lerp(t - step_size, t, intp_factor);
                    do {
                        t_test = lerp(t - step_size, t, intp_factor); // intp_factor changes in each loop
                        double f_test = real.evaluate(p + t_test * rd, GPId);
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
                    return true;
                }

                pf = f_c;
                t += step_size;
            }

            t = ray.farT();
            return false;
        }
    }
}
