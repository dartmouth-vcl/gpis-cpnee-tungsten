#include "FunctionSpaceGaussianProcessMedium.hpp"
#include "WeightSpaceGaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

namespace Tungsten {

    FunctionSpaceGaussianProcessMedium::FunctionSpaceGaussianProcessMedium()
        : GaussianProcessMedium(
            std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>()), 
            {},
            0.f, 0.f, 1.f, GPCorrelationContext::RenewalPlus, GPIntersectMethod::GPDiscrete, GPNormalSamplingMethod::ConditionedGaussian),
            _samplePoints(32),
            _stepSizeCov(0.),
            _stepSize(0.),
            _skipSpace(0.),
            _surfVolPhaseSeparate(false),
            _surfVolPhaseAmpThresh(0.0)
    {
    }

    void FunctionSpaceGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("sample_points", _samplePoints);
        value.getField("step_size_cov", _stepSizeCov);
        value.getField("step_size", _stepSize);
        value.getField("skip_space", _skipSpace);
        value.getField("surf_vol_phase_separate", _surfVolPhaseSeparate);
        value.getField("surf_vol_phase_amp_thresh", _surfVolPhaseAmpThresh);
    }

    rapidjson::Value FunctionSpaceGaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "function_space_gaussian_process",
            "sample_points", _samplePoints,
            "step_size_cov", _stepSizeCov,
            "step_size", _stepSize,
            "skip_space", _skipSpace,
            "surf_vol_phase_separate", _surfVolPhaseSeparate,
            "surf_vol_phase_amp_thresh", _surfVolPhaseAmpThresh,
        };
    }

    bool FunctionSpaceGaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool firstIsectAlongRay) const {
        std::vector<Vec3d> points(_samplePoints);
        std::vector<Derivative> derivs(_samplePoints);
        std::vector<double> ts(_samplePoints);
        double tOffset = sampler.next1D();

        auto ro = vec_conv<Vec3d>(ray.pos());
        auto rd = vec_conv<Vec3d>(ray.dir()).normalized();

        auto gp = _gp->get_gaussian_process();

        double nearT = ray.nearT();
        double farT = ray.farT();
        if (_skipSpace > 0) {
            double emptySpaceStepSize = _stepSize > 0 ? _stepSize : 0.01;
            while (gp->cdf(ro + (nearT + emptySpaceStepSize) * rd) < _skipSpace && nearT < farT) {
                nearT += emptySpaceStepSize;
            }

            if (farT - nearT < emptySpaceStepSize) {
                t = ray.farT();
                return false;
            }

            /*while (_gp->cdf(ro + (farT - emptySpaceStepSize) * rd) < _skipSpace && nearT < farT) {
                farT -= emptySpaceStepSize;
            }

            if (farT - nearT < emptySpaceStepSize) {
                t = ray.farT();
                return false;
            }*/
        }


        double maxRayDist = farT - nearT;
        double determinedStepSize = maxRayDist / _samplePoints;

        // Note: remove this parameter, for fair comparison with sparse convolution noise
        if (false) {
        // if (_stepSizeCov > 0) {
            double goodStepSize = gp->goodStepsize(ro, _stepSizeCov, rd);

            if (goodStepSize < determinedStepSize) {
                determinedStepSize = goodStepSize;
            }
        }
        else if (_stepSize > 0 && _stepSize < determinedStepSize) {
            determinedStepSize = _stepSize;
        }


        maxRayDist = determinedStepSize * _samplePoints;

        double maxT = nearT + maxRayDist;

        for (int i = 0; i < _samplePoints; i++) {
            double rt = lerp((double)ray.nearT()+determinedStepSize*0.1, ray.nearT() + maxRayDist, clamp((i - tOffset) / (_samplePoints-1), 0., 1.));
            if (i == 0)
                rt = nearT + determinedStepSize * 0.1;
            else if (i == _samplePoints - 1)
                rt = nearT + maxRayDist;

            ts[i] =  rt;
            points[i] = ro + rt * rd;
            derivs[i] = Derivative::None;
        }

        std::shared_ptr<GPRealNode> gpSamples;

        std::shared_ptr<GPContextFunctionSpace> ctxt;
        if (!state.firstScatter && state.gpContext) {
            ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
            if (ctxt->points.size() == 0) {
                std::cerr << "Empty context!\n";
                return false;
            }
        }

        if (state.firstScatter || !state.gpContext) {
            auto rp = vec_conv<Vec3d>(ray.pos() + ray.nearT() * ray.dir());
            std::array<Vec3d, 1> cond_pts = { rp };
            std::array<Derivative, 1> cond_deriv = { Derivative::None };
            std::shared_ptr<GPRealNode> cond_vs = gp->sample_start_value(rp, sampler);
            gpSamples = gp->sample_cond(
                points.data(), derivs.data(), _samplePoints, nullptr,
                cond_pts.data(), cond_vs.get(), cond_deriv.data(),
                0, // 0 or 1, 1 = condition on the positive start value
                nullptr,
                nullptr, 0,
                rd, 1, sampler);

        }
        else {
            // assert(ctxt->points.size() > 0);

            Vec3d lastIntersectPt = ctxt->points[ctxt->points.size() - 1];

            ctxt->values->applyMemory(_ctxt, rd);

            switch (_ctxt) {
                case GPCorrelationContext::None:
                {
                    gpSamples = gp->sample(
                        points.data(), derivs.data(), _samplePoints, nullptr,
                        nullptr, 0,
                        rd, 1, sampler);
                    break;
                }
                case GPCorrelationContext::Renewal:
                {
                    std::array<Vec3d, 1> cond_pts = { lastIntersectPt };
                    std::array<Derivative, 1> cond_deriv = { Derivative::None };

                    gpSamples = gp->sample_cond(
                        points.data(), derivs.data(), _samplePoints, nullptr,
                        cond_pts.data(), ctxt->values.get(), cond_deriv.data(), cond_pts.size(), nullptr,
                        nullptr, 0,
                        rd, 1, sampler);
                    break;
                }
                case GPCorrelationContext::RenewalPlus:
                {
                    std::array<Vec3d, 2> cond_pts = { lastIntersectPt, lastIntersectPt };
                    std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };

                    gpSamples = gp->sample_cond(
                        points.data(), derivs.data(), _samplePoints, nullptr,
                        cond_pts.data(), ctxt->values.get(), cond_deriv.data(), cond_pts.size(), nullptr,
                        nullptr, 0,
                        rd, 1, sampler);
                    break;
                }
                case GPCorrelationContext::Global:
                {
                    std::vector<Vec3d> cond_pts = ctxt->points;
                    std::vector<Derivative> cond_derivs = ctxt->derivs;

                    if (!std::static_pointer_cast<GPRealNodeValues>(ctxt->values)->_isIntersect) {
                        cond_pts.push_back(lastIntersectPt);
                        cond_derivs.push_back(Derivative::First);
                    }

                    gpSamples = gp->sample_cond(
                        points.data(), derivs.data(), _samplePoints, nullptr,
                        cond_pts.data(), ctxt->values.get(), cond_derivs.data(), cond_pts.size(), nullptr,
                        nullptr, 0,
                        rd, 1, sampler);
                    break;
                }
            }
        }

        auto [sampleValues, sampleIds] = gpSamples->flatten();

        double prevV = sampleValues(0);

        int sign0 = prevV < 0 ? -1 : 1;

        // Note: related to whether conditioning on a positive start value or not
//        prevV = max(prevV, 0.);

        double prevT = ts[0];
        for (int p = 1; p < _samplePoints; p++) {
            double currV = sampleValues(p);
            double currT = ts[p];

            int signc = currV < 0 ? -1 : 1;

            // if (currV < 0) {
            if (!state.firstScatter && firstIsectAlongRay && p == 1) {
                sign0 = signc;
            }
            else if (signc != sign0) {
                double offsetT = 0;
                t = (points[p-1][0] - ro[0]) / rd[0];

                offsetT = prevV / (prevV - currV);
                t = lerp(prevT, currT, offsetT);

                if (t >= maxT) {
                    std::cerr << "Somehow got a distance that's greater than the max distance.\n";
                }

                derivs.resize(p + 2);
                points.resize(p + 2);

                gpSamples->makeIntersect(p, offsetT, prevT - currT);

                points[p] = ro + t * rd;
                derivs[p] = Derivative::None;

                points[p+1] = ro + t * rd;
                derivs[p+1] = Derivative::First;

                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = std::move(derivs);
                ctxt->points = std::move(points);
                ctxt->values = gpSamples;
                state.gpContext = ctxt;
                state.lastGPId = sampleIds(p,0);

                if (_surfVolPhaseSeparate) {
                    if (gp->_cov->getUnscaledVariance(ro + t * rd) < _surfVolPhaseAmpThresh)
                        state.lastGPId = 0;
                    else
                        state.lastGPId = 1;
                }

                return true;
            }
            prevV = currV;
            prevT = currT;
        }

        t = maxT;
        auto ctxt_ret = std::make_shared<GPContextFunctionSpace>();
        ctxt_ret->derivs = std::move(derivs);
        ctxt_ret->points = std::move(points);
        ctxt_ret->values = gpSamples;
        state.gpContext = ctxt_ret;
        state.lastGPId = sampleIds(sampleIds.size()-1,0);

        return false;
    }

    bool FunctionSpaceGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& isect_p, const float&, MediumState& state, Vec3d& grad) const {
        if (!state.gpContext)
            return false;

        GPContextFunctionSpace& ctxt = *(GPContextFunctionSpace*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch (_normalSamplingMethod) {
        case GPNormalSamplingMethod::FiniteDifferences: // Get rid of this for simplicity
        case GPNormalSamplingMethod::ConditionedGaussian:
        {
            if (ctxt.points.size() == 0)
                return false;

            auto gpRealNodeValues = std::static_pointer_cast<GPRealNodeValues>(ctxt.values);
            gpRealNodeValues->_gp = _gp->get_gaussian_process_from_id(state.lastGPId);
            ctxt.values->sampleGrad(state.lastGPId, isect_p, rd, ctxt.points.data(), ctxt.derivs.data(), sampler, grad);

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
            float alpha = std::min(_gp->compute_beckmann_roughness(isect_p), 10.);
            BeckmannNDF ndf(0, alpha, alpha);

            grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
            if (ctxt.values)
                ctxt.values->setGrad(grad);
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
            if (ctxt.values)
                ctxt.values->setGrad(grad);
            break;
        }
        }

        return true;
    }
    

    /*Vec3f FunctionSpaceGaussianProcessMedium::transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface,
        bool endOnSurface, MediumSample * sample) const
    {
        if (ray.farT() == Ray::infinity())
            return Vec3f(0.0f);

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch (_intersectMethod) {
            case GPIntersectMethod::GPDiscrete:
            {
                std::vector<Vec3d> points(_samplePoints);
                std::vector<Derivative> derivs(_samplePoints);

                for (int i = 0; i < points.size(); i++) {
                    float t = lerp(ray.nearT(), ray.farT(), (i + sampler.next1D()) / _samplePoints);
                    points[i] = vec_conv<Vec3d>(ray.pos() + t * ray.dir());
                    derivs[i] = Derivative::None;
                }

                Eigen::MatrixXd gpSamples;
                int startSign = 1;

                if (startOnSurface) {
                    std::array<Vec3d, 1> cond_pts = { points[0] };
                    std::array<Derivative, 1> cond_deriv = { Derivative::None };
                    std::array<double, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
                    std::array<Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
                    gpSamples = _gp->sample_cond(
                        points.data(), derivs.data(), points.size(), nullptr,
                        cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                        constraints.data(), constraints.size(),
                        rd, 10, sampler);
                }
                else {
                    if (!sample) {
                        std::cout << "what\n";
                        return Vec3f(0.f);
                    }

                    std::array<Vec3d, 2> cond_pts = { points[0], points[0] };
                    std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };

                    double deriv = sample->aniso.dot(vec_conv<Vec3d>(ray.dir().normalized()));
                    startSign = deriv < 0 ? -1 : 1;
                    std::array<double, 2> cond_vs = { 0, deriv };

                    gpSamples = _gp->sample_cond(
                        points.data(), derivs.data(), points.size(), nullptr,
                        cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                        nullptr, 0,
                        rd, 10, sampler) * startSign;
                }

                int madeItCnt = 0;
                for (int s = 0; s < gpSamples.cols(); s++) {

                    bool madeIt = true;
                    for (int p = 0; p < _samplePoints; p++) {
                        if (gpSamples(p, s) < 0) {
                            madeIt = false;
                            break;
                        }
                    }

                    if (madeIt) madeItCnt++;
                }

                return Vec3f(float(madeItCnt) / gpSamples.cols());
            }
            case GPIntersectMethod::Mean:
            {
                MediumState state;
                state.firstScatter = startOnSurface;
                double t;
                return intersectMean(sampler, ray, state, t) ? Vec3f(0.f) : Vec3f(1.f);
            }
        }
    }*/
}
