#include "SparseConvolutionNoise.hpp"
#include "sampling/UniformPathSampler.hpp"

#include <sampling/Gaussian.hpp>
// #include <sstream>

namespace Tungsten {

SparseConvolutionNoiseRealization::SparseConvolutionNoiseRealization(
        std::shared_ptr<GaussianProcess> gp, GPCorrelationContext ctxt,
        uint globalSeed, float impulseDensity, bool useSingleRealization,
        bool isotropicSpace3DSampling, bool oneDSampling, 
        SparseConv1DSamplingScheme oneDSamplingScheme, Vec3f oneDSamplingSchemeSampleCountScale,
        bool correlationXY,
        bool surfVolPhaseSeparate, float surfVolPhaseAmpThresh) :
    _gp(gp), _ctxt(ctxt),
    _globalSeed(globalSeed), _impulseDensity(impulseDensity), _useSingleRealization(useSingleRealization),
    _isotropicSpace3DSampling(isotropicSpace3DSampling),
    _isotropicRaySpace3DSampling(true),
    _1DSampling(oneDSampling), _correlationXY(correlationXY),
    _surfVolPhaseSeparate(surfVolPhaseSeparate), _surfVolPhaseAmpThresh(surfVolPhaseAmpThresh)
    {
        _activateConditioning = !_useSingleRealization && (_ctxt == GPCorrelationContext::Renewal || _ctxt == GPCorrelationContext::RenewalPlus);

        _1DsamplingScheme = SparseConv1DSamplingScheme::UNI;
        _1DsamplingSchemeSampleCountScale = Vec3f(1.f, 0.f, 0.f);
        if (!_useSingleRealization && _1DSampling) {
            _1DsamplingScheme = oneDSamplingScheme;
            _1DsamplingSchemeSampleCountScale = oneDSamplingSchemeSampleCountScale;
        }

        // Use multi-resolution sparse convolution noise [Lagae et al. 2011] to efficiently approximate
        // a non-stationary GP with only variation in scale (but not anisotropy)
        _multiResolutionGrid = !_gp->_cov->isStationaryKernel() && _gp->_cov->useMultiResolutionGrid();

        _base = 2.5; // Empirical parameter for multi-resolution noise
        coeff_3D.value_scale = 0.;
        coeff_3D.gradient_scale = Vec3f(0.);
        coeff_1D.value_scale = 0.;
        coeff_1D.gradient_scale = Vec3f(0.f);
    }

// Sparse convolution noise utility functions
inline uint SparseConvolutionNoiseRealization::computeSeed(const Vec4u pixelSampleSegment, const uint sceneSeed) const {
    uint seed = _globalSeed;
    if (!_useSingleRealization) {
        Vec4u pixelSampleSegment_proxy = pixelSampleSegment;
        if (_ctxt == GPCorrelationContext::Global)
            pixelSampleSegment_proxy.w() = 0;
        seed += MathUtil::xxhash32(pixelSampleSegment_proxy) + MathUtil::xxhash32(sceneSeed);
    }
    return seed;
}

std::pair<SparseConv1DSamplingScheme, Vec3f> SparseConvolutionNoiseRealization::samplingScheme(const Vec3f& p) const {
    float amplitude = _gp->_cov->sparseConvNoiseAmplitude(p);
    // If the amplitude is too small to see any noise, disable any NEE/MIS strategy
    if (amplitude < 1e-6f)
        return {SparseConv1DSamplingScheme::UNI, Vec3f(1.f, 0.f, 0.f)};
    else
        return {_1DsamplingScheme, _1DsamplingSchemeSampleCountScale};
}

Vec4f SparseConvolutionNoiseRealization::kernelScaleLevelRatio(const Vec3f& p) const {
    float lateral_scale = _gp->_cov->sparseConvNoiseLateralScale(p);
    float level_low = floor(log(lateral_scale) / log(_base));
    float level_high = level_low + 1;
    float lateral_scale_low = pow(_base, level_low);
    float lateral_scale_high = pow(_base, level_high);
    float frac = (lateral_scale - lateral_scale_low) / (lateral_scale_high - lateral_scale_low);
    float ratio_low = (1.0 - frac) / sqrt(1.0 - 2.0 * frac + 2.0 * frac * frac);
    float ratio_high = frac / sqrt(1.0 - 2.0 * frac + 2.0 * frac * frac);
    return Vec4f(lateral_scale_low, lateral_scale_high, ratio_low, ratio_high);
}

// Evaluate GPIS Value
float SparseConvolutionNoiseRealization::evaluateValue(const Vec3f& p, const float t, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler, int& GPId) {
    uint seed = computeSeed(info.pixelSampleSegment, info.sceneSeed);
    float amplitude = _gp->_cov->sparseConvNoiseAmplitude(p);
    float noise_val = _1DSampling ? evaluateValueNoise1D(p, t + info.t, rayDir, seed, sampler, true) : evaluateValueNoise3D(p, rayDir, seed, sampler, true);
    Eigen::Vector2d mean_and_id = _gp->mean_weight_space(Vec3d(p), Derivative::None);
    GPId = (int)mean_and_id(1);

    // Tricky way to assign different phase functions, only used for the coffee mug scene
    if (_surfVolPhaseSeparate) {
        if (_gp->_cov->getUnscaledVariance(Vec3d(p)) < _surfVolPhaseAmpThresh)
            GPId = 0;
        else
            GPId = 1;
    }

    return amplitude * noise_val + mean_and_id(0);
}

// Evaluate GPIS Gradient
Vec3f SparseConvolutionNoiseRealization::evaluateGradient(const Vec3f& p, const float t, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler) {
    uint seed = computeSeed(info.pixelSampleSegment, info.sceneSeed);
    float amplitude = _gp->_cov->sparseConvNoiseAmplitude(p);
    Vec3f noise_grad = _1DSampling ? evaluateGradientNoise1D(p, t + info.t, t, rayDir, info, sampler, true) : evaluateGradientNoise3D(p, rayDir, seed, sampler, true);
    Eigen::Vector2d mean_and_id = _gp->mean_weight_space(Vec3d(p), Derivative::None);
    Vec3f mean_grad = Vec3f(mean_and_id(1) == _gp->_id ? _gp->_mean->dmean_da(Vec3d(p)) : _gp->_mean_additional->dmean_da(Vec3d(p)));
    return amplitude * noise_grad + mean_grad;
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) {
    if (!_isotropicSpace3DSampling) {
        if (!_multiResolutionGrid) {
            float kernelSpatialScale = _gp->_cov->worldSamplingSpatialScale();
            return evaluateNoise3DNormalized(p, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(false, 1.0), kernelSpatialScale, conditioning);
        }
        else {
            Vec4f info = kernelScaleLevelRatio(p);
            Vec4f noise_low = evaluateNoise3DNormalized(p, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(false, info.x()), info.x(), conditioning);
            Vec4f noise_high = evaluateNoise3DNormalized(p, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(false, info.y()), info.y(), conditioning);
            return info.z() * noise_low + info.w() * noise_high;
        }
    }
    else {
        if (!_multiResolutionGrid)
            return evaluateNoise3DIsotropicNormalizedSelect(p, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), 1.0, conditioning);
        else {
            Vec4f info = kernelScaleLevelRatio(p);
            Vec4f noise_low = evaluateNoise3DIsotropicNormalizedSelect(p, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), info.x(), conditioning);
            Vec4f noise_high = evaluateNoise3DIsotropicNormalizedSelect(p, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), info.y(), conditioning);
            return info.z() * noise_low + info.w() * noise_high;
        }
    }
}

float SparseConvolutionNoiseRealization::evaluateValueNoise1D(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) {
    if (!_multiResolutionGrid)
        return evaluateNoise1DNormalized(p, t, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), 1.0, conditioning).x();
    else {
        Vec4f info = kernelScaleLevelRatio(p);
        float noise_low = evaluateNoise1DNormalized(p, t, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), info.x(), conditioning, true).x();
        float noise_high = evaluateNoise1DNormalized(p, t, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), info.y(), conditioning, false).x();
        return info.z() * noise_low + info.w() * noise_high;
    }
}

Vec3f SparseConvolutionNoiseRealization::evaluateGradientNoise1D(const Vec3f& p, const float tTotal, const float tSegment, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler, bool conditioning) {
    uint seed = computeSeed(info.pixelSampleSegment, info.sceneSeed);

    Vec3f ray_dir_iso = _gp->transformPosDirWorldtoLocal(p, rayDir, 1.0).normalized();
    TangentFrame coord = TangentFrame(Vec3f(ray_dir_iso));

    // For non-stationary anisotropy
    float scaleX = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, coord.tangent);
    float scaleY = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, coord.bitangent);
    Eigen::VectorXd xy_scale(2);
    xy_scale(0) = 1.0 / scaleX, xy_scale(1) = 1.0 / scaleY;

    if (!_multiResolutionGrid) {
        Vec3f grad_iso_ray_base = evaluateNoise1DNormalized(p, tTotal, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), 1.0, conditioning).yzw();
        sampler.set_state(MathUtil::xxhash32(info.pixelSampleSegment) + MathUtil::xxhash32(info.sceneSeed) + 1u);
        Eigen::VectorXd xy_1 = sample_standard_normal(2, sampler) / sqrt(2.f);
        Vec3f grad_iso_ray;
        if (tSegment == 0) {
            // Evaluate the gradient at the start of the ray
            // Called when conditioning for 1D GPIS Renewal+, in conditioning1D(..)
            xy_1 = xy_1.cwiseProduct(xy_scale);
            grad_iso_ray = Vec3f(xy_1(0), xy_1(1), 0.f) + grad_iso_ray_base;
        }
        else {
            // Evaluate the gradient at the current intersection (i.e. the end of the ray)
            if (!(_ctxt == GPCorrelationContext::RenewalPlus && _correlationXY)) {
                Eigen::VectorXd xy_2 = sample_standard_normal(2, sampler) / sqrt(2.f);
                xy_2 = xy_2.cwiseProduct(xy_scale);
                // Need a second sampling process to avoid using the same xy as tSegment == 0!!
                grad_iso_ray = Vec3f(xy_2(0), xy_2(1), 0.f) + grad_iso_ray_base;
            }
            else {
                // Note: _correlationXY only works for stationary GP; the logic here is for an isotropic convolution kernel
                if (!_gp->_cov->isStationaryKernel()) {
                    std::cerr << "(Unidirectional) 1D CorrelationXY does not work for brute-force non-stationary covariance kernel!\n";
                }
                // Consider the covariance between the xy gradients at the previous path vertex (on the same GPIS) and those at the current intersection
                // Considering/ignoring this covariance only makes a difference under the Renewal+ memory model
                // This distinguishes between Renewal+ (consider) and Renewal Half+ (ignore)
                // Not much difference in the final converged rendering, but NEE (and thus MIS) provides more noise reduction under Renewal Half+
                float factor = std::exp(- tSegment * tSegment / 4.0) * (0.5 - tSegment * tSegment * 0.25);
                Eigen::VectorXd mu = factor * 2.0 * xy_1;
                float cov = 0.5 - factor * factor * 2;
                Eigen::VectorXd xy_2 = sample_standard_normal(2, sampler) * sqrt(cov) + mu;
                grad_iso_ray = Vec3f(xy_2(0), xy_2(1), 0.f) + grad_iso_ray_base;
            }
        }
        Vec3f grad_iso = Vec3f(coord.toGlobal(Vec3f(grad_iso_ray)));
        Vec3f grad_world = _gp->transformGradLocaltoWorld(p, grad_iso, 1.0);
        return grad_world;
    }
    else {
        // The same logic as above, extended to multi-resolution noise
        Vec4f scaleInfo = kernelScaleLevelRatio(p);
        Vec3f grad_iso_ray_base_low = evaluateNoise1DNormalized(p, tTotal, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), scaleInfo.x(), conditioning, true).yzw();
        Vec3f grad_iso_ray_base_high = evaluateNoise1DNormalized(p, tTotal, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), scaleInfo.y(), conditioning, false).yzw();
        sampler.set_state(MathUtil::xxhash32(info.pixelSampleSegment) + MathUtil::xxhash32(info.sceneSeed) + 1u);
        Eigen::VectorXd xy_1_low = sample_standard_normal(2, sampler) / sqrt(2.f);
        Eigen::VectorXd xy_1_high = sample_standard_normal(2, sampler) / sqrt(2.f);
        Vec3f grad_iso_ray_low, grad_iso_ray_high;
        if (tSegment == 0) {
            xy_1_low = xy_1_low.cwiseProduct(xy_scale);
            xy_1_high = xy_1_high.cwiseProduct(xy_scale);
            grad_iso_ray_low = Vec3f(xy_1_low(0), xy_1_low(1), 0.f) + grad_iso_ray_base_low;
            grad_iso_ray_high = Vec3f(xy_1_high(0), xy_1_high(1), 0.f) + grad_iso_ray_base_high;
        }
        else {
            if (!(_ctxt == GPCorrelationContext::RenewalPlus && _correlationXY)) {
                Eigen::VectorXd xy_2_low = sample_standard_normal(2, sampler) / sqrt(2.f);
                Eigen::VectorXd xy_2_high = sample_standard_normal(2, sampler) / sqrt(2.f);
                // Need a second sampling process to avoid using the same xy as tSegment == 0!!
                xy_2_low = xy_2_low.cwiseProduct(xy_scale);
                xy_2_high = xy_2_high.cwiseProduct(xy_scale);
                grad_iso_ray_low = Vec3f(xy_2_low(0), xy_2_low(1), 0.f) + grad_iso_ray_base_low;
                grad_iso_ray_high = Vec3f(xy_2_high(0), xy_2_high(1), 0.f) + grad_iso_ray_base_high;
            }
            else {
                // Note: the logic only works for covariance kernels with non-stationary scale but not non-stationary anisotropy
                if (_gp->_cov->isNonstationaryAnisotropicKernel()) {
                    std::cerr << "(Unidirectional) 1D CorrelationXY does not work for covariance kernel with non-stationary anisotropy!\n";
                }
                // Consider the correlation of gx and gy at ray start (tSegment == 0) and at ray end (here)
                float factor = std::exp(- tSegment * tSegment / 4.0) * (0.5 - tSegment * tSegment * 0.25);
                Eigen::VectorXd mu_low = factor * 2.0 * xy_1_low;
                Eigen::VectorXd mu_high = factor * 2.0 * xy_1_high;
                float cov = 0.5 - factor * factor * 2;
                Eigen::VectorXd xy_2_low = sample_standard_normal(2, sampler) * sqrt(cov) + mu_low;
                Eigen::VectorXd xy_2_high = sample_standard_normal(2, sampler) * sqrt(cov) + mu_high;
                grad_iso_ray_low = Vec3f(xy_2_low(0), xy_2_low(1), 0.f) + grad_iso_ray_base_low;
                grad_iso_ray_high = Vec3f(xy_2_high(0), xy_2_high(1), 0.f) + grad_iso_ray_base_high;
            }
        }
        Vec3f ray_dir_iso_low = _gp->transformPosDirWorldtoLocal(p, rayDir, scaleInfo.x()).normalized();
        Vec3f ray_dir_iso_high = _gp->transformPosDirWorldtoLocal(p, rayDir, scaleInfo.y()).normalized();
        TangentFrame coord_low = TangentFrame(Vec3f(ray_dir_iso_low));
        TangentFrame coord_high = TangentFrame(Vec3f(ray_dir_iso_high));
        Vec3f grad_iso_low = Vec3f(coord_low.toGlobal(Vec3f(grad_iso_ray_low)));
        Vec3f grad_iso_high = Vec3f(coord_high.toGlobal(Vec3f(grad_iso_ray_high)));
        Vec3f grad_world_low = _gp->transformGradLocaltoWorld(p, grad_iso_low, scaleInfo.x());
        Vec3f grad_world_high = _gp->transformGradLocaltoWorld(p, grad_iso_high, scaleInfo.y());
        return scaleInfo.z() * grad_world_low + scaleInfo.w() * grad_world_high;
    }
}

// Three spaces:
//      - World space
//      - Isotropic space: scale the world space s.t. the convolution kernel become isotropic in this space
//      - Isotropic ray space: rotate the isotropic space s.t. the ray direction is aligned with the Z axis

// Parameters:
//      - kernelRadius: cell size in the space of the splatting kernel (e.g., world or isotropic ray space)
//      - kernelSpatialScale: the grid scaling factor for the multi-resolution grid approach, 1.0 otherwise; actually used to inversely transform the point, so that this is equivalent to grid scaling

// Evaluate 3D noise in world space
Vec4f SparseConvolutionNoiseRealization::evaluateNoise3DNormalized(const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise
    Vec4f noise = noise3D(p, p, seed + additional_seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale);
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance3D(p, impulseDensity, kernelRadius, false, kernelSpatialScale));
    noise /= normalization_factor;
    if (_activateConditioning && conditioning) {
        noise += _gp->_cov->splattingKernel3D(p, coeff_3D.ray_origin, true, false, kernelSpatialScale, p) * coeff_3D.value_scale + _gp->_cov->splattingKernel3DGrad(p, coeff_3D.ray_origin, coeff_3D.gradient_scale, true, false, kernelSpatialScale, p);
    }
    return noise;
}

inline Vec4f SparseConvolutionNoiseRealization::evaluateNoise3DIsotropicNormalizedSelect(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    if (_isotropicRaySpace3DSampling) // Evaluate 3D noise in isotropic ray space
        return evaluateNoise3DIsotropicRayNormalized(p, rayDir, seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale, conditioning);
    else // Evaluate 3D noise in isotropic space
        return evaluateNoise3DIsotropicNormalized(p, rayDir, seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale, conditioning);
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise3DIsotropicNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    // Transform the point from world space into isotropic space
    Vec3f p_iso = _gp->transformPosDirWorldtoLocal(p, p, kernelSpatialScale);
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise
    // Evaluate the noise in isotropic space
    Vec4f noise_iso = noise3D(p, p_iso, seed + additional_seed, sampler, impulseDensity, kernelRadius, 1.0);
    Vec3f grad_iso = noise_iso.yzw();
    // Transform the gradient from isotropic space back to world space
    Vec3f grad_world = _gp->transformGradLocaltoWorld(p, grad_iso, kernelSpatialScale);
    Vec4f noise_world = Vec4f(noise_iso.x(), grad_world.x(), grad_world.y(), grad_world.z());
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance3D(p, impulseDensity, kernelRadius, true, 1.0));
    noise_world /= normalization_factor;
    if (_activateConditioning && conditioning) {
        Vec3f origin_iso = _gp->transformPosDirWorldtoLocal(p, coeff_3D.ray_origin, kernelSpatialScale);
        Vec4f noise_delta_iso = _gp->_cov->splattingKernel3D(p_iso, origin_iso, true, true, 1.0, p) * coeff_3D.value_scale + _gp->_cov->splattingKernel3DGrad(p_iso, origin_iso, coeff_3D.gradient_scale, true, true, 1.0, p);
        Vec3f grad_delta_iso = noise_delta_iso.yzw();
        Vec3f grad_delta_world = _gp->transformGradLocaltoWorld(p, grad_delta_iso, kernelSpatialScale);
        noise_world += Vec4f(noise_delta_iso.x(), grad_delta_world.x(), grad_delta_world.y(), grad_delta_world.z());
    }
    return noise_world;
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise3DIsotropicRayNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    // Note: Only isotropic space supports global model, but not isotropic ray space. We could probably reconcile this with more spatial transform.
    // So the current code only achieves global model through the world space implementation (evaluateNoise3DNormalized)

    // Transform the point from world space into isotropic ray space
    Vec3f ray_dir_iso = _gp->transformPosDirWorldtoLocal(p, rayDir, 1.0).normalized();
    TangentFrame coord = TangentFrame(ray_dir_iso);
    Vec3f p_iso = _gp->transformPosDirWorldtoLocal(p, p, kernelSpatialScale);
    Vec3f p_iso_ray = coord.toLocal(p_iso);
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise
    // Evaluate the noise in isotropic ray space
    Vec4f noise_iso_ray = noise3D(p, p_iso_ray, seed + additional_seed, sampler, impulseDensity, kernelRadius, 1.0);
    Vec3f grad_iso_ray = noise_iso_ray.yzw();
    // Transform the gradient from isotropic ray space back to world space
    Vec3f grad_iso = Vec3f(coord.toGlobal(Vec3f(grad_iso_ray)));
    Vec3f grad_world = _gp->transformGradLocaltoWorld(p, grad_iso, kernelSpatialScale);
    Vec4f noise_world = Vec4f(noise_iso_ray.x(), grad_world.x(), grad_world.y(), grad_world.z());
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance3D(p, impulseDensity, kernelRadius, true, 1.0));
    noise_world /= normalization_factor;
    if (_activateConditioning && conditioning) {
        Vec3f origin_iso = _gp->transformPosDirWorldtoLocal(p, coeff_3D.ray_origin, kernelSpatialScale);
        Vec3f origin_iso_ray = coord.toLocal(origin_iso);
        Vec4f noise_delta_iso_ray = _gp->_cov->splattingKernel3D(p_iso_ray, origin_iso_ray, true, true, 1.0, p) * coeff_3D.value_scale + _gp->_cov->splattingKernel3DGrad(p_iso_ray, origin_iso_ray, coeff_3D.gradient_scale, true, true, 1.0, p);
        Vec3f grad_delta_iso_ray = noise_delta_iso_ray.yzw();
        Vec3f grad_delta_iso = Vec3f(coord.toGlobal(Vec3f(grad_delta_iso_ray)));
        Vec3f grad_delta_world = _gp->transformGradLocaltoWorld(p, grad_delta_iso, kernelSpatialScale);
        noise_world += Vec4f(noise_delta_iso_ray.x(), grad_delta_world.x(), grad_delta_world.y(), grad_delta_world.z());
    }
    return noise_world;
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise1DNormalized(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning, bool multiResLowLevel) {
    // The logic is similar to evaluateNoise3DIsotropicRayNormalized, but evaluating 1D noise along the ray in isotropic ray space
    Vec3f ray_dir_iso = _gp->transformPosDirWorldtoLocal(p, rayDir, 1.0).normalized();
    TangentFrame coord = TangentFrame(ray_dir_iso);
    Vec3f p_iso = _gp->transformPosDirWorldtoLocal(p, p, kernelSpatialScale);
    Vec3f p_iso_ray = coord.toLocal(p_iso);
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise

    Vec2f noise = noise1D(p, rayDir, p_iso_ray.z(), seed + additional_seed, sampler, impulseDensity, kernelRadius);
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance1D(p, impulseDensity, kernelRadius));
    noise /= normalization_factor;

    // Only need to account for non-zero xy components when considering the correlation with the previous vertex (see comments in evaluateGradientNoise1D(..))
    float grad_condition_splat_x = 0.f;
    float grad_condition_splat_y = 0.f;
    if (_activateConditioning && conditioning) {
        float origin_scale_factor = 1.0;
        if (_multiResolutionGrid) {
            Vec4f origin_info = kernelScaleLevelRatio(coeff_1D.ray_origin);
            origin_scale_factor = multiResLowLevel ? origin_info.z() : origin_info.w();
        }

        Vec3f origin_iso = _gp->transformPosDirWorldtoLocal(p, coeff_1D.ray_origin, kernelSpatialScale);
        Vec3f origin_iso_ray = coord.toLocal(origin_iso);
        Vec2f value_condition_splat = coeff_1D.value_scale *
                _gp->_cov->covarianceKernel1D(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, ray_dir_iso);
        Vec2f grad_condition_splat_z = kernelSpatialScale * coeff_1D.gradient_scale.z() *
                _gp->_cov->covarianceKernel1DGrad(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, ray_dir_iso);
        noise += origin_scale_factor * (value_condition_splat + grad_condition_splat_z);
        if (_correlationXY) {
            // Consider the effect of the conditioning kernels placed at the start of the ray
            grad_condition_splat_x = origin_scale_factor * kernelSpatialScale * coeff_1D.gradient_scale.x() * _gp->_cov->covarianceKernel1DGradFor3DNormal(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, coord.tangent);
            grad_condition_splat_y = origin_scale_factor * kernelSpatialScale * coeff_1D.gradient_scale.y() * _gp->_cov->covarianceKernel1DGradFor3DNormal(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, coord.bitangent);
        }
    }
    Vec4f noise_full = Vec4f(noise.x(), grad_condition_splat_x, grad_condition_splat_y, noise.y());
    // The gradient is not transformed to world space here, but in evaluateGradientNoise1D()
    return noise_full;
}

Vec4f SparseConvolutionNoiseRealization::noise3D(const Vec3f& p_world, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const {
    Vec3f p_grid = p / kernelRadius;
    Vec3f frac = Vec3f(p_grid - floor(p_grid));
    Vec3i ijk = Vec3i(floor(p_grid));
    Vec4f sum = Vec4f(0.0);

    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz)
                sum += cell3D(p_world, Vec3u(ijk + Vec3i(dx, dy, dz)), frac - Vec3f((float)dx, (float)dy, (float)dz), seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale);
    return sum;
}

Vec4f SparseConvolutionNoiseRealization::cell3D(const Vec3f& p_world, const Vec3u& ijk, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const{
    sampler.set_state(MathUtil::xxhash32(Vec4u(ijk.z(), ijk.y(), ijk.x(), seed)) + 1u);

    // Replace the Poisson point distribution with fixed number of splats per cell for efficiency ([Tavernier et al. 2019])
    // uint number_of_impulses = poisson(impulse_density_per_kernel, sampler);
    uint number_of_impulses = uint(impulseDensity);

    Vec4f sum = Vec4f(0.f);
    for (uint k = 0u; k < number_of_impulses; ++k) {
        Vec3f p_i = Vec3f(sampler.next3D());
        float w_i = MathUtil::Bernoulli(sampler.next1D(), -1.f, 1.f, 0.5f); // Bernoulli distribution
        // float w_i = (sampler.next1f() * 2.0 - 1.0) * sqrt(3.0); // Uniform distribution
        Vec3f to_point = p - p_i;

        if (to_point.lengthSq() < 1.0) {
            sum += w_i * _gp->_cov->splattingKernel3D(kernelRadius * p, kernelRadius * p_i, false, _isotropicSpace3DSampling, kernelSpatialScale, p_world);
        }
    }

    return sum;
}

Vec2f SparseConvolutionNoiseRealization::noise1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const {
    float t_grid = t / kernelRadius;
    float frac = t_grid - floor(t_grid);
    int i = int(floor(t_grid));
    Vec2f sum = Vec2f(0.0);

    for (int dx = -1; dx <= 1; ++dx)
        sum += cell1D(p_world, ray_dir_world, uint(i + dx), frac - dx, seed, sampler, impulseDensity, kernelRadius);
    return sum;
}

Vec2f SparseConvolutionNoiseRealization::cell1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const uint i, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const {
    sampler.set_state(MathUtil::xxhash32(Vec2u(i, seed)) + 1u);

    // Replace the Poisson point distribution with fixed number of splats per cell for efficiency ([Tavernier et al. 2019])
    // uint number_of_impulses = poisson(impulse_density_per_kernel, sampler);
    uint number_of_impulses = uint(impulseDensity);

    Vec2f sum = Vec2f(0.f);
    for (uint k = 0u; k < number_of_impulses; ++k) {
        float t_i = sampler.next1D();
        float w_i = MathUtil::Bernoulli(sampler.next1D(), -1.f, 1.f, 0.5f); // Bernoulli distribution
        // float w_i = (sampler.next1f() * 2.0 - 1.0) * sqrt(3.0); // Uniform distribution
        float to_point = t - t_i;

        if (to_point * to_point < 1.0) {
            // Note: take the length scale parameter from the query location (not the kernel center)
            sum += w_i * _gp->_cov->splattingKernel1D(kernelRadius * t, kernelRadius * t_i, p_world, ray_dir_world);
        }
    }
    return sum;
}

// Perform conditioning (Renewal/Renewal+), support both 3D and 1D GPISes
void SparseConvolutionNoiseRealization::conditioning(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler) {
    if (!_activateConditioning)
        return;

    if (_1DSampling)
        conditioning1D(p, rayDir, targetVal, targetGrad, info, sampler);
    else
        conditioning3D(p, rayDir, targetVal, targetGrad, info, sampler);
}

void SparseConvolutionNoiseRealization::conditioning3D(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler) {
    coeff_3D.value_scale = 0.;
    coeff_3D.gradient_scale = Vec3f(0.);
    coeff_3D.ray_origin = p;

    float amplitude;
    Eigen::Vector2d mean_and_id;
    Vec3f mean_grad;
    Vec4f kernelScaleInfo;
    if (_multiResolutionGrid) {
        kernelScaleInfo = kernelScaleLevelRatio(p);
    }

    uint seed = computeSeed(info.pixelSampleSegment, info.sceneSeed);

    // Conditioning for the value
    if (_ctxt == GPCorrelationContext::Renewal || _ctxt == GPCorrelationContext::RenewalPlus) {
        amplitude = _gp->_cov->sparseConvNoiseAmplitude(p);
        if (amplitude == 0)
            return;
        mean_and_id = _gp->mean_weight_space(Vec3d(p), Derivative::None);
        float mean = mean_and_id(0);
        float current_value = evaluateValueNoise3D(p, rayDir, seed, sampler, false);
        coeff_3D.value_scale = (targetVal - mean) / amplitude - current_value;
        if (_multiResolutionGrid) {
            coeff_3D.value_scale /= kernelScaleInfo.z() + kernelScaleInfo.w();
        }

        // Sanity check for value
        float val = mean + amplitude * evaluateValueNoise3D(p, rayDir, seed, sampler, true);
        // std::cout << "Conditioning 3D value: " << targetVal - val << std::endl;
        if (fabs(targetVal - val) > 1e-2) {
            std::cerr << "Conditioning 3D value error: " << targetVal - val << std::endl;
        }
    }

    // Conditioning for the 3D gradient
    if (_ctxt == GPCorrelationContext::RenewalPlus) {
        mean_grad = Vec3f(mean_and_id(1) == _gp->_id ? _gp->_mean->dmean_da(Vec3d(p)) : _gp->_mean_additional->dmean_da(Vec3d(p)));
        Vec3f current_grad_world = evaluateGradientNoise3D(p, rayDir, seed, sampler, false);
        Vec3f delta = (targetGrad - mean_grad) / amplitude - current_grad_world;
        if (_isotropicSpace3DSampling) {
            Vec3f gradient_scale_iso = _gp->transformGradWorldtoLocal(p, delta, 1.0);

            if (_isotropicRaySpace3DSampling) {
                Vec3f ray_dir_iso = _gp->transformPosDirWorldtoLocal(p, rayDir, 1.0).normalized();
                TangentFrame coord = TangentFrame(Vec3f(ray_dir_iso));
                gradient_scale_iso = Vec3f(coord.toLocal(gradient_scale_iso));
            }

            coeff_3D.gradient_scale = to_vec3f(_gp->_cov->sparseConvNoiseOneOverSecondDerivative(p, nullptr, true) * to_eigen3f(gradient_scale_iso));

            if (_multiResolutionGrid) {
                coeff_3D.gradient_scale /= kernelScaleInfo.z() / kernelScaleInfo.x() + kernelScaleInfo.w() / kernelScaleInfo.y();
            }
            else {
                coeff_3D.gradient_scale *= sqr(_gp->_cov->nonStationarySplattingKernelScale(p));
            }
        }
        else {
            coeff_3D.gradient_scale = to_vec3f(_gp->_cov->sparseConvNoiseOneOverSecondDerivative(p, nullptr, false) * to_eigen3f(delta));

            if (_multiResolutionGrid) {
                coeff_3D.gradient_scale /= kernelScaleInfo.z() / (kernelScaleInfo.x() * kernelScaleInfo.x()) + kernelScaleInfo.w() / (kernelScaleInfo.y() * kernelScaleInfo.y());
            }
            else {
                coeff_3D.gradient_scale *= sqr(_gp->_cov->sparseConvNoiseLateralScale(p));
            }
        }

        // Sanity check for gradient
        Vec3f grad_diff = targetGrad - mean_grad - amplitude * evaluateGradientNoise3D(p, rayDir, seed, sampler, true);
        if (fabs(grad_diff.length()) > 1e-2) {
            std::cerr << "Conditioning 3D grad error: " << grad_diff << std::endl;
        }
    }
}

void SparseConvolutionNoiseRealization::conditioning1D(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler) {
    coeff_1D.value_scale = 0.;
    coeff_1D.gradient_scale = Vec3f(0.f);
    coeff_1D.ray_origin = p;

    float amplitude;
    Eigen::Vector2d mean_and_id;
    Vec3f mean_grad;
    Vec4f kernelScaleInfo;
    if (_multiResolutionGrid) {
        kernelScaleInfo = kernelScaleLevelRatio(p);
    }

    uint seed = computeSeed(info.pixelSampleSegment, info.sceneSeed);

    // Conditioning for the value
    if (_ctxt == GPCorrelationContext::Renewal || _ctxt == GPCorrelationContext::RenewalPlus) {
        amplitude = _gp->_cov->sparseConvNoiseAmplitude(p);
        if (amplitude == 0)
            return;
        mean_and_id = _gp->mean_weight_space(Vec3d(p), Derivative::None);
        float mean = mean_and_id(0);
        float current_value = evaluateValueNoise1D(p, info.t, rayDir, seed, sampler, false);
        coeff_1D.value_scale = (targetVal - mean) / amplitude - current_value;
        if (_multiResolutionGrid) {
            coeff_1D.value_scale /= sqr(kernelScaleInfo.z()) + sqr(kernelScaleInfo.w());
        }

        // Sanity check for value
        float val = mean + amplitude * evaluateValueNoise1D(p, info.t, rayDir, seed, sampler, true);
        if (fabs(targetVal - val) > 1e-2) {
            std::cerr << "Conditioning 1D value error: " << targetVal - val << std::endl;
        }
    }

    // Conditioning for the 3D gradient (even though we're only generating 1D GPIS!)
    if (_ctxt == GPCorrelationContext::RenewalPlus) {
        mean_grad = Vec3f(mean_and_id(1) == _gp->_id ? _gp->_mean->dmean_da(Vec3d(p)) : _gp->_mean_additional->dmean_da(Vec3d(p)));
        Vec3f current_grad_world = evaluateGradientNoise1D(p, info.t, 0, rayDir, info, sampler, false);
        Vec3f delta = (targetGrad - mean_grad) / amplitude - current_grad_world;

        Vec3f ray_dir_iso = _gp->transformPosDirWorldtoLocal(p, rayDir, 1.0).normalized();
        TangentFrame coord = TangentFrame(Vec3f(ray_dir_iso));

        Vec3f gradient_scale_iso = _gp->transformGradWorldtoLocal(p, delta, 1.0);
        gradient_scale_iso = Vec3f(coord.toLocal(gradient_scale_iso));

        // getNonstationaryCovSplatCov1D() encodes both kernel anisotropy and (relative) scale (the latter only exists for brute-force but not multi-res grid)
        float scaleZ = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, ray_dir_iso);
        float scaleX = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, coord.tangent);
        float scaleY = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, coord.bitangent);
        coeff_1D.gradient_scale = -2.f * gradient_scale_iso * Vec3f(sqr(scaleX), sqr(scaleY), sqr(scaleZ));

        if (_multiResolutionGrid) {
            coeff_1D.gradient_scale /= sqr(kernelScaleInfo.z()) + sqr(kernelScaleInfo.w());
        }

        if (!_correlationXY) {
            coeff_1D.gradient_scale.x() = 0;
            coeff_1D.gradient_scale.y() = 0;
        }

        // Sanity check for gradient
        Vec3f grad_diff = targetGrad - mean_grad - amplitude * evaluateGradientNoise1D(p, info.t, 0, rayDir, info, sampler, true);
        if (!_correlationXY) {
            if (fabs(grad_diff.dot(rayDir)) > 1e-2) {
                std::cerr << "Conditioning 1D grad error (match 1D): " << grad_diff.dot(rayDir) << std::endl;
            }
        }
        else {
            if (fabs(grad_diff.length()) > 1e-2) {
                std::cerr << "coeff_1D.gradient_scale: " << coeff_1D.gradient_scale << std::endl;
                std::cerr << "Conditioning 1D grad error (match 3D): " << grad_diff << std::endl;
            }
        }
    }
}

// Next-event estimation
// Support for 1D GPIS, non-global memory models, mirror/conductor as the micro-surface material

// Shared logic for NEE gradient computation and PDF evaluation
void SparseConvolutionNoiseRealization::neeShared(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info, NeeShared& ctx) {
    uint seed = computeSeed(info.pixelSampleSegment, info.sceneSeed);
    float amplitude = _gp->_cov->sparseConvNoiseAmplitude(p);

    // Compute the mean gradient
    Eigen::Vector2d mean_and_id = _gp->mean_weight_space(Vec3d(p), Derivative::None);
    ctx.mean_grad = Vec3f(mean_and_id(1) == _gp->_id ? _gp->_mean->dmean_da(Vec3d(p)) : _gp->_mean_additional->dmean_da(Vec3d(p)));

    ctx.ray_dir_iso = _gp->transformPosDirWorldtoLocal(p, rayDir, 1.0).normalized();
    ctx.coord = TangentFrame(Vec3f(ctx.ray_dir_iso));
    // Matrix to transform gradient from local ray space to local space
    // Since it's only a rotation matrix, inverse transpose = itself
    Eigen::Matrix3f mtx_ray_coord(3, 3);
    mtx_ray_coord.col(0) << ctx.coord.tangent.x(), ctx.coord.tangent.y(), ctx.coord.tangent.z();
    mtx_ray_coord.col(1) << ctx.coord.bitangent.x(), ctx.coord.bitangent.y(), ctx.coord.bitangent.z();
    mtx_ray_coord.col(2) << ctx.coord.normal.x(), ctx.coord.normal.y(), ctx.coord.normal.z();
    // The matrix to transforming gradients from isotropic ray space to world space
    ctx.mtx_pt = amplitude * _gp->localToWorldInvTransposeMatrix(p) * mtx_ray_coord;
    // The matrix to transforming gradients from world space to isotropic ray space
    Eigen::Matrix3f mtx_pt_inv = ctx.mtx_pt.inverse();

    // Compute the directional derivative (gz) along the ray in isotropic ray space
    UniformSampler sampler;
    if (!_multiResolutionGrid) {
        ctx.grad_constraint_3d = evaluateNoise1DNormalized(p, info.t, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), 1.0, true).yzw();
    }
    else {
        Vec4f multiResinfo = kernelScaleLevelRatio(p);
        Vec3f grad_iso_ray_base_low = evaluateNoise1DNormalized(p, info.t, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), multiResinfo.x(), true, true).yzw();
        Vec3f grad_iso_ray_base_high = evaluateNoise1DNormalized(p, info.t, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), multiResinfo.y(), true, false).yzw();
        ctx.grad_constraint_3d = grad_iso_ray_base_low * multiResinfo.z() / multiResinfo.x() + grad_iso_ray_base_high * multiResinfo.w() / multiResinfo.y();
    }

    // Compute the gradient plane in world space
    Vec3f plane_pt_world = to_vec3f(mult(ctx.mtx_pt, to_eigen3f(Vec3f(0.f, 0.f, ctx.grad_constraint_3d.z())))) + ctx.mean_grad; // A point on the plane
    Vec3f plane_x_axis_world = to_vec3f(mult(ctx.mtx_pt, to_eigen3f(Vec3f(1.f, 0.f, 0.f))));
    Vec3f plane_y_axis_world = to_vec3f(mult(ctx.mtx_pt, to_eigen3f(Vec3f(0.f, 1.f, 0.f))));
    Vec3f cross_product = plane_x_axis_world.cross(plane_y_axis_world);
    ctx.plane_normal_world = cross_product.normalized(); // The normal of the plane
    ctx.plane_stretch_jacobian = cross_product.length();
    // Compute the intersection distance along the normal direction and the plane
    ctx.isect_dist = plane_pt_world.dot(ctx.plane_normal_world) / normal.dot(ctx.plane_normal_world);

    // The sampled 3D gradient in the world space
    Vec3f grad_3d_world = ctx.isect_dist * normal;
    // The sampled 3D gradient in the isotropic ray space
    ctx.grad_3d_iso_ray = mult(mtx_pt_inv, grad_3d_world - ctx.mean_grad);
}

// Compute the 3D gradient from the normal direction
// Needed for correct conditioning at the start of the subsequent path
Vec3f SparseConvolutionNoiseRealization::neeGrad(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info) {
    NeeShared ctx;
    neeShared(rayDir, normal, p, info, ctx);
    return ctx.isect_dist * normal;
}

// Evaluate the PDF of sampling the 3D gradient from forward/GPIS sampling
float SparseConvolutionNoiseRealization::neePDF(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const float tSegment, const RayInfo info) {
    NeeShared ctx;
    neeShared(rayDir, normal, p, info, ctx);

    if (ctx.isect_dist < 0.0)
        return 0.0;

    // For non-stationary anisotropy
    float scaleX = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, ctx.coord.tangent);
    float scaleY = _gp->_cov->getNonstationaryCovSplatCov1D(p, p, ctx.coord.bitangent);

    float sample_x, sample_y;
    float pdf_area; // The PDF of sampling gx, gy in the isotropic ray space, under area measure
    UniformSampler sampler;
    if (!_multiResolutionGrid) {
        // Different conditioning leads to different gx, gy distribution at the current vertex, thus different PDF evaluation routine
        if (!(_ctxt == GPCorrelationContext::RenewalPlus && _correlationXY)) {
            sample_x = (ctx.grad_3d_iso_ray.x() - ctx.grad_constraint_3d.x()) * sqrt(2.0) * scaleX;
            sample_y = (ctx.grad_3d_iso_ray.y() - ctx.grad_constraint_3d.y()) * sqrt(2.0) * scaleY;
            pdf_area = exp(-sample_x * sample_x / 2.0) *
                       exp(-sample_y * sample_y / 2.0) / (2.0 * M_PI) * 2.0 * scaleX * scaleY;
        }
        else {
            // Note: _correlationXY only works for stationary GP; the logic here is for an isotropic convolution kernel
            if (!_gp->_cov->isStationaryKernel()) {
                std::cerr << "(NEE) 1D CorrelationXY does not work for brute-force non-stationary covariance kernel!\n";
            }
            sampler.set_state(MathUtil::xxhash32(info.pixelSampleSegment) + MathUtil::xxhash32(info.sceneSeed) + 1u);
            Eigen::VectorXd xy_1 = sample_standard_normal(2, sampler) / sqrt(2.f);
            float factor = std::exp(- tSegment * tSegment / 4.0) * (0.5 - tSegment * tSegment * 0.25);
            Eigen::VectorXd mu = factor * 2.0 * xy_1;
            float cov = 0.5 - factor * factor * 2;
            sample_x = (ctx.grad_3d_iso_ray.x() - ctx.grad_constraint_3d.x() - mu(0)) / sqrt(cov);
            sample_y = (ctx.grad_3d_iso_ray.y() - ctx.grad_constraint_3d.y() - mu(1)) / sqrt(cov);
            pdf_area = exp(-sample_x * sample_x / 2.0) *
                       exp(-sample_y * sample_y / 2.0) / (2.0 * M_PI) / cov;
        }
    }
    else {
        // The same logic as above, extended to multi-resolution noise
        Vec4f multiResinfo = kernelScaleLevelRatio(p);
        float nonStationaryScale = 1.0 / sqrt(sqr(multiResinfo.z() / multiResinfo.x()) + sqr(multiResinfo.w() / multiResinfo.y()));
        scaleX *= nonStationaryScale;
        scaleY *= nonStationaryScale;
        if (!(_ctxt == GPCorrelationContext::RenewalPlus && _correlationXY)) {
            sample_x = (ctx.grad_3d_iso_ray.x() - ctx.grad_constraint_3d.x()) * sqrt(2.0) * scaleX;
            sample_y = (ctx.grad_3d_iso_ray.y() - ctx.grad_constraint_3d.y()) * sqrt(2.0) * scaleY;
            pdf_area = exp(-sample_x * sample_x / 2.0) *
                       exp(-sample_y * sample_y / 2.0) / (2.0 * M_PI) * 2.0 * scaleX * scaleY;
        }
        else {
            // Note: the logic only works for covariance kernels with non-stationary scale but not non-stationary anisotropy
            if (_gp->_cov->isNonstationaryAnisotropicKernel()) {
                std::cerr << "(NEE) 1D CorrelationXY does not work for covariance kernel with non-stationary anisotropy!\n";
            }
            sampler.set_state(MathUtil::xxhash32(info.pixelSampleSegment) + MathUtil::xxhash32(info.sceneSeed) + 1u);
            Eigen::VectorXd xy_1 = sample_standard_normal(2, sampler) / sqrt(2.f);
            Eigen::VectorXd xy_2 = sample_standard_normal(2, sampler) / sqrt(2.f);
            float factor = std::exp(- tSegment * tSegment / 4.0) * (0.5 - tSegment * tSegment * 0.25);
            Eigen::VectorXd mu = factor * 2.0 * (xy_1 * multiResinfo.z() / multiResinfo.x() + xy_2 * multiResinfo.w() / multiResinfo.y());
            float cov = 0.5 - factor * factor * 2;
            sample_x = (ctx.grad_3d_iso_ray.x() - ctx.grad_constraint_3d.x() - mu(0)) / sqrt(cov) * nonStationaryScale;
            sample_y = (ctx.grad_3d_iso_ray.y() - ctx.grad_constraint_3d.y() - mu(1)) / sqrt(cov) * nonStationaryScale;
            pdf_area = exp(-sample_x * sample_x / 2.0) *
                       exp(-sample_y * sample_y / 2.0) / (2.0 * M_PI * cov) * sqr(nonStationaryScale);
        }
    }

    if (fabs(ctx.grad_3d_iso_ray.z() / ctx.grad_constraint_3d.z() - 1) > 1e-2 && fabs(ctx.grad_constraint_3d.z()) > 1e-2) {
        std::cerr << "NEE sanity check failed! " << ctx.grad_3d_iso_ray.z() << " " << ctx.grad_constraint_3d.z() << std::endl;
    }

    // Transform the PDF from area measure in isotropic ray space to solid angle measure in world space
    // Need to consider the Jacobian for the stretching of the gradient plane between spaces
    float cosTheta_light = std::abs(normal.dot(ctx.plane_normal_world));
    float pdf_normal = pdf_area * (ctx.isect_dist * ctx.isect_dist) / cosTheta_light / ctx.plane_stretch_jacobian;
    float pdf_omega = pdf_normal / (4.0 * normal.dot(-rayDir));
    return pdf_omega;
}

void SparseConvolutionNoiseRealization::cpneeShared(NeeShared& ctx){
    // The direction of the maximum correlation. 
    Vec3f maxCorrDir = _gp->_cov->maxCorrelationDirection().normalized();
    Vec3f gu_iso = maxCorrDir.cross(ctx.ray_dir_iso).normalized();
    Vec3f gv_iso = gu_iso.cross(ctx.ray_dir_iso);
    ctx.gu_iso_ray = ctx.coord.toLocal(gu_iso);
    ctx.gu_world = to_vec3f(mult(ctx.mtx_pt, to_eigen3f(ctx.gu_iso_ray)));
    // Project the full isotropic-ray gradient to gu_iso_ray to get the component of gvw_iso_ray. 
    Vec3f gvw_3d_iso_ray = ctx.grad_3d_iso_ray - ctx.grad_3d_iso_ray.dot(ctx.gu_iso_ray) * ctx.gu_iso_ray;
    Vec3f g_vw_world = to_vec3f(mult(ctx.mtx_pt, to_eigen3f(gvw_3d_iso_ray)));
    ctx.g_fix_world = g_vw_world + ctx.mean_grad;
    Vec3f gv_iso_ray = ctx.coord.toLocal(gv_iso);
    ctx.gv_world = to_vec3f(mult(ctx.mtx_pt, to_eigen3f(gv_iso_ray)));
}

DirectionCircle SparseConvolutionNoiseRealization::neeScatteringCircle(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info) {
    NeeShared ctx;
    neeShared(rayDir, normal, p, info, ctx);
    cpneeShared(ctx);
    ctx.gu_world.normalize();
    Vec3f gu_ortho = ctx.g_fix_world - ctx.g_fix_world.dot(ctx.gu_world) * ctx.gu_world;
    auto reflect = [](const Vec3f &d, const Vec3f &n) {
        return d - 2.0f * d.dot(n) * n;
    };
    Vec3f reflect_d_min = reflect(rayDir, gu_ortho.normalized());
    Vec3f reflect_d_max = rayDir;
    return DirectionCircle(0.5f * (reflect_d_min + reflect_d_max));
}

float SparseConvolutionNoiseRealization::cpneePDF(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, 
    const float t, const RayInfo info, const Primitive& light, const CircleSegment& segment){
    // The matrix to transforming gradients from isotropic ray space to world space
    NeeShared ctx;
    neeShared(rayDir, normal, p, info, ctx);

    if (ctx.isect_dist < 0.0)
        return 0.0;
    
    cpneeShared(ctx);

    float sample_x, sample_y;
    float pdf_area; // The PDF of sampling gx, gy in the isotropic ray space, under area measure
    float gu_stretch_jacobian = ctx.gu_world.length();
            float gv_stretch_jacobian = ctx.gv_world.length();
    UniformSampler sampler;
    if (!_multiResolutionGrid) {
        // Different conditioning leads to different gx, gy distribution at the current vertex, thus different PDF evaluation routine
        if (!(_ctxt == GPCorrelationContext::RenewalPlus && _correlationXY)) {
            sample_x = (ctx.grad_3d_iso_ray.x() - ctx.grad_constraint_3d.x());
            sample_y = (ctx.grad_3d_iso_ray.y() - ctx.grad_constraint_3d.y());
            float cos_theta = ctx.gu_iso_ray.x();
            float sin_theta = ctx.gu_iso_ray.y();
            float sample_v = -sin_theta * sample_x + cos_theta * sample_y;
            
            float std_gu_world = sqrt(0.5f) * gu_stretch_jacobian;
            IntersectionTemporary nullIsectData;
            IntersectionInfo isectInfo;
            isectInfo.w = rayDir; 
            isectInfo.Ng = isectInfo.Ns = normal;
            float pdf_gu = light.directPdf1D(0, nullIsectData, isectInfo, p, segment, ctx.gu_world.normalized(), ctx.g_fix_world, std_gu_world);
            float pdf_gv = exp(-sample_v * sample_v) / sqrt(M_PI) / gv_stretch_jacobian;
            pdf_area = pdf_gu * pdf_gv;
        }
        else {
            std::cerr << "SparseConvolutionNoiseRealization::cpneePDF()'s RenewalPlus model with correlationXY not yet implemented! \n"; 
            pdf_area = 0.0f;
        }
    }
    else {
        std::cerr << "SparseConvolutionNoiseRealization::cpneePDF()'s Non-stationary model not yet implemented! \n"; 
        pdf_area = 0.0f;
    }

    if (fabs(ctx.grad_3d_iso_ray.z() / ctx.grad_constraint_3d.z() - 1) > 1e-2 && fabs(ctx.grad_constraint_3d.z()) > 1e-2) {
        std::cerr << "1D NEE sanity check failed! " << ctx.grad_3d_iso_ray.z() << " " << ctx.grad_constraint_3d.z() << std::endl;
    }

    // Transform the PDF from area measure in isotropic ray space to solid angle measure in world space
    // Need to consider the Jacobian for the stretching of the gradient plane between spaces
    // ctx.gu_world and ctx.gv_world are not orthogonal in general (e.g. anisoXY case with lx != ly).
    // pdf_area is in (s_u, s_v) world-space bi-scalar measure; converting to gradient-plane area
    // requires dividing by |eu x ev| = ctx.plane_stretch_jacobian / (gu_jac * gv_jac).
    float cosTheta_light = std::abs(normal.dot(ctx.plane_normal_world));
    float pdf_normal = pdf_area * gu_stretch_jacobian * gv_stretch_jacobian / ctx.plane_stretch_jacobian * (ctx.isect_dist * ctx.isect_dist) / cosTheta_light;
    float pdf_omega = pdf_normal / (4.0 * normal.dot(-rayDir));
    return pdf_omega;
}

bool SparseConvolutionNoiseRealization::neeSampleDirect1D(SurfaceScatterEvent& event) {
    Vec3f normalLocal = (event.wi + event.wo) * 0.5;
    Vec3f normal = event.frame.toGlobal(Vec3f(normalLocal)).normalized();
    Vec3f rayDir = event.info->w;
    Vec3f p = event.info->p;
    RayInfo info = event.info->rayInfo;
    // The matrix to transforming gradients from isotropic ray space to world space
    NeeShared ctx;
    neeShared(rayDir, normal, p, info, ctx);

    if (ctx.isect_dist < 0.0)
        return false;

    cpneeShared(ctx);

    float sample_x = (ctx.grad_3d_iso_ray.x() - ctx.grad_constraint_3d.x());
    float sample_y = (ctx.grad_3d_iso_ray.y() - ctx.grad_constraint_3d.y());
    float cos_theta = ctx.gu_iso_ray.x();
    float sin_theta = ctx.gu_iso_ray.y();
    float sample_v = -sin_theta * sample_x + cos_theta * sample_y;
    float gu_stretch_jacobian = ctx.gu_world.length();
    float gv_stretch_jacobian = ctx.gv_world.length();
    float std_gu_world = sqrt(0.5f) * gu_stretch_jacobian;
    LightSample lightSample;
    if(!event.light->sampleDirect1D(0, p, *event.sampler, lightSample, *event.info, event.segment, ctx.gu_world.normalized(), ctx.g_fix_world, std_gu_world))
        return false;
    if (lightSample.pdf <= 1.0e-8f){
        // std::cerr << "Warning: light sample pdf is 0 in SparseConvolutionNoiseRealization::neeSampleDirect1D()! \n";
        return false;
    }
    event.wo = lightSample.d;
    float pdf_gu = lightSample.pdf;
    float pdf_gv = exp(-sample_v * sample_v) / sqrt(M_PI) / gv_stretch_jacobian;
    float pdf_area = pdf_gu * pdf_gv;
    // Transform the PDF from area measure in isotropic ray space to solid angle measure in world space
    // Need to consider the Jacobian for the stretching of the gradient plane between spaces
    // Also need to re-calculate the normal and the ctx.isect_dist again since we are sampling a new gradient. 
    normal = (0.5f * (-rayDir + lightSample.d)).normalized();
    neeShared(rayDir, normal, p, info, ctx);
    float cosTheta_light = std::abs(normal.dot(ctx.plane_normal_world));
    // ctx.gu_world and ctx.gv_world are not orthogonal in general (e.g. anisoXY case with lx != ly).
    // pdf_area is in (s_u, s_v) world-space bi-scalar measure; converting to gradient-plane area
    // requires dividing by |eu x ev| = ctx.plane_stretch_jacobian / (gu_jac * gv_jac).
    float pdf_normal = pdf_area * gu_stretch_jacobian * gv_stretch_jacobian / ctx.plane_stretch_jacobian * (ctx.isect_dist * ctx.isect_dist) / cosTheta_light;
    float pdf_omega = pdf_normal / (4.0 * normal.dot(-rayDir));
    event.pdf = pdf_omega;
    return true;
}
}