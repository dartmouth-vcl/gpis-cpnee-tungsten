#ifndef SPARSECONVOLUTIONNOISE_HPP_
#define SPARSECONVOLUTIONNOISE_HPP_

#include <math/GaussianProcess.hpp>

namespace Tungsten {
    struct SparseConvConditioningCoefficients3D
    {
        // 3D Renewal+ conditioning, pathwise update
        float value_scale = 0.;
        Vec3f gradient_scale = Vec3f(0.);
        Vec3f ray_origin  = Vec3f(0.);
    };

    struct SparseConvConditioningCoefficients1D
    {
        // 1D Renewal+ conditioning, pathwise update
        float value_scale = 0.;
        Vec3f gradient_scale = Vec3f(0.);
        Vec3f ray_origin  = Vec3f(0.);
    };

    // Per-vertex data shared by the NEE routines, so that each quantity is computed once.
    // neeShared() fills every field except the 1-DoF block; cpneeShared() fills that block.
    struct NeeShared
    {
        // The ray frame in the isotropic ray space
        Vec3f ray_dir_iso = Vec3f(0.f);
        TangentFrame coord;
        // The mean gradient at p, in world space
        Vec3f mean_grad = Vec3f(0.f);

        // The matrix transforming gradients from the isotropic ray space to world space
        Eigen::Matrix3f mtx_pt = Eigen::Matrix3f::Identity();
        Vec3f grad_constraint_3d = Vec3f(0.f);
        Vec3f grad_3d_iso_ray = Vec3f(0.f);
        Vec3f plane_normal_world = Vec3f(0.f);
        float plane_stretch_jacobian = 0.f;
        float isect_dist = 0.f;

        // 1-DoF block
        Vec3f g_fix_world = Vec3f(0.f);
        Vec3f gu_world = Vec3f(0.f);
        Vec3f gv_world = Vec3f(0.f);
        // gu in the ray frame; its x()/y() are the cos/sin of the angle between gu and coord.tangent
        Vec3f gu_iso_ray = Vec3f(0.f);
    };

    struct SparseConvolutionNoiseRealization {
        std::shared_ptr<GaussianProcess> _gp;
        GPCorrelationContext _ctxt;

        uint _globalSeed;
        bool _useSingleRealization;
        float _impulseDensity;
        bool _isotropicSpace3DSampling;
        bool _isotropicRaySpace3DSampling;
        bool _1DSampling;
        bool _activateConditioning;
        bool _correlationXY;
        SparseConv1DSamplingScheme _1DsamplingScheme;
        Vec3f _1DsamplingSchemeSampleCountScale;
        bool _multiResolutionGrid;

        float _base;
        SparseConvConditioningCoefficients3D coeff_3D;
        SparseConvConditioningCoefficients1D coeff_1D;

        // Tricky way to assign different phase functions, only used for the coffee mug scene
        bool _surfVolPhaseSeparate;
        float _surfVolPhaseAmpThresh;

        SparseConvolutionNoiseRealization() {}

        SparseConvolutionNoiseRealization(std::shared_ptr<GaussianProcess> gp, GPCorrelationContext ctxt,
                                          uint globalSeed, float impulseDensity, bool useSingleRealization,
                                          bool isotropicSpace3DSampling, bool oneDSampling,
                                          SparseConv1DSamplingScheme oneDSamplingScheme,
                                          Vec3f oneDSamplingSchemeSampleCountScale,
                                          bool correlationXY,
                                          bool surfVolPhaseSeparate, float surfVolPhaseAmpThresh);

        inline uint computeSeed(const Vec4u pixelSampleSegment, const uint sceneSeed) const;

        std::pair<SparseConv1DSamplingScheme, Vec3f> samplingScheme(const Vec3f& p) const;

        float evaluateValue(const Vec3f& p, const float t, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler, int& GPId);
        Vec3f evaluateGradient(const Vec3f& p, const float t, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler);

        float evaluateValueNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) { return evaluateNoise3D(p, rayDir, seed, sampler, conditioning).x(); }
        Vec3f evaluateGradientNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) { return evaluateNoise3D(p, rayDir, seed, sampler, conditioning).yzw(); }
        Vec4f evaluateNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning);

        float evaluateValueNoise1D(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning);
        Vec3f evaluateGradientNoise1D(const Vec3f& p, const float tTotal, const float tSegment, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler, bool conditioning);

        Vec4f evaluateNoise3DNormalized(const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        inline Vec4f evaluateNoise3DIsotropicNormalizedSelect(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        Vec4f evaluateNoise3DIsotropicNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        Vec4f evaluateNoise3DIsotropicRayNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        Vec4f evaluateNoise1DNormalized(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning, bool multiResLowLevel = false);

        Vec4f noise3D(const Vec3f& p_world, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const;
        Vec4f cell3D(const Vec3f& p_world, const Vec3u& ijk, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const;
        Vec2f noise1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const;
        Vec2f cell1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const uint i, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const;

        Vec4f kernelScaleLevelRatio(const Vec3f& p) const;

        void conditioning(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler);
        void conditioning3D(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler);
        void conditioning1D(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler);

        void neeShared(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info, NeeShared& ctx);
        void cpneeShared(NeeShared& ctx);
        Vec3f neeGrad(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info);
        float neePDF(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const float t, const RayInfo info);
        
        float cpneePDF(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const float t, const RayInfo info, const Primitive& light, const CircleSegment& segment);
        DirectionCircle neeScatteringCircle(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info);
        bool neeSampleDirect1D(SurfaceScatterEvent& event);
    };
}

#endif /* SPARSECONVOLUTIONNOISE_HPP_ */
