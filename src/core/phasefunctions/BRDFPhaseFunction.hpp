#ifndef BRDFPHASEFUNCTION_HPP_
#define BRDFPHASEFUNCTION_HPP_

#include "PhaseFunction.hpp"
#include "core/bsdfs/Bsdf.hpp"
#include "primitives/DirectionCone.hpp"

namespace Tungsten {

class Primitive;

class BRDFPhaseFunction : public PhaseFunction
{
private:
    std::shared_ptr<Bsdf> _bsdf;

public:
    void setEventIsectInfo(const Vec3f &wi, const MediumSample &mediumSample, SurfaceScatterEvent& se, IntersectionInfo& info, 
        bool is1DNEE = false, const Primitive* plight = nullptr, const CircleSegment& segment = CircleSegment()) const;

    virtual rapidjson::Value toJson(Allocator& allocator) const override;
    virtual void fromJson(JsonPtr value, const Scene& scene) override;

    virtual Vec3f eval(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
    virtual Vec3f evalGrad(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const override;
    virtual bool sampleDirect1D(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const override;
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
    virtual float pdf_cpnee(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
    DirectionCircle getScatteringCircle(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const override;

    virtual bool isSpecular() const override;
};

}

#endif /* BRDFPHASEFUNCTION_HPP_ */
