#ifndef PHASEFUNCTION_HPP_
#define PHASEFUNCTION_HPP_

#include "samplerecords/PhaseSample.hpp"

#include "sampling/WritablePathSampleGenerator.hpp"

#include "io/JsonSerializable.hpp"

#include "primitives/DirectionCone.hpp"

namespace Tungsten {

class PathSampleGenerator;
class Scene;
class MediumSample;

class PhaseFunction : public JsonSerializable
{
public:
    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual Vec3f eval(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const = 0;
    virtual Vec3f evalGrad(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const {
        std::cout << "PhaseFunction::evalGrad() not implemented!\n";
        return Vec3f(0.f);
    };
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const = 0;
    virtual bool sampleDirect1D(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const{
        std::cerr << "PhaseFunction::sampleDirect1D() not implemented! \n";
        return false;
    }
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const = 0;
    virtual float pdf_cpnee(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const {
        std::cerr << "PhaseFunction::pdf_cpnee() not implemented!\n";
        return 0.0f;
    }
    virtual DirectionCircle getScatteringCircle(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const {
        std::cerr << "PhaseFunction::getScatteringCircle() not implemented!\n";
        return DirectionCircle();
    }

    virtual bool isSpecular() const { return false; }
};

}

#endif /* PHASEFUNCTION_HPP_ */
