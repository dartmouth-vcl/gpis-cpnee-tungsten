#include "BRDFPhaseFunction.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

void BRDFPhaseFunction::fromJson(JsonPtr value, const Scene& scene) {
    PhaseFunction::fromJson(value, scene);

    if (auto bsdf = value["bsdf"])
        _bsdf = scene.fetchBsdf(bsdf);
}


rapidjson::Value BRDFPhaseFunction::toJson(Allocator &allocator) const
{
    return JsonObject{PhaseFunction::toJson(allocator), allocator,
        "type", "brdf",
        "bsdf", * _bsdf,
    };
}

void BRDFPhaseFunction::setEventIsectInfo(const Vec3f &wi, const MediumSample &mediumSample, SurfaceScatterEvent& se, IntersectionInfo& info, 
    bool is1DNEE, const Primitive* plight, const CircleSegment& segment) const {
    auto normal = vec_conv<Vec3f>(mediumSample.aniso.normalized());
    se.requestedLobe = BsdfLobes::AllLobes;
    se.frame = TangentFrame(normal);
    se.wi = se.frame.toLocal(-wi).normalized();
    se.sparseConv1DSamplingScheme = mediumSample.sparseConv1DSamplingScheme;
    se.sparseConv1DSamplingSchemeSampleCountScale = mediumSample.sparseConv1DSamplingSchemeSampleCountScale;
    se.ctxt = mediumSample.ctxt;
    info.bsdf = _bsdf.get();
    info.Ng = normal;
    info.Ns = normal;
    info.p = mediumSample.p;
    info.primitive = nullptr;
    info.uv = Vec2f(0.f);
    info.w = wi;
    info.t = mediumSample.t;
    info.rayInfo = mediumSample.rayInfo;
    se.info = &info;
    se.is1DNEE = is1DNEE;
    se.light = plight;
    se.segment = segment;
}

bool BRDFPhaseFunction::isSpecular() const {
    return _bsdf->lobes().isPureSpecular();
}

Vec3f BRDFPhaseFunction::eval(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const
{
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();
    return _bsdf->eval(se);
}

Vec3f BRDFPhaseFunction::evalGrad(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const
{
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();
    return _bsdf->evalGrad(se);
}

bool BRDFPhaseFunction::sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const
{
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.sampler = &sampler;

    if (!_bsdf->sample(se)) return false;
    sample.w = se.frame.toGlobal(se.wo).normalized();
    sample.weight = se.weight;
    sample.pdf = se.pdf;
    return true;
}

bool BRDFPhaseFunction::sampleDirect1D(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const{
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info, true, mediumSample.light, *mediumSample.segment);
    se.sampler = &sampler;
    se.wo = se.frame.toLocal(sample.w).normalized();
    if (!_bsdf->sample(se)) return false;
    // sample.w = se.frame.toGlobal(se.wo).normalized(); // The sampled direction is in world space. We do not need this conversion. 
    sample.w = se.wo;
    sample.weight = se.weight;
    sample.pdf = se.pdf;
    return true;
}

bool BRDFPhaseFunction::invert(WritablePathSampleGenerator &sampler, const Vec3f &/*wi*/, const Vec3f &wo, const MediumSample& mediumSample) const
{
    return false;
}

float BRDFPhaseFunction::pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const
{
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();

    return _bsdf->pdf(se);
}

float BRDFPhaseFunction::pdf_cpnee(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const{
    if(!mediumSample.is1DNEE || mediumSample.light == nullptr){
        std::cerr << "Incorrect mediumSample settings for BRDFPhaseFunction::pdf_cpnee()!\n";
        return 0.0f;
    }
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info, true, mediumSample.light, *mediumSample.segment);
    se.wo = se.frame.toLocal(wo).normalized();

    return _bsdf->pdf(se);
}

DirectionCircle BRDFPhaseFunction::getScatteringCircle(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const {
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();

    return _bsdf->getScatteringCircle(se);
}
}
