#include "ConductorBsdf.hpp"
#include "ComplexIor.hpp"
#include "Fresnel.hpp"

#include "samplerecords/SurfaceScatterEvent.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "math/MathUtil.hpp"
#include "math/Angle.hpp"
#include "math/Vec.hpp"

#include "io/JsonObject.hpp"

#include <tinyformat/tinyformat.hpp>
#include <rapidjson/document.h>

#include "media/GaussianProcessMedium.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"

namespace Tungsten {

ConductorBsdf::ConductorBsdf()
: _materialName("Cu"),
  _eta(0.200438f, 0.924033f, 1.10221f),
  _k(3.91295f, 2.45285f, 2.14219f)
{
    _lobes = BsdfLobes(BsdfLobes::SpecularReflectionLobe);
}

bool ConductorBsdf::lookupMaterial()
{
    return ComplexIorList::lookup(_materialName, _eta, _k);
}

void ConductorBsdf::fromJson(JsonPtr value, const Scene &scene)
{
    Bsdf::fromJson(value, scene);
    if (value.getField("eta", _eta) && value.getField("k", _k))
        _materialName.clear();
    if (value.getField("material", _materialName) && !lookupMaterial())
        value.parseError(tfm::format("Unable to find material with name '%s'", _materialName));
}

rapidjson::Value ConductorBsdf::toJson(Allocator &allocator) const
{
    JsonObject result{Bsdf::toJson(allocator), allocator,
        "type", "conductor"
    };
    if (_materialName.empty())
        result.add("eta", _eta, "k", _k);
    else
        result.add("material", _materialName);

    return result;
}

bool ConductorBsdf::sample(SurfaceScatterEvent &event) const
{
    if (!event.requestedLobe.test(BsdfLobes::SpecularReflectionLobe))
        return false;

    if (event.is1DNEE)
    {
        GPContextSparseConvNoise &ctxt = *(GPContextSparseConvNoise *)event.ctxt;
        if (!ctxt.noise.neeSampleDirect1D(event))
            return false;
        event.sampledLobe = BsdfLobes::SpecularReflectionLobe;
        event.weight = albedo(event.info) * Fresnel::conductorReflectance(_eta, _k, event.wi.z());
        return true;
    }
    else
    {
        event.wo = Vec3f(-event.wi.x(), -event.wi.y(), event.wi.z());
        GPContextSparseConvNoise &ctxt = *(GPContextSparseConvNoise *)event.ctxt;
        std::shared_ptr<GaussianProcess> gp = ctxt.noise._gp;
        event.weight = albedo(event.info) * Fresnel::conductorReflectance(_eta, _k, event.wi.z());
        event.pdf = 1.0f;
        if (event.sparseConv1DSamplingSchemeSampleCountScale.x() != 1.0f)
        {
            if (!event.ctxt)
            {
                std::cerr << "GPContext does not exist in ConductorBsdf::sample()!\n";
            }

            event.pdf = ctxt.noise.neePDF(event.info->w, event.info->Ng, event.info->p, event.info->t, event.info->rayInfo);
        }
        return true;
    }
}

Vec3f ConductorBsdf::eval(const SurfaceScatterEvent &event) const
{
    bool evalR = event.requestedLobe.test(BsdfLobes::SpecularReflectionLobe);
    if (!evalR)
        return Vec3f(0.0f);

    if (event.sparseConv1DSamplingSchemeSampleCountScale.x() == 1.0f) {
        if (checkReflectionConstraint(event.wi, event.wo))
            return albedo(event.info)*Fresnel::conductorReflectance(_eta, _k, event.wi.z());
        else
            return Vec3f(0.0f);
    }
    else {
        if (!event.ctxt) {
            std::cerr << "GPContext does not exist in ConductorBsdf::sample()!\n";
        }
        Vec3f normalLocal = (event.wi + event.wo) * 0.5;
        Vec3f normalWorld = event.frame.toGlobal(Vec3f(normalLocal)).normalized();
        GPContextSparseConvNoise& ctxt = *(GPContextSparseConvNoise*)event.ctxt;
        double pdf = ctxt.noise.neePDF(event.info->w, normalWorld, event.info->p, event.info->t, event.info->rayInfo);
        return albedo(event.info) * Fresnel::conductorReflectance(_eta, _k, event.wi.z()) * pdf;
    }
}

Vec3f ConductorBsdf::evalGrad(const SurfaceScatterEvent &event) const {
    assert(event.sparseConv1DSamplingScheme != SparseConv1DSamplingScheme::UNI);
    Vec3f normalLocal = (event.wi + event.wo) * 0.5;
    Vec3f normalWorld = event.frame.toGlobal(Vec3f(normalLocal)).normalized();
    GPContextSparseConvNoise& ctxt = *(GPContextSparseConvNoise*)event.ctxt;
    return ctxt.noise.neeGrad(event.info->w, normalWorld, event.info->p, event.info->rayInfo);
}

bool ConductorBsdf::invert(WritablePathSampleGenerator &/*sampler*/, const SurfaceScatterEvent &event) const
{
    bool evalR = event.requestedLobe.test(BsdfLobes::SpecularReflectionLobe);
    if (evalR && checkReflectionConstraint(event.wi, event.wo))
        return true;
    else
        return false;
}

float ConductorBsdf::pdf(const SurfaceScatterEvent &event) const
{
    bool sampleR = event.requestedLobe.test(BsdfLobes::SpecularReflectionLobe);
    if (!sampleR)
        return 0.0f;
    if (event.sparseConv1DSamplingSchemeSampleCountScale.x() == 1.0f) {
        if (checkReflectionConstraint(event.wi, event.wo))
            return 1.0f;
        else
            return 0.0f;
    }
    else {
        if (!event.ctxt) {
            std::cerr << "GPContext does not exist in ConductorBsdf::sample()!\n";
        }
        Vec3f normalLocal = (event.wi + event.wo) * 0.5;
        Vec3f normalWorld = event.frame.toGlobal(Vec3f(normalLocal)).normalized();
        GPContextSparseConvNoise &ctxt = *(GPContextSparseConvNoise *)event.ctxt;
        if (event.is1DNEE)
        {
            return ctxt.noise.cpneePDF(event.info->w, normalWorld, event.info->p, event.info->t, event.info->rayInfo, *event.light, event.segment);
        }
        else
        {
            return (float)ctxt.noise.neePDF(event.info->w, normalWorld, event.info->p, event.info->t, event.info->rayInfo);
        }
    }
}

DirectionCircle ConductorBsdf::getScatteringCircle(const SurfaceScatterEvent &event) const
{
    Vec3f normalLocal = (event.wi + event.wo) * 0.5;
    Vec3f normalWorld = event.frame.toGlobal(Vec3f(normalLocal)).normalized();
    GPContextSparseConvNoise &ctxt = *(GPContextSparseConvNoise *)event.ctxt;
    DirectionCircle result = ctxt.noise.neeScatteringCircle(event.info->w, normalWorld, event.info->p, event.info->rayInfo);
    return result;
}
}
