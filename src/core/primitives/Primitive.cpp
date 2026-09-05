#include "Primitive.hpp"

#include "bsdfs/LambertBsdf.hpp"
#include "bsdfs/Bsdf.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

#include "sampling/Gaussian.hpp"

namespace Tungsten {

std::shared_ptr<Bsdf> Primitive::_defaultBsdf = std::make_shared<LambertBsdf>();

Primitive::Primitive() : _emScale(1.f)
{
}

Primitive::Primitive(const std::string &name)
: JsonSerializable(name), _emScale(1.f)
{
}

void Primitive::fromJson(JsonPtr value, const Scene &scene)
{
    JsonSerializable::fromJson(value, scene);
    value.getField("transform", _transform);

    value.getField("scale", _emScale);

    if (auto emission = value["emission"]) _emission = scene.fetchTexture(emission, TexelConversion::REQUEST_RGB);
    if (auto power    = value["power"   ]) _power    = scene.fetchTexture(power,    TexelConversion::REQUEST_RGB);

    if (auto intMedium  = value["int_medium"]) _intMedium = scene.fetchMedium(intMedium);
    if (auto extMedium  = value["ext_medium"]) _extMedium = scene.fetchMedium(extMedium);
}

rapidjson::Value Primitive::toJson(Allocator &allocator) const
{
    JsonObject result{JsonSerializable::toJson(allocator), allocator,
        "transform", _transform
    };
    if (_power)
        result.add("power", *_power);
    else if (_emission)
        result.add("emission", *_emission);
    if (_intMedium)
        result.add("int_medium", *_intMedium);
    if (_extMedium)
        result.add("ext_medium", *_extMedium);

    result.add("scale", _emScale);

    return result;
}

bool Primitive::samplePosition(PathSampleGenerator &/*sampler*/, PositionSample &/*sample*/) const
{
    return false;
}

bool Primitive::sampleDirection(PathSampleGenerator &/*sampler*/, const PositionSample &/*point*/, DirectionSample &/*sample*/) const
{
    return false;
}

bool Primitive::sampleDirect(uint32 /*threadIndex*/, const Vec3f &/*p*/, PathSampleGenerator &/*sampler*/, LightSample &/*sample*/) const
{
    return false;
}

bool Primitive::sampleDirect1D(uint32 /*threadIndex*/, const Vec3f &p, PathSampleGenerator &sampler, 
        LightSample &sample, const IntersectionInfo& info,
        const CircleSegment& segment, const Vec3f& gu, const Vec3f& g_fix, 
        const float std_gu, const SegmentSamplingMethod samplingMethod) const
{
    Vec3f ray_d = info.w;
    BSphere3f sbounds = this->sbounds();
    if(sbounds.empty())
        return false;
    if(!this->isEmissive())
        FAIL("Trying to sample direct lighting on a non-emissive primitive!");
    auto reflect = [](const Vec3f& d, const Vec3f& n) {
        return d - 2.0f * d.dot(n) * n;
        };
    if(abs(segment.angle - 2.0*PI) >= 1.0e-5f){
        Vec3f k = segment.axis.normalized();
        Vec3f v = segment.op;
        float angle = segment.angle;
        Vec3f op1 = v * cosf(angle * 0.5f) + k.cross(v) * sinf(angle * 0.5f) + k * k.dot(v) * (1.0f - cosf(angle * 0.5f));
        Vec3f op2 = v * cosf(-angle * 0.5f) + k.cross(v) * sinf(-angle * 0.5f) + k * k.dot(v) * (1.0f - cosf(-angle * 0.5f));
        Vec3f d1 = (segment.axis + op1).normalized();
        Vec3f d2 = (segment.axis + op2).normalized();
        Vec3f n1 = (d1 - ray_d).normalized();
        Vec3f n2 = (d2 - ray_d).normalized();
        Vec3f gu_ortho = g_fix - g_fix.dot(gu) * gu;
        float alpha_1 = n1.dot(gu);
        Vec3f w_1 = n1 - alpha_1 * gu;
        float t1 = gu_ortho.dot(w_1) / w_1.dot(w_1);
        float alpha_2 = n2.dot(gu);
        Vec3f w_2 = n2 - alpha_2 * gu;
        float t2 = gu_ortho.dot(w_2) / w_2.dot(w_2);
        Vec3f g1 = n1 * t1;
        Vec3f g2 = n2 * t2;
        float gu1 = (g1 - g_fix).dot(gu);
        float gu2 = (g2 - g_fix).dot(gu);
        if (gu1 > gu2) { std::swap(gu1, gu2); }
        float gu_value = segment.InfIncluded ? 
        truncated_normal_ab_tails_sample(0.0f, std_gu, gu1, gu2, sampler.next1D()):
        truncated_normal_ab_sample(0.0f, std_gu, gu1, gu2, sampler.next1D());
        float pdf_gu_truncated_gaussian = segment.InfIncluded ? 
        truncated_normal_ab_tails_pdf(gu_value, 0.0f, std_gu, gu1, gu2):
        truncated_normal_ab_pdf(gu_value, 0.0f, std_gu, gu1, gu2);
        // Uniform sampling for verification: 
        // float gu_value = sampler.next1D() * (gu2 - gu1) + gu1;
        // float pdf_gu_truncated_gaussian = 1.0f / (gu2 - gu1);
        Vec3f grad = g_fix + gu_value * gu;
        Vec3f normal = grad.normalized();
        sample.d = reflect(ray_d, normal);
        sample.dist = 0.0f;
        sample.pdf = pdf_gu_truncated_gaussian;
        return true;
    }
    else{
        Vec3f n = info.Ng.normalized();
        float alpha_3 = n.dot(gu);
        Vec3f w = n - alpha_3 * gu;
        Vec3f gu_ortho = g_fix - g_fix.dot(gu) * gu;
        float t = gu_ortho.dot(w) / w.dot(w);
        Vec3f g = n * t; 
        float gu_value = (g - g_fix).dot(gu);
        Vec3f grad = g_fix + gu_value * gu;
        Vec3f normal = grad.normalized();
        sample.d = reflect(ray_d, normal);
        sample.dist = 0.0f;
        sample.pdf = normal_ms_pdf(gu_value, 0.0f, std_gu);
        return true;
    }
}

DirectionCircle Primitive::getLightCircle(const Vec3f & p) const
{
    BSphere3f sbounds = this->sbounds();
    Vec3f p2center = sbounds.center() - p;
    float r2 = sbounds.radius()*sbounds.radius();
    float d2 = p2center.lengthSq();
    float length = sqrt((d2 - r2) / d2);
    return DirectionCircle(length * p2center.normalized());
}

BSphere3f Primitive::sbounds() const
{
    FAIL("Spherical bounds not implemented for this primitive!");
    return BSphere3f();
}

bool Primitive::invertPosition(WritablePathSampleGenerator &/*sampler*/, const PositionSample &/*point*/) const
{
    FAIL("Primitive::invertPosition not implemented!");
}
bool Primitive::invertDirection(WritablePathSampleGenerator &/*sampler*/, const PositionSample &/*point*/,
        const DirectionSample &/*direction*/) const
{
    FAIL("Primitive::invertDirection not implemented!");
}

float Primitive::positionalPdf(const PositionSample &/*point*/) const
{
    return 0.0f;
}

float Primitive::directionalPdf(const PositionSample &/*point*/, const DirectionSample &/*sample*/) const
{
    return 0.0f;
}

float Primitive::directPdf(uint32 /*threadIndex*/, const IntersectionTemporary &/*data*/,
        const IntersectionInfo &/*info*/, const Vec3f &/*p*/) const
{
    return 0.0f;
}

float Primitive::directPdf1D(uint32 /*threadIndex*/, const IntersectionTemporary &/*data*/,
        const IntersectionInfo & info, const Vec3f &/*p*/, const CircleSegment& segment, 
        const Vec3f& gu, const Vec3f& g_fix, const float std_gu, const SegmentSamplingMethod samplingMethod) const
{
    if(abs(segment.angle-2.0*PI) >= 1.0e-5f){
        // Get the range of gu.
        Vec3f ray_d = info.w;
        Vec3f k = segment.axis.normalized();
        Vec3f v = segment.op;
        Vec3f op1 = v * cosf(segment.angle / 2.0f) + k.cross(v) * sinf(segment.angle / 2.0f) + k * k.dot(v) * (1.0f - cosf(segment.angle / 2.0f));
        Vec3f op2 = v * cosf(-segment.angle / 2.0f) + k.cross(v) * sinf(-segment.angle / 2.0f) + k * k.dot(v) * (1.0f - cosf(-segment.angle / 2.0f));
        Vec3f d1 = (segment.axis + op1).normalized();
        Vec3f d2 = (segment.axis + op2).normalized();
        Vec3f n1 = (d1 - ray_d).normalized();
        Vec3f n2 = (d2 - ray_d).normalized();
        Vec3f n = info.Ng.normalized();
        Vec3f gu_ortho = g_fix - g_fix.dot(gu) * gu;
        float alpha_1 = n1.dot(gu);
        Vec3f w_1 = n1 - alpha_1 * gu;
        float t1 = gu_ortho.dot(w_1) / w_1.dot(w_1);
        float alpha_2 = n2.dot(gu);
        Vec3f w_2 = n2 - alpha_2 * gu;
        float t2 = gu_ortho.dot(w_2) / w_2.dot(w_2);
        float alpha_3 = n.dot(gu);
        Vec3f w = n - alpha_3 * gu;
        float t = gu_ortho.dot(w) / w.dot(w);
        Vec3f g1 = n1 * t1;
        Vec3f g2 = n2 * t2;
        Vec3f g = n * t;
        float gu1 = (g1 - g_fix).dot(gu);
        float gu2 = (g2 - g_fix).dot(gu);
        float gu_value = (g - g_fix).dot(gu);
        if (gu1 > gu2)
        {
            std::swap(gu1, gu2);
        }
        float pdf_gu_truncated_gaussian = segment.InfIncluded ? 
        truncated_normal_ab_tails_pdf(gu_value, 0.0f, std_gu, gu1, gu2) : 
        truncated_normal_ab_pdf(gu_value, 0.0f, std_gu, gu1, gu2);
        return pdf_gu_truncated_gaussian;
        // Uniform sampling for verification:
        // float pdf_gu_truncated_gaussian = 1.0f / (gu2 - gu1);
    }
    else{
        Vec3f n = info.Ng.normalized();
        float alpha_3 = n.dot(gu);
        Vec3f w = n - alpha_3 * gu;
        Vec3f gu_ortho = g_fix - g_fix.dot(gu) * gu;
        float t = gu_ortho.dot(w) / w.dot(w);
        Vec3f g = n * t;
        float gu_value = (g - g_fix).dot(gu);
        return normal_ms_pdf(gu_value, 0.0f, std_gu);
    }
}

Vec3f Primitive::evalPositionalEmission(const PositionSample &/*sample*/) const
{
    return Vec3f(0.0f);
}

Vec3f Primitive::evalDirectionalEmission(const PositionSample &/*point*/, const DirectionSample &/*sample*/) const
{
    return Vec3f(0.0f);
}

Vec3f Primitive::evalDirect(const IntersectionTemporary &data, const IntersectionInfo &info) const
{
    if (!_emission)
        return Vec3f(0.0f);
    if (hitBackside(data))
        return Vec3f(0.0f);
    return (*_emission)[info];
}

void Primitive::prepareForRender()
{
    if (_power) {
        _emission = std::shared_ptr<Texture>(_power->clone());
        _emission->scaleValues(powerToRadianceFactor());
    }
}

void Primitive::teardownAfterRender()
{
    if (_power)
        _emission.reset();
}

void Primitive::setupTangentFrame(const IntersectionTemporary &data,
        const IntersectionInfo &info, TangentFrame &dst) const
{
    const Texture *bump = info.bsdf ? info.bsdf->bump().get() : nullptr;

    if ((!bump || bump->isConstant()) && !info.bsdf->lobes().isAnisotropic()) {
        dst = TangentFrame(info.Ns);
        return;
    }
    Vec3f T, B, N(info.Ns);
    if (!tangentSpace(data, info, T, B)) {
        dst = TangentFrame(info.Ns);
        return;
    }
    if (bump && !bump->isConstant()) {
        Vec2f dudv;
        bump->derivatives(info.uv, dudv);

        T += info.Ns*(dudv.x() - info.Ns.dot(T));
        B += info.Ns*(dudv.y() - info.Ns.dot(B));
        N = T.cross(B);
        if (N == 0.0f) {
            dst = TangentFrame(info.Ns);
            return;
        }
        if (N.dot(info.Ns) < 0.0f)
            N = -N;
        N.normalize();
    }
    T = (T - N.dot(T)*N);
    if (T == 0.0f) {
        dst = TangentFrame(info.Ns);
        return;
    }
    T.normalize();
    B = N.cross(T);

    dst = TangentFrame(N, T, B);
}

}
