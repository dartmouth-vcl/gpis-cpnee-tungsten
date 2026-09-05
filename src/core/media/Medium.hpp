#ifndef MEDIUM_HPP_
#define MEDIUM_HPP_

#include "transmittances/Transmittance.hpp"

#include "phasefunctions/PhaseFunction.hpp"

#include "samplerecords/MediumSample.hpp"

#include "sampling/WritablePathSampleGenerator.hpp"

#include "math/Ray.hpp"

#include "io/JsonSerializable.hpp"

#include <memory>

#include <nlohmann/json.hpp>

namespace Tungsten {

struct DumpData {
    std::vector<float> ts; // distance along the ray
    std::vector<float> vals; // value of the SDF (mean + gp)

    DumpData() {}

    void add_data(float t, float val) {
        ts.push_back(t);
        vals.push_back(val);
    }
};

void to_json(nlohmann::json& j, const DumpData& data);

void from_json(const nlohmann::json& j, DumpData& data);

class Scene;

enum class SparseConv1DSamplingScheme {
    UNI,
    NEE,
    MIS,
    CPNEE,
    MIS_3WAY
};

struct GPContext {
    virtual void reset() = 0;
};

class Medium : public JsonSerializable
{
protected:
    std::shared_ptr<Transmittance> _transmittance;
    std::shared_ptr<PhaseFunction> _phaseFunction;
    int _maxBounce;

public:

    struct MediumState
    {
        bool firstScatter;
        int component;
        int bounce;
        int lastGPId;
        Vec3d lastAniso;
        float lastVal;
        RayInfo info;
        std::shared_ptr<GPContext> gpContext;
        SparseConv1DSamplingScheme sparseConv1DSamplingScheme;
        // x: scale for UNI, y: scale for NEE, z: scale for CPNEE
        Vec3f sparseConv1DSamplingSchemeSampleCountScale; 

        void reset()
        {
            firstScatter = true;
            bounce = 0;
            if (gpContext) {
                gpContext->reset();
            }
            lastGPId = 0;
            sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
            sparseConv1DSamplingSchemeSampleCountScale = Vec3f(1.f, 0.f, 0.f);
        }

        void advance()
        {
            firstScatter = false;
            sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
            sparseConv1DSamplingSchemeSampleCountScale = Vec3f(1.f, 0.f, 0.f);
            bounce++;
        }
    };

    Medium();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool isHomogeneous() const = 0;

    virtual void prepareForRender() {}
    virtual void teardownAfterRender() {}

    virtual Vec3f sigmaA(Vec3f p) const = 0;
    virtual Vec3f sigmaS(Vec3f p) const = 0;
    virtual Vec3f sigmaT(Vec3f p) const = 0;

    virtual bool sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
            MediumState &state, MediumSample &sample) const = 0;
    virtual bool invertDistance(WritablePathSampleGenerator &sampler, const Ray &ray, bool onSurface) const;
    virtual Vec3f transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface,
            bool endOnSurface, MediumState* state) const = 0;
    virtual float pdf(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const = 0;
    virtual Vec3f transmittanceAndPdfs(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface,
            bool endOnSurface, MediumState* state, float &pdfForward, float &pdfBackward) const;
    virtual const PhaseFunction *phaseFunction(const Vec3f &p) const;

    bool isDirac() const;
};

}



#endif /* MEDIUM_HPP_ */
