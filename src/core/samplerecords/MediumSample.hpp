#ifndef MEDIUMSAMPLE_HPP_
#define MEDIUMSAMPLE_HPP_

#include "math/Vec.hpp"
#include <Eigen/Dense>

namespace Tungsten {

class PhaseFunction;
struct GPContext;
enum class SparseConv1DSamplingScheme;
class Primitive;
struct CircleSegment;

struct RayInfo {
    Vec4u pixelSampleSegment;
    uint sceneSeed;
    float t;
};


struct MediumSample
{
    PhaseFunction *phase;
    Vec3f p;
    float continuedT;
    Vec3f continuedWeight;
    float t;
    Vec3f weight;
    Vec3f emission;
    float pdf;
    bool exited;
    Vec3d aniso;
    int gpId;
    SparseConv1DSamplingScheme sparseConv1DSamplingScheme;
    // x: scale for UNI, y: scale for NEE, z: scale for CPNEE
    Vec3f sparseConv1DSamplingSchemeSampleCountScale;
    const Primitive* light;
    bool is1DNEE;
    CircleSegment* segment;
    GPContext* ctxt;
    RayInfo rayInfo;
};

}

#endif /* MEDIUMSAMPLE_HPP_ */
