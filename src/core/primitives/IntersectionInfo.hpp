#ifndef INTERSECTIONINFO_HPP_
#define INTERSECTIONINFO_HPP_

#include "math/Vec.hpp"
#include "samplerecords/MediumSample.hpp"

namespace Tungsten {

class Primitive;
class Bsdf;

struct IntersectionInfo
{
    Vec3f Ng;
    Vec3f Ns;
    Vec3f p;
    Vec3f w;
    Vec2f uv;
    float t;
    float epsilon;

    const Primitive *primitive;
    const Bsdf *bsdf;

    RayInfo rayInfo;
};

}

#endif /* INTERSECTIONINFO_HPP_ */
