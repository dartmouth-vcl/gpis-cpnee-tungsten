#ifndef MATH_BSPHERE_HPP_
#define MATH_BSPHERE_HPP_

#include "IntTypes.hpp"
#include "MathUtil.hpp"
#include "Vec.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>

namespace Tungsten {

template<typename TVec, typename ElementType, unsigned Size>
class BSphere {
    TVec _center;
    ElementType _radius;

public:
    BSphere()
    : _center(ElementType(0)),
      _radius(ElementType(0))
    {
    }

    BSphere(const TVec &center, ElementType radius)
    : _center(center),
      _radius(radius)
    {
    }

    inline const TVec &center() const
    {
        return _center;
    }

    inline TVec &center()
    {
        return _center;
    }

    inline ElementType radius() const
    {
        return _radius;
    }

    inline ElementType &radius()
    {
        return _radius;
    }

    inline bool empty() const
    {
        return _radius <= ElementType(0);
    }

    inline ElementType area() const
    {
        if (Size == 2)
            return ElementType(2) * ElementType(PI) * _radius;
        else if (Size == 3)
            return ElementType(4) * ElementType(PI) * _radius * _radius;
        return ElementType(0);
    }

    inline ElementType volume() const
    {
        if (Size == 2)
            return ElementType(PI) * _radius * _radius;
        else if (Size == 3)
            return (ElementType(4) / ElementType(3)) * ElementType(PI) * _radius * _radius * _radius;
        return ElementType(0);
    }

    void grow(const TVec &p)
    {
        ElementType dist(0);
        for (unsigned i = 0; i < Size; ++i)
            dist += (p[i] - _center[i]) * (p[i] - _center[i]);
        dist = std::sqrt(dist);
        _radius = std::max(_radius, dist);
    }

    void grow(ElementType t)
    {
        _radius += t;
    }

    inline bool contains(const TVec &p) const
    {
        ElementType dist2(0);
        for (unsigned i = 0; i < Size; ++i)
            dist2 += (p[i] - _center[i]) * (p[i] - _center[i]);
        return dist2 <= _radius * _radius;
    }

    template<bool Strict>
    inline bool contains(const TVec &p) const
    {
        ElementType dist2(0);
        for (unsigned i = 0; i < Size; ++i)
            dist2 += (p[i] - _center[i]) * (p[i] - _center[i]);
        if (Strict)
            return dist2 < _radius * _radius;
        else
            return dist2 <= _radius * _radius;
    }

    friend std::ostream &operator<<(std::ostream &stream, const BSphere &sphere) {
        stream << "(center=" << sphere._center << ", radius=" << sphere._radius << ')';
        return stream;
    }
};

typedef BSphere<Vec4f, float, 4> BSphere4f;
typedef BSphere<Vec3f, float, 3> BSphere3f;
typedef BSphere<Vec2f, float, 2> BSphere2f;

typedef BSphere<Vec4d, double, 4> BSphere4d;
typedef BSphere<Vec3d, double, 3> BSphere3d;
typedef BSphere<Vec2d, double, 2> BSphere2d;

}

#endif // MATH_SPHERE_HPP_