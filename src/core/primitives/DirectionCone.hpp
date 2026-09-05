#ifndef DIRECTIONCONE_HPP_
#define DIRECTIONCONE_HPP_
#include "math/Vec.hpp"
#include <algorithm>
#include <iostream>
 namespace Tungsten {

    struct CircleSegment
    {
        /// Pointing from the origin of the circle to the middle point of the circle segment.
        Vec3f op;
        /// The angle spaned by the arc.
        float angle;
        /// The rotation axis of the circle. Perpendicular to the circular plane.
        Vec3f axis;
        /// If the infinity point of Gaussian lies within the segment. 
        bool InfIncluded;
    };

    enum SegmentSamplingMethod {
        Equiangular = 0,
        Gaussian,
        NotSupported
    };

    class DirectionCircle
    {
        public:
            Vec3f axis; ///< Axis of the cone, pointing from the vertex. The length of the axis is the height of the cone. 
            /// Construct a direction cone from one axis and one vertex.
            DirectionCircle(const Vec3f& axis): axis(axis){};
            DirectionCircle() = default;

            /// @brief Take the c_l's edges that are included in c_s.  
            /// @param c_s Scattering circle.
            /// @param infPoint The point on the cone where two infinity points on the Gaussian meets. 
            /// This point is also along the incoming ray direction. 
            /// @param c_l Light circle. the cone spanned by the sphere light source. 
            /// @param segment The resulting subset of the scattering circle (c_s), defined by a circle segment. 
            /// @return If the 2 cones intersect.
            static bool intersect_two_circles(const DirectionCircle &c_s, const Vec3f &infPoint, 
                const DirectionCircle &c_l, CircleSegment &segment);
    };
}

#endif