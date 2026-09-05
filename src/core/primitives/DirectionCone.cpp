#include "DirectionCone.hpp"
#include "math/MathUtil.hpp"

namespace Tungsten {

    Vec3f get_pt_on_plane_intersection(const Vec3f& n1, const Vec3f& n2, 
        const float d1, const float d2, Vec3f& shared_dir)  {
        //d1 and d2 must be larger or equal than 0. It is the distance from the origin to the plane. 
        if(d1 < 0.0f || d2 < 0.0f) {
            std::cerr << "Plane distance must be non-negative.";
            return Vec3f(1.0f);
        }
        Vec3f v = n1.cross(n2);
        Vec3f p = (d1 * n2.cross(v) - d2 * n1.cross(v)) / (v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
        shared_dir = v.normalized();
        return p;
    }

    bool ray_sphere_intersect(const Vec3f& d, const Vec3f& o, Vec3f& p1, Vec3f& p2){
        const double a = (double)d.lengthSq();
        if (!(a > 0.0)) return false;
        const double h = -(double)o.dot(d);
        const double c = (double)o.dot(o) - 1.0;
        double det = h * h - a * c;
        const double scale = std::max({ h * h, a * a, std::abs(a * c), 1.0 });
        const double eps = 1e-12 * scale;
        if (det <= eps) return false;
        const double s = std::sqrt(det);
        double q = h + (h >= 0.0 ? s : -s);
        if (q == 0.0) {
            const double tA = (h - s) / a;
            const double tB = (h + s) / a;
            p1 = o + (float)std::min(tA, tB) * d;
            p2 = o + (float)std::max(tA, tB) * d;
            return true;
            }
        const double t_far = q / a;
        const double t_near = c / q; 
        double t1 = std::min(t_near, t_far);
        double t2 = std::max(t_near, t_far);
        p1 = o + (float)t1 * d;
        p2 = o + (float)t2 * d;
        return true;
    }

    bool DirectionCircle::intersect_two_circles(const DirectionCircle& c_s, const Vec3f& infPoint,
        const DirectionCircle& c_l, 
        CircleSegment& segment){
        Vec3f n1 = c_s.axis.normalized();
        Vec3f n2 = c_l.axis.normalized();
        float len1 = c_s.axis.length();
        float len2 = c_l.axis.length();
        Vec3f d_ray;
        Vec3f o_ray = get_pt_on_plane_intersection(n1, n2, len1, len2, d_ray);
        float r_square = o_ray.lengthSq();
        if (r_square >= 1.0f) {
            // Fully contained when: 
            // Scattering circle is smaller than light circle, AND, 
            // Half angle difference is larger than beta. 
            float beta = acos(clamp(n1.dot(n2), -1.0f, 1.0f));
            float halfAngle_scattering = acos(clamp(len1, -1.0f, 1.0f));
            float halfAngle_light = acos(clamp(len2, -1.0f, 1.0f));
            float halfAngleDiff = halfAngle_light - halfAngle_scattering;
            bool fullyContained = halfAngleDiff > beta;
            if(!fullyContained) return false; 
            else{
                segment.angle = 2.0f * M_PI;
                // Sample a full gaussian. Don't need the following parameters here.
                // segment.op = Vec3f(0.0f);
                // segment.axis = c_s.axis;
                // segment.InfIncluded = true;
                return true;
            }
        }
        Vec3f d = d_ray.normalized();
        Vec3f o = o_ray;
        Vec3f ws1, ws2;
        bool intersect = ray_sphere_intersect(d, o, ws1, ws2);
        if (intersect) {
            Vec3f axis1 = (ws1 - c_s.axis).normalized();
            Vec3f axis2 = (ws2 - c_s.axis).normalized();
            Vec3f axis_mid = (axis1 + axis2).normalized(); // normalize((ws1 - cs) + (ws2 - cs))
            float cosine = axis_mid.dot(n2);
            float cosine_ws1ws2 = axis1.dot(axis2);
            // If cosine >= 0, we choose the minor arc.
            // Otherwise, we use the major arc.
            Vec3f axisX = cosine >= 0.0 ? axis_mid : -axis_mid;
            float angle_ws1ws2 = cosine >= 0.0f ? acos(cosine_ws1ws2)
                                                : 2.0 * M_PI - acos(cosine_ws1ws2);
            // The cosines between axisX and (infPoint - cs) / (ws1 - cs)
            float cosine_infaxis = (infPoint - c_s.axis).normalized().dot(axisX);
            float cosine_ws1axis = (ws1 - c_s.axis).normalized().dot(axisX);
            bool InfIncluded = cosine_infaxis > cosine_ws1axis;
            float r1 = sqrt(1.0f - c_s.axis.lengthSq());
            segment.op = axisX * r1;
            segment.angle = angle_ws1ws2;
            segment.axis = c_s.axis;
            segment.InfIncluded = InfIncluded;
            return true;
            }
        return false;
    }

}