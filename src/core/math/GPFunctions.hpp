#ifndef GPFUNCTIONS_HPP_
#define GPFUNCTIONS_HPP_

#include <math/Vec.hpp>
#include <math/Angle.hpp>
#include <math/TangentFrame.hpp>
#include "io/JsonSerializable.hpp"
#include "io/JsonObject.hpp"
#include <math/AffineArithmetic.hpp>
#include <sampling/SampleWarp.hpp>
#include <sampling/Gaussian.hpp>

#include <math/SdfFunctions.hpp>
#include <math/GPNeuralNetwork.hpp>

#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <igl/fast_winding_number.h>
#include <igl/AABB.h>

#include <boost/math/special_functions/bessel.hpp>

namespace Tungsten {
    using Dual = autodiff::dual;
    using FloatDD = autodiff::dual2nd;
    using Vec3DD = autodiff::Vector3dual2nd;
    using Mat3DD = autodiff::Matrix3dual2nd;
    using VecXDD = autodiff::VectorXdual2nd;

    using FloatD = autodiff::real2nd;
    using Vec3Diff = autodiff::Vector3real2nd;
    using Vec4Diff = autodiff::Vector4real2nd;
    using Mat3Diff = autodiff::Matrix3real2nd;
    using VecXDiff = autodiff::VectorXreal2nd;

    inline Vec3d from_diff(const Vec3DD& vd) {
        return Vec3d{ vd.x().val.val, vd.y().val.val, vd.z().val.val };
    }

    inline Vec3d from_diff(const Vec3Diff& vd) {
        return Vec3d{ vd.x().val(), vd.y().val(), vd.z().val() };
    }

    inline Vec3Diff to_diff(const Vec3f& vd) {
        return Vec3Diff{ vd.x(), vd.y(), vd.z() };
    }

    inline Vec3Diff to_diff(const Vec3d& vd) {
        return Vec3Diff{ vd.x(), vd.y(), vd.z() };
    }

    inline Vec3d to_vec3d(const Eigen::Vector3d& v) {
        return Vec3d(v(0), v(1), v(2));
    }

    inline Vec3f to_vec3f(const Eigen::Vector3f& v) {
        return Vec3f(v(0), v(1), v(2));
    }

    inline Eigen::Vector3d to_eigen3d(const Vec3d& v) {
        Eigen::Vector3d v_eigen;
        v_eigen << v.x(), v.y(), v.z();
        return v_eigen;
    }

    inline Eigen::Vector3f to_eigen3f(const Vec3f& v) {
        Eigen::Vector3f v_eigen;
        v_eigen << v.x(), v.y(), v.z();
        return v_eigen;
    }

    template<typename Vec>
    inline auto dist2_ab(Vec ab, Eigen::Matrix3f aniso) {
        return ab.transpose() * aniso * ab;
    }

    template<typename Vec>
    inline auto dist2_ab(Vec ab, Vec3f aniso) {
        return ab.dot(Vec{ aniso.x(), aniso.y(), aniso.z() }.cwiseProduct(ab));
    }

    template<typename Vec>
    inline auto dist2(Vec a, Vec b, Vec3f aniso) {
        auto d = b - a;
        return d.dot(Vec{ aniso.x(), aniso.y(), aniso.z() }.cwiseProduct(d));
    }

    template<typename Vec>
    inline auto dist2(Vec a, Vec b) {
        auto d = b - a;
        return d.dot(d);
    }

    template <typename Mat, typename Vec>
    inline Mat compute_ansio_full(const float& angle, const Vec& aniso);

    template <typename Mat, typename Vec>
    inline Mat compute_ansio_simplified(const Vec& grad, const Vec& aniso);

    template<typename Vec>
    static inline Vec mult(const Eigen::Matrix2d& a, const Vec& b)
    {
        return Vec(
            a(0, 0) * b.x() + a(0, 1) * b.y(),
            a(1, 0) * b.x() + a(1, 1) * b.y()
        );
    }

    template<typename Vec>
    static inline Vec mult(const Eigen::Matrix2f& a, const Vec& b)
    {
        return Vec(
                a(0, 0) * b.x() + a(0, 1) * b.y(),
                a(1, 0) * b.x() + a(1, 1) * b.y()
        );
    }

    template<typename Vec>
    static inline Vec mult(const Eigen::Matrix3d& a, const Vec& b)
    {
        return Vec(
            a(0, 0) * b.x() + a(0, 1) * b.y() + a(0, 2) * b.z(),
            a(1, 0) * b.x() + a(1, 1) * b.y() + a(1, 2) * b.z(),
            a(2, 0) * b.x() + a(2, 1) * b.y() + a(2, 2) * b.z()
        );
    }

    template<typename Vec>
    static inline Vec mult(const Eigen::Matrix3f& a, const Vec& b)
    {
        return Vec(
                a(0, 0) * b.x() + a(0, 1) * b.y() + a(0, 2) * b.z(),
                a(1, 0) * b.x() + a(1, 1) * b.y() + a(1, 2) * b.z(),
                a(2, 0) * b.x() + a(2, 1) * b.y() + a(2, 2) * b.z()
        );
    }

    template<typename Vec>
    static inline Vec mult(const Mat4f& a, const Vec& b)
    {
        return Vec(
            a(0, 0) * b.x() + a(0, 1) * b.y() + a(0, 2) * b.z() + a(0, 3),
            a(1, 0) * b.x() + a(1, 1) * b.y() + a(1, 2) * b.z() + a(1, 3),
            a(2, 0) * b.x() + a(2, 1) * b.y() + a(2, 2) * b.z() + a(2, 3)
        );
    }

    inline Vec3f filterWithZero(Vec3f input) {
        Vec3f output = input;
        if (std::isinf(output.x()) || std::isnan(output.x()))
            output.x() = 0;
        if (std::isinf(output.y()) || std::isnan(output.y()))
            output.y() = 0;
        if (std::isinf(output.z()) || std::isnan(output.z()))
            output.z() = 0;
        return output;
    }

    inline Vec3f filterWithOne(Vec3f input) {
        Vec3f output = input;
        if (std::isinf(output.x()) || std::isnan(output.x()))
            output.x() = 1;
        if (std::isinf(output.y()) || std::isnan(output.y()))
            output.y() = 1;
        if (std::isinf(output.z()) || std::isnan(output.z()))
            output.z() = 1;
        return output;
    }

    template<typename ElemType>
    ElemType trilinearInterpolation(const Vec3d& uv, ElemType(&data)[2][2][2]) {
        return lerp(
                lerp(
                        lerp(data[0][0][0], data[1][0][0], uv.x()),
                        lerp(data[0][1][0], data[1][1][0], uv.x()),
                        uv.y()
                ),
                lerp(
                        lerp(data[0][0][1], data[1][0][1], uv.x()),
                        lerp(data[0][1][1], data[1][1][1], uv.x()),
                        uv.y()
                ),
                uv.z()
        );
    };

    template<class ElemType, size_t N>
    ElemType triquadraticInterpolation(const Vec3d& uvw, ElemType(&data)[N][N][N])
    {
        auto _interpolate = [](const ElemType* value, double weight)
        {
            const ElemType
                    a = static_cast<ElemType>(0.5 * (value[0] + value[2]) - value[1]),
                    b = static_cast<ElemType>(0.5 * (value[2] - value[0])),
                    c = static_cast<ElemType>(value[1]);
            const auto temp = weight * (weight * a + b) + c;
            return static_cast<ElemType>(temp);
        };

        /// @todo For vector types, interpolate over each component independently.
        ElemType vx[3];
        for (int dx = 0; dx < 3; ++dx) {
            ElemType vy[3];
            for (int dy = 0; dy < 3; ++dy) {
                // Fit a parabola to three contiguous samples in z
                // (at z=-1, z=0 and z=1), then evaluate the parabola at z',
                // where z' is the fractional part of inCoord.z, i.e.,
                // inCoord.z - inIdx.z.  The coefficients come from solving
                //
                // | (-1)^2  -1   1 || a |   | v0 |
                // |    0     0   1 || b | = | v1 |
                // |   1^2    1   1 || c |   | v2 |
                //
                // for a, b and c.
                const ElemType* vz = &data[dx][dy][0];
                vy[dy] = _interpolate(vz, uvw.z());
            }//loop over y
            // Fit a parabola to three interpolated samples in y, then
            // evaluate the parabola at y', where y' is the fractional
            // part of inCoord.y.
            vx[dx] = _interpolate(vy, uvw.y());
        }//loop over x
        // Fit a parabola to three interpolated samples in x, then
        // evaluate the parabola at the fractional part of inCoord.x.
        return _interpolate(vx, uvw.x());
    }

    template<typename ElemType>
    static ElemType bilinearInterpolation(const Vec2d& uv, ElemType(&data)[2][2]) {
        return lerp(
                lerp(data[0][0], data[1][0], uv.x()),
                lerp(data[0][1], data[1][1], uv.x()),
                uv.y()
        );
    };

    template<typename ElemType>
    struct RegularGrid : JsonSerializable {

        InterpolateMethod interp = InterpolateMethod::Linear;

        RegularGrid(Box3d bounds = Box3d(), size_t res = 0, std::vector<ElemType> values = {}, InterpolateMethod interp = InterpolateMethod::Linear)
                : bounds(bounds), res(res), values(values), interp(interp) {}

        Box3d bounds;
        size_t res;
        std::vector<ElemType> values;

        PathPtr path;

        ElemType getValue(Vec3i coord) const {
            coord = clamp(coord, Vec3i(0), Vec3i(res - 1));
            return values[ (coord.x() * res + coord.y()) * res + coord.z()];
        };

        void getValues(const Vec3i& coord, ElemType(&data)[2][2][2]) const {
            data[0][0][0] = getValue(coord + Vec3i(0, 0, 0));
            data[1][0][0] = getValue(coord + Vec3i(1, 0, 0));
            data[0][1][0] = getValue(coord + Vec3i(0, 1, 0));
            data[1][1][0] = getValue(coord + Vec3i(1, 1, 0));

            data[0][0][1] = getValue(coord + Vec3i(0, 0, 1));
            data[1][0][1] = getValue(coord + Vec3i(1, 0, 1));
            data[0][1][1] = getValue(coord + Vec3i(0, 1, 1));
            data[1][1][1] = getValue(coord + Vec3i(1, 1, 1));
        };

        void getValues(const Vec3i& coord, ElemType(&data)[3][3][3]) const {
            Vec3i inLoIdx = coord - 1;
            // Retrieve the values of the 27 voxels surrounding the
            // fractional source coordinates.
            for (int dx = 0, ix = inLoIdx.x(); dx < 3; ++dx, ++ix) {
                for (int dy = 0, iy = inLoIdx.y(); dy < 3; ++dy, ++iy) {
                    for (int dz = 0, iz = inLoIdx.z(); dz < 3; ++dz, ++iz) {
                        data[dx][dy][dz] = getValue(Vec3i{ ix, iy, iz });
                    }
                }
            }
        }

        ElemType getValue(Vec3d p) const {
            Vec3d p_grid = (double(res) * (p - bounds.min()) / bounds.diagonal()) - 0.5;

            Vec3i coord = Vec3i(std::floor(p_grid));
            Vec3d uvw = p_grid - Vec3d(coord);

            switch (interp) {
                case InterpolateMethod::Point:
                    return getValue(coord);
                case InterpolateMethod::Linear:
                {
                    ElemType data[2][2][2];
                    getValues(coord, data);
                    return trilinearInterpolation(uvw, data);
                }
                case InterpolateMethod::Quadratic:
                {
                    ElemType data[3][3][3];
                    getValues(coord, data);
                    return triquadraticInterpolation(uvw, data);
                }
            }
        }

        std::vector<Vec3d> makePoints(bool centered = false) const {
            std::vector<Vec3d> points(res * res * res);
            int idx = 0;
            for (int i = 0; i < res; i++) {
                for (int j = 0; j < res; j++) {
                    for (int k = 0; k < res; k++) {
                        if (centered) {
                            points[idx] = lerp(bounds.min(), bounds.max(), (Vec3d((double)i + 0.5, (double)j + 0.5, (double)k + 0.5) / res));
                        }
                        else {
                            points[idx] = lerp(bounds.min(), bounds.max(), (Vec3d((double)i, (double)j, (double)k) / (res - 1)));
                        }
                        idx++;
                    }
                }
            }
            return points;
        }

        virtual void saveResources() override {
            if (path) {
                std::ofstream xfile(path->absolute().asString(), std::ios::out | std::ios::binary);
                xfile.write((char*)values.data(), sizeof(values[0]) * values.size());
                xfile.close();
            }
        }


        virtual void loadResources() override {
            if (path) {
                std::ifstream xfile(path->absolute().asString(), std::ios::in | std::ios::binary);
                ElemType value;
                while (xfile.read(reinterpret_cast<char*>(&value), sizeof(ElemType))) {
                    values.push_back(value);
                }
                xfile.close();
            }
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                               "type", "regular_grid",
                               "bounds_min", bounds.min(),
                               "bounds_max", bounds.max(),
                               "res", res,
                               "path", *path,
                               "interpolate", interpolateMethodToString(interp)
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            JsonSerializable::fromJson(value, scene);
            value.getField("bounds_min", bounds.min());
            value.getField("bounds_max", bounds.max());
            value.getField("res", res);

            std::string interpString = interpolateMethodToString(interp);
            value.getField("interpolate", interpString);
            interp = stringToInterpolateMethod(interpString);

            if (auto f = value["path"]) path = scene.fetchResource(f);
        }
    };

    template<typename Elem>
    void save_grid(RegularGrid<Elem>& grid, Path path) {

        DirectoryChange context(path.parent());

        grid.path = std::make_shared<Path>(path.stripParent().stripExtension() + "-values.bin");
        grid.saveResources();

        rapidjson::Document document;
        document.SetObject();
        *(static_cast<rapidjson::Value*>(&document)) = grid.toJson(document.GetAllocator());
        FileUtils::writeJson(document, path.stripParent());
    }

    template<typename Elem>
    RegularGrid<Elem> load_grid(Path path) {
        JsonDocument document(path);

        DirectoryChange context(path.parent());

        Scene scene(path.parent(), nullptr);
        scene.setPath(path);

        RegularGrid<Elem> grid;
        grid.fromJson(document, scene);
        grid.loadResources();

        return grid;
    }

    class ProceduralScalar : public JsonSerializable {
    public:
        virtual double operator()(Vec3d p) const = 0;
    };

    class ConstantScalar : public ProceduralScalar {
        double _v;
    public:
        ConstantScalar(double v = 0) : _v(v) { }
        virtual double operator()(Vec3d p) const override {
            return _v;
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{
                    ProceduralScalar::toJson(allocator), allocator,
                    "type", "constant",
                    "val", _v,
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralScalar::fromJson(value, scene);
            value.getField("val", _v);
        }
    };

    class RegularGridScalar : public ProceduralScalar {
        std::shared_ptr<RegularGrid<double>> _grid;
        PathPtr _file;
    public:
        RegularGridScalar(std::shared_ptr<RegularGrid<double>> grid = nullptr) : _grid(grid) {}

        virtual double operator()(Vec3d p) const override {
            return _grid->getValue(p);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                               "type", "regular_grid",
                               "file", *_file
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralScalar::fromJson(value, scene);
            if (auto f = value["file"]) _file = scene.fetchResource(f);
        }

        virtual void loadResources() override {
            _grid = std::make_shared<RegularGrid<double>>();
            *_grid = load_grid<double>(*_file);
        }
    };

    class LinearRampScalar : public ProceduralScalar {
        Vec2d _minMax;
        Vec3d _dir;
        Vec2d _range;
    public:
        LinearRampScalar(Vec2d minMax, Vec3d dir, Vec2d range) : _minMax(minMax), _dir(dir), _range(range) { }

        virtual double operator()(Vec3d p) const override {
            double i = p.dot(_dir);
            i = clamp((i - _range.x()) / (_range.y() - _range.x()), 0., 1.);
            return lerp(_minMax.x(), _minMax.y(), i);
        }
    };

    class ProceduralVector : public JsonSerializable {
    public:
        virtual Vec3d operator()(Vec3d p) const = 0;
        virtual float maxVal() const {
            std::cout << "maxVal() not implemented!\n";
            return 0.f;
        };
    };

    class ConstantVector : public ProceduralVector {
        Vec3d _v;
    public:
        ConstantVector(Vec3d v = Vec3d(0.)) : _v(v) { }
        virtual Vec3d operator()(Vec3d p) const override {
            return _v;
        }

        virtual float maxVal() const override { return max(max(_v.x(), _v.y()), _v.z()); }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{
                    ProceduralVector::toJson(allocator), allocator,
                    "type", "constant",
                    "val", _v,
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralVector::fromJson(value, scene);
            value.getField("val", _v);
        }
    };

    class RegularGridVector : public ProceduralVector {
        std::shared_ptr<RegularGrid<Vec3d>> _grid;
        PathPtr _file;
    public:
        RegularGridVector(std::shared_ptr<RegularGrid<Vec3d>> grid = nullptr) : _grid(grid) {}

        virtual Vec3d operator()(Vec3d p) const override {
            return _grid->getValue(p);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralVector::toJson(allocator), allocator,
                               "type", "regular_grid",
                               "file", *_file
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralVector::fromJson(value, scene);
            if (auto f = value["file"]) _file = scene.fetchResource(f);
        }

        virtual void loadResources() override {
            _grid = std::make_shared<RegularGrid<Vec3d>>();
            *_grid = load_grid<Vec3d>(*_file);
        }
    };

    class LinearRampVector : public ProceduralVector {
        Vec3d _min;
        Vec3d _max;
        Vec3d _dir;
        Vec2d _range;
    public:
        LinearRampVector(Vec3d min, Vec3d max, Vec3d dir, Vec2d range) : _min(min), _max(max), _dir(dir), _range(range) { }

        virtual Vec3d operator()(Vec3d p) const override {
            double i = p.dot(_dir);
            i = clamp((i - _range.x()) / (_range.y() - _range.x()), 0., 1.);
            return lerp(_min, _max, i);
        }
    };

    class ProceduralScalarCode : public ProceduralScalar {
        std::function<double(Vec3d)> _fn;
    public:
        ProceduralScalarCode(std::function<double(Vec3d)> fn = nullptr) : _fn(fn) {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                               "type", "code"
            };
        }

        virtual double operator()(Vec3d p) const override {
            return _fn(p);
        }
    };

    class ProceduralVectorCode : public ProceduralVector {
        std::function<Vec3d(Vec3d)> _fn;
    public:
        ProceduralVectorCode(std::function<Vec3d(Vec3d)> fn = nullptr) : _fn(fn) {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralVector::toJson(allocator), allocator,
                               "type", "code"
            };
        }

        virtual Vec3d operator()(Vec3d p) const override {
            return _fn(p);
        }
    };

    class ProceduralSdf : public ProceduralScalar {
        SdfFunctions::Function _fn;
    public:
        ProceduralSdf(SdfFunctions::Function fn = SdfFunctions::Function::Knob) : _fn(fn) {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                               "type", "sdf",
                               "function", SdfFunctions::functionToString(_fn)
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralScalar::fromJson(value, scene);
            std::string fnString;
            value.getField("function", fnString);
            _fn = SdfFunctions::stringToFunction(fnString);
        }

        virtual double operator()(Vec3d p) const override {
            int mat;
            return SdfFunctions::eval(_fn, vec_conv<Vec3f>(p), mat);
        }
    };

    class ProceduralNoise: public ProceduralScalar {
        double _min = 1., _max = 500.;
        double _start = 0., _end = 1.;
        double _scale = 1., _offset = 0.;
        double _min_2 = 1., _max_2 = 500.;
        double _start_2 = 0., _end_2 = 1.;
        double _scale_2 = 1., _offset_2 = 0.;
        double _const = 1.;

        enum class NoiseType {
            BottomTop,
            LeftRight,
            FrontBack,
            BottomTopLeftRight,
            Sandstone,
            Rust
        };

        NoiseType type = NoiseType::BottomTop;

        static std::string noiseTypeToString(NoiseType v) {
            switch (v) {
                case NoiseType::BottomTop: return "bottom_top";
                case NoiseType::LeftRight: return "left_right";
                case NoiseType::FrontBack: return "front_back";
                case NoiseType::BottomTopLeftRight: return "bottom_top_left_right";
                case NoiseType::Sandstone: return "sandstone";
                case NoiseType::Rust: return "rust";
            }
        }

        static NoiseType stringToNoiseType(std::string v) {
            if (v == "bottom_top")
                return NoiseType::BottomTop;
            else if (v == "left_right")
                return NoiseType::LeftRight;
            else if (v == "front_back")
                return NoiseType::FrontBack;
            else if (v == "bottom_top_left_right")
                return NoiseType::BottomTopLeftRight;
            else if (v == "sandstone")
                return NoiseType::Sandstone;
            else if (v == "rust")
                return NoiseType::Rust;

            FAIL("Invalid noise typ function: '%s'", v);
        }

    public:
        ProceduralNoise() {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                               "type", "noise",
                               "noise", noiseTypeToString(type),
                               "min", _min,
                               "max", _max,
                               "start", _start,
                               "end", _end,
                               "min2", _min_2,
                               "max2", _max_2,
                               "start2", _start_2,
                               "end2", _end_2,
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralScalar::fromJson(value, scene);
            std::string noiseString = noiseTypeToString(type);
            value.getField("noise", noiseString);
            type = stringToNoiseType(noiseString);

            value.getField("min", _min);
            value.getField("max", _max);
            value.getField("start", _start);
            value.getField("end", _end);
            value.getField("min2", _min_2);
            value.getField("max2", _max_2);
            value.getField("start2", _start_2);
            value.getField("end2", _end_2);
            _scale = 1.0 / (_end - _start);
            _offset = - _start * _scale;
            _scale_2 = 1.0 / (_end_2 - _start_2);
            _offset_2 = - _start_2 * _scale_2;
        }

        virtual double operator()(const Vec3d p) const override;
    };

    class ProceduralNoiseVec: public ProceduralVector {
        double _min = 1., _max = 500.;
        double _start = 0., _end = 1.;
        double _scale = 1., _offset = 0.;
        double _min_2 = 1., _max_2 = 500.;
        double _start_2 = 0., _end_2 = 1.;
        double _scale_2 = 1., _offset_2 = 0.;
        double _const = 1.;

        enum class NoiseType {
            BottomTop,
            LeftRight,
            FrontBack,
            BottomTopLeftRight,
            Sandstone,
            Rust
        };

        NoiseType type = NoiseType::BottomTop;

        static std::string noiseTypeToString(NoiseType v) {
            switch (v) {
                case NoiseType::BottomTop: return "bottom_top";
                case NoiseType::LeftRight: return "left_right";
                case NoiseType::FrontBack: return "front_back";
                case NoiseType::BottomTopLeftRight: return "bottom_top_left_right";
                case NoiseType::Sandstone: return "sandstone";
                case NoiseType::Rust: return "rust";
            }
        }

        static NoiseType stringToNoiseType(std::string v) {
            if (v == "bottom_top")
                return NoiseType::BottomTop;
            else if (v == "left_right")
                return NoiseType::LeftRight;
            else if (v == "front_back")
                return NoiseType::FrontBack;
            else if (v == "bottom_top_left_right")
                return NoiseType::BottomTopLeftRight;
            else if (v == "sandstone")
                return NoiseType::Sandstone;
            else if (v == "rust")
                return NoiseType::Rust;

            FAIL("Invalid noise typ function: '%s'", v);
        }

    public:
        ProceduralNoiseVec() {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralVector::toJson(allocator), allocator,
                               "type", "noise",
                               "noise", noiseTypeToString(type),
                               "min", _min,
                               "max", _max,
                               "start", _start,
                               "end", _end,
                               "min2", _min_2,
                               "max2", _max_2,
                               "start2", _start_2,
                               "end2", _end_2,
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralVector::fromJson(value, scene);
            std::string noiseString = noiseTypeToString(type);
            value.getField("noise", noiseString);
            type = stringToNoiseType(noiseString);

            value.getField("min", _min);
            value.getField("max", _max);
            value.getField("start", _start);
            value.getField("end", _end);
            value.getField("min2", _min_2);
            value.getField("max2", _max_2);
            value.getField("start2", _start_2);
            value.getField("end2", _end_2);
            _scale = 1.0 / (_end - _start);
            _offset = - _start * _scale;
            _scale_2 = 1.0 / (_end_2 - _start_2);
            _offset_2 = - _start_2 * _scale_2;
        }

        virtual Vec3d operator()(const Vec3d p) const override;

        virtual float maxVal() const override;
    };



    class Grid;
    class MeanFunction;
    class StationaryCovariance;

    enum class Derivative : uint8_t {
        None = 0,
        First = 1
    };

    class MeanFunction : public JsonSerializable {
    public:
        std::shared_ptr<ProceduralVector> _col = nullptr;
        std::shared_ptr<ProceduralVector> _emission = nullptr;

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            JsonSerializable::fromJson(value, scene);

            if (auto c = value["color"]) {
                _col = scene.fetchProceduralVector(c);
            }

            if (auto e = value["emission"]) {
                _emission = scene.fetchProceduralVector(e);
            }
        }

        virtual void loadResources() override {
            if (_col) _col->loadResources();
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            auto obj = JsonObject{ JsonSerializable::toJson(allocator), allocator };
            if (_col) {
                obj.add("color", *_col);
            }

            if (_emission) {
                obj.add("emission", *_emission);
            }

            return obj;
        }

        double operator()(Derivative a, Vec3d p, Vec3d d) const {
            if (a == Derivative::None) {
                return mean(p);
            }
            else {
                return d.dot(dmean_da(p));
            }
        }

        virtual Vec3d dmean_da(Vec3d a) const = 0;

        virtual Affine<1> mean(const Affine<3>& a) const {
            assert(false && "Not implemented!");
            return Affine<1>(0.);
        }

        virtual double lipschitz() const {
            return 1.;
        }

        virtual Vec3d color(Vec3d a) const {
            if (!_col) { return Vec3d(1.); }
            return (*_col)(a);
        }

        virtual Vec3d emission(Vec3d a) const {
            if (!_emission) { return Vec3d(0.); }
            return (*_emission)(a);
        }

        virtual Vec3d shell_embedding(Vec3d a) const {
            return a;
        }

    private:
        virtual double mean(Vec3d a) const = 0;
    };

    class HomogeneousMean : public MeanFunction {
    public:
        HomogeneousMean(float offset = 0.f) : _offset(offset) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);
            value.getField("offset", _offset);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "homogeneous",
                "offset", _offset
            };
        }

        virtual double lipschitz() const override {
            return 0.;
        }

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return Affine<1>(_offset);
        }

    private:
        float _offset;

        virtual double mean(Vec3d a) const override {
            return _offset;
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            return Vec3d(0.);
        }
    };

    class SphericalMean : public MeanFunction {
    public:

        SphericalMean(Vec3d c = Vec3d(0.), float r = 1.) : _c(c), _r(r) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);
            value.getField("center", _c);
            value.getField("radius", _r);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "spherical",
                "center", _c,
                "radius", _r,
            };
        }

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return (a - _c).length() - _r;
        }

        virtual Vec3d shell_embedding(Vec3d a) const {
            Vec3d pc = a - _c;
            double r = pc.length();
            double theta = acos(pc.y() / r);
            double phi = atan2(pc.z(), pc.x());
            return Vec3d(theta * _r, phi * _r, (r - _r) );
        }

    private:
        Vec3d _c;
        float _r;

        virtual double mean(Vec3d a) const override {
            return (a - _c).length() - _r;
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            return (a - _c).normalized();
        }
    };

    class LinearMean : public MeanFunction {
    public:

        LinearMean(Vec3d ref = Vec3d(0.), Vec3d dir = Vec3d(1., 0., 0.), float scale = 1.0f, float min = -FLT_MAX) :
            _ref(ref), _dir(dir.normalized()), _scale(scale), _min(min), _tf(Vec3f(dir.normalized())) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);

            value.getField("reference_point", _ref);
            value.getField("direction", _dir);
            value.getField("scale", _scale);
            value.getField("min", _min);
            _dir.normalize();
            _tf = TangentFrame(Vec3f(_dir));
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "linear",
                "reference_point", _ref,
                "direction", _dir,
                "scale", _scale,
                "min", _min
            };
        }

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return dot(_dir, a - _ref) * _scale;
        }

        virtual Vec3d shell_embedding(Vec3d a) const {
            return Vec3d(_tf.toLocal(Vec3f(a)));
        }

    private:
        Vec3d _ref;
        Vec3d _dir;
        float _scale;
        float _min = -FLT_MAX;
        TangentFrame _tf;

        virtual double mean(Vec3d a) const override {
            return max((a - _ref).dot(_dir) * _scale, (double)_min);
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            if ((a - _ref).dot(_dir) * _scale < _min) {
                return Vec3d(0.);
            }
            else {
                return _dir * _scale;
            }
        }

        virtual double lipschitz() const {
            return _scale * _dir.length();
        }
    };

    class TabulatedMean : public MeanFunction {
    public:

        TabulatedMean(std::shared_ptr<Grid> grid = nullptr, float offset = 0, float scale = 1, bool isVolume = false) : _grid(grid), _offset(offset), _scale(scale), _isVolume(isVolume) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        std::shared_ptr<Grid> _grid;
        float _offset = 0;
        float _scale = 1;
        bool _isVolume = false;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d emission(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class NeuralMean : public MeanFunction {
    public:

        NeuralMean(std::shared_ptr<GPNeuralNetwork> nn = nullptr, float offset = 0, float scale = 1) : _nn(nn), _offset(offset), _scale(scale) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        std::shared_ptr<GPNeuralNetwork> _nn;
        float _offset = 0;
        float _scale = 1;
        PathPtr _path;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class ProceduralMean : public MeanFunction {
    public:

        ProceduralMean(std::shared_ptr<ProceduralScalar> f = nullptr) : _f(f) {}
        ProceduralMean(SdfFunctions::Function fn) {
            _f = std::make_shared<ProceduralSdf>(fn);
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        virtual void loadResources() override {
            MeanFunction::loadResources();
            if (_f) _f->loadResources();
        }

        virtual Vec3d color(Vec3d a) const override  {
            if (!_col) {
                return Vec3d(1.);
            }
            else {
                a = vec_conv<Vec3d>(_invConfigTransform.transformPoint(vec_conv<Vec3f>(a)));
                return (*_col)(a);
            }
        }

    private:
        std::shared_ptr<ProceduralScalar> _f;

        float _min = -FLT_MAX;
        float _offset = 0;
        float _scale = 1.f;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class MeshSdfMean : public MeanFunction {
    public:

        MeshSdfMean(PathPtr path = nullptr, bool isSigned = false) : _path(path), _signed(isSigned) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual Vec3d color(Vec3d a) const override;
        virtual Vec3d shell_embedding(Vec3d a) const override;


        Box3d bounds() {
            return _bounds;
        }

    private:
        PathPtr _path;
        bool _signed;
        double _min = -DBL_MAX;

        Eigen::MatrixXd V;
        Eigen::MatrixXi T, F;
        Eigen::MatrixXd FN, VN, EN;
        Eigen::MatrixXi E;
        Eigen::VectorXi EMAP;

        std::vector<Vec3f> _colors;
        std::vector<Vec2f> _uvs;

        igl::FastWindingNumberBVH fwn_bvh;
        igl::AABB<Eigen::MatrixXd, 3> tree;

        Box3d _bounds;
        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class CovarianceFunction : public JsonSerializable {

    public:
        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            JsonSerializable::fromJson(value, scene);
            value.getField("lateralScale", _lateralScale);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                               "lateralScale", _lateralScale
            };
        }

        virtual double operator()(Derivative a, Derivative b, Vec3d pa, Vec3d pb, Vec3d gradDirA, Vec3d gradDirB) const {
            if (a == Derivative::None && b == Derivative::None) {
                return cov(pa, pb);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (double)dcov_da(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirA));
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (double)dcov_db(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirB));
            }
            else {
                return (double)dcov2_dadb(vec_conv<Vec3DD>(pa), vec_conv<Vec3DD>(pb), vec_conv<Eigen::Array3d>(gradDirA), vec_conv<Eigen::Array3d>(gradDirB));
            }
        }

        // Note: differentiating with autodiff (for sparse convolution noise) is slow
        // The sparse convolution noise implementation always uses manual differentiation
        Vec4f splattingKernel3D_autodiff(Vec3d pa, Vec3d pb) const {
            float f = splatting_kernel_3D(Derivative::None, Derivative::None, pa, pb, Vec3d(), Vec3d());
            float dfdx = splatting_kernel_3D(Derivative::First, Derivative::None, pa, pb, Vec3d(1., 0., 0.), Vec3d());
            float dfdy = splatting_kernel_3D(Derivative::First, Derivative::None, pa, pb, Vec3d(0., 1., 0.), Vec3d());
            float dfdz = splatting_kernel_3D(Derivative::First, Derivative::None, pa, pb, Vec3d(0., 0., 1.), Vec3d());
            return Vec4f(f, dfdx, dfdy, dfdz);
        }

        Vec4f splattingKernel3DIsotropic_autodiff(Vec3d pa, Vec3d pb) const {
            float f = splatting_kernel_3D_isotropic(Derivative::None, Derivative::None, pa, pb, Vec3d(), Vec3d());
            float dfdx = splatting_kernel_3D_isotropic(Derivative::First, Derivative::None, pa, pb, Vec3d(1., 0., 0.), Vec3d());
            float dfdy = splatting_kernel_3D_isotropic(Derivative::First, Derivative::None, pa, pb, Vec3d(0., 1., 0.), Vec3d());
            float dfdz = splatting_kernel_3D_isotropic(Derivative::First, Derivative::None, pa, pb, Vec3d(0., 0., 1.), Vec3d());
            return Vec4f(f, dfdx, dfdy, dfdz);
        }

        Vec2f splattingKernel1D_autodiff(float pa, float pb) const {
            float f = splatting_kernel_1D(Derivative::None, Derivative::None, pa, pb);
            float dfdx = splatting_kernel_1D(Derivative::First, Derivative::None, pa, pb);
            return Vec2f(f, dfdx);
        }

        virtual double getVariance(const Vec3d p) const { return 1.f; }
        virtual float getUnscaledVariance(const Vec3d p) const { return 1.f; }

        Vec4f splattingKernel3D(Vec3f pa, Vec3f pb, bool isCov, bool isIsotropic, float globalScale, const Vec3f& p_world) const;

        Vec4f splattingKernel3DGrad(Vec3f pa, Vec3f pb, Vec3f coeff, bool isCov, bool isIsotropic, float globalScale, const Vec3f& p_world) const;

        virtual Vec2f splattingKernel1D(float pQuery, float pCenter, const Vec3f& pCenterWorld, const Vec3f& rayDirectionWorld) const;

        Vec2f covarianceKernel1D(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const;

        Vec2f covarianceKernel1DGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const;

        float covarianceKernel1DGradFor3DNormal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const;

        virtual bool isStationaryKernel() const { return false; }
        virtual bool isNonstationaryAnisotropicKernel() const { return false; }

        virtual bool useMultiResolutionGrid() const { return false; }

        virtual void getNonstationaryAniso3D(const Vec3f& p, std::shared_ptr<Eigen::Matrix3f>& aniso) const {
            aniso.reset();
            return;
        }

        virtual float getNonstationaryAniso1D(const Vec3f& p, const Vec3f& dirWorld) const { return 1.0; }

        virtual Eigen::Matrix3f getNonstationaryCovSplatCov3D(const Vec3f& pQueryWorld, const Vec3f& pCenterWorld) const { return Eigen::Matrix3f::Identity(); }

        virtual float getNonstationaryCovSplatCov1D(const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& dirLocal) const { return 1.0; }

        virtual float sparseConvNoiseLateralScale(const Vec3f& p) const {
            std::cerr << "sparseConvNoiseLateralScale not implemented!\n";
            return 1.0;
        }

        virtual float sparseConvNoiseAmplitude(const Vec3f& p) const {
            std::cerr << "sparseConvNoiseAmplitude(const Vec3f& p) not implemented!\n";
            return 0.;
        }

        virtual Eigen::Matrix3f sparseConvNoiseOneOverSecondDerivative(const Vec3f& p, std::shared_ptr<Eigen::Matrix3f> aniso_inv, bool isIsotropic) const {
            std::cerr << "sparseConvNoiseOneOverSecondDerivative() not implemented!\n";
            return Eigen::Matrix3f::Identity();
        }

        virtual float nonStationarySplattingKernelScale(const Vec3f& p) const {
            std::cerr << "nonStationarySplattingKernelScale not implemented!\n";
            return 1.;
        }

        virtual float worldSamplingSpatialScale() const {
            std::cerr << "worldSamplingSpatialScale not implemented!\n";
            return 1.;
        }

        virtual float splattingKernelRadius(bool isIdentity, float localScale) const {
            std::cerr << "splattingKernelRadius not implemented!\n";
            return 0.;
        }

        virtual float sparseConvNoiseVariance3D(const Vec3f& p, float impulseDensity, float kernelRadius, bool isIdentity, float localScale) const {
            std::cerr << "sparseConvNoiseVariance3D(const Vec3f& p) not implemented!\n";
            return 0.;
        }

        virtual float sparseConvNoiseVariance1D(const Vec3f& p, float impulseDensity, float kernelRadius) const {
            std::cerr << "sparseConvNoiseVariance1D(const Vec3f& p) not implemented!\n";
            return 0.;
        }

        virtual Vec3f transformPosDirWorldtoLocal(const Vec3f& posOrDir, const float localScale) const {
            std::cerr << "transformPosDirWorldtoLocal not implemented!\n";
            return posOrDir;
        }

        virtual Vec3f transformPosDirLocaltoWorld(const Vec3f& posOrDir, const float localScale) const {
            std::cerr << "transformPosDirLocaltoWorld not implemented!\n";
            return posOrDir;
        }

        virtual Vec3f transformGradWorldtoLocal(const Vec3f& grad, const float localScale) const {
            std::cerr << "transformGradWorldtoLocal not implemented!\n";
            return grad;
        }

        virtual Vec3f transformGradLocaltoWorld(const Vec3f& grad, const float localScale) const {
            std::cerr << "transformGradLocaltoWorld not implemented!\n";
            return grad;
        }

        virtual Eigen::Matrix3f worldToLocalMatrix(const Vec3f& p) const {
            std::cerr << "worldToLocalMatrix not implemented!\n";
        }

        virtual Eigen::Matrix3f localToWorldMatrix(const Vec3f& p) const {
            std::cerr << "localToWorldMatrix not implemented!\n";
        }

        virtual Eigen::Matrix3f worldToLocalInvTransposeMatrix(const Vec3f& p) const {
            std::cerr << "worldToLocalInvTransposeMatrix not implemented!\n";
        }

        virtual Eigen::Matrix3f localToWorldInvTransposeMatrix(const Vec3f& p) const {
            std::cerr << "localToWorldInvTransposeMatrix not implemented!\n";
        }

        virtual Vec3f maxCorrelationDirection() const {
            std::cerr << "maxCorrelationDirection() not implemented!\n";
        }

        double spectral_density(Derivative a, Derivative b, double s) {
            if (a == Derivative::None && b == Derivative::None) {
                return spectral_density(s);
            }
            return 0.;
        }

        virtual bool hasAnalyticSpectralDensity() const { return false; }
        virtual bool requireProjection() const { return false; }
        virtual double spectral_density(double s) const;
        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const;
        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const;
        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const;

        virtual void loadResources() override;

        virtual bool isMonotonic() const { return true; }

        double compute_beckmann_roughness(Vec3d p = Vec3d(0.)) {
            double L2 = (*this)(Derivative::First, Derivative::First, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.));
            return sqrt(2 * L2);
        }

        double compute_rices_formula(Vec3d p = Vec3d(0.), double u = 0.) {
            float L0 = (*this)(Derivative::None, Derivative::None, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.));
            float L2 = (*this)(Derivative::First, Derivative::First, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.));
            return exp(-u * u / (2 * L0)) * sqrt(L2 / L0) / (2 * PI);
        }

        virtual std::string id() const = 0;

        Vec3f _aniso = Vec3f(1.f);
        bool _localAniso = false;
        float _lateralScale = 1.0;

        std::vector<double> discreteSpectralDensity;

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const = 0;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const = 0;
        virtual double cov(Vec3d a, Vec3d b) const = 0;

        virtual FloatD dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const;
        virtual FloatD dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const;
        virtual FloatDD dcov2_dadb(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const;

        virtual float splattingKernel3DVal(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
            std::cerr << "splattingKernel3DVal(Vec3d a, Vec3d b) not implemented!\n";
            return 0.;
        }

        virtual Vec3f splattingKernel3D1stGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
            std::cerr << "splattingKernel3D1stGrad(Vec3f a, Vec3f b) not implemented!\n";
            return Vec3f(0.);
        }

        virtual Eigen::Matrix3f splattingKernel3D2ndGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
            std::cerr << "splattingKernel3D2ndGrad(Vec3f a, Vec3f b) not implemented!\n";
            return Eigen::Matrix3f(1, 1);
        }

        virtual float splattingKernel1DVal(float a, float b, float localScale) const {
            std::cerr << "splattingKernel1DVal(float a, float b) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel1DVal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirection) const {
            std::cerr << "covarianceKernel1DVal(float a, float b) not implemented!\n";
            return 0.;
        }

        virtual float splattingKernel1D1stGrad(float a, float b, float localScale) const {
            std::cerr << "splattingKernel1D1stGrad(double a, double b) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel1D1stGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirection) const {
            std::cerr << "covarianceKernel1D1stGrad(double a, double b) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel2D2ndGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirection) const {
            std::cerr << "covarianceKernel2D2ndGrad(double a, double b) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel2D2ndGradFor3DNormal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
            std::cerr << "covarianceKernel2D2ndGradFor3DNormal(double a, double b) not implemented!\n";
            return 0.;
        }

        virtual FloatD splatting_kernel_3D(Vec3Diff a, Vec3Diff b) const {
            std::cerr << "splatting_kernel_3D(Vec3Diff a, Vec3Diff b) not implemented!\n";
            return 0.0;
        }
        virtual FloatDD splatting_kernel_3D(Vec3DD a, Vec3DD b) const {
            std::cerr << "splatting_kernel_3D(Vec3DD a, Vec3DD b) not implemented!\n";
            return 0.0;
        }
        virtual double splatting_kernel_3D(Vec3d a, Vec3d b) const {
            std::cerr << "splatting_kernel_3D(Vec3d a, Vec3d b) not implemented!\n";
            return 0.0;
        }

        virtual FloatD splatting_kernel_3D_isotropic(Vec3Diff a, Vec3Diff b) const {
            std::cerr << "splatting_kernel_3D_isotropic(Vec3Diff a, Vec3Diff b) not implemented!\n";
            return 0.0;
        }
        virtual FloatDD splatting_kernel_3D_isotropic(Vec3DD a, Vec3DD b) const {
            std::cerr << "splatting_kernel_3D_isotropic(Vec3DD a, Vec3DD b) not implemented!\n";
            return 0.0;
        }
        virtual double splatting_kernel_3D_isotropic(Vec3d a, Vec3d b) const {
            std::cerr << "splatting_kernel_3D_isotropic(Vec3d a, Vec3d b) not implemented!\n";
            return 0.0;
        }

        virtual FloatDD splatting_kernel_1D(FloatDD a, FloatDD b) const {
            std::cerr << "splatting_kernel_1D(FloatDD a, FloatDD b) not implemented!\n";
            return 0.0;
        }
        virtual Dual splatting_kernel_1D(Dual a, Dual b) const {
            std::cerr << "splatting_kernel_1D(Dual a, Dual b) not implemented!\n";
            return 0.0;
        }
        virtual double splatting_kernel_1D(double a, double b) const {
            std::cerr << "splatting_kernel_1D(double a, double b) not implemented!\n";
            return 0.0;
        }

        virtual FloatD dsplat_da_3D(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const;
        virtual FloatD dsplat_db_3D(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const;
        virtual FloatDD dsplat2_dadb_3D(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const;

        virtual FloatD dsplat_da_3D_isotropic(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const;
        virtual FloatD dsplat_db_3D_isotropic(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const;
        virtual FloatDD dsplat2_dadb_3D_isotropic(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const;

        virtual FloatD dsplat_da_1D(Dual a, Dual b) const;
        virtual FloatD dsplat_db_1D(Dual a, Dual b) const;
        virtual FloatDD dsplat2_dadb_1D(FloatDD a, FloatDD b) const;

        double splatting_kernel_3D(Derivative a, Derivative b, Vec3d pa, Vec3d pb, Vec3d gradDirA, Vec3d gradDirB) const {
            if (a == Derivative::None && b == Derivative::None) {
                return splatting_kernel_3D(pa, pb);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (double)dsplat_da_3D(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirA));
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (double)dsplat_db_3D(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirB));
            }
            else {
                return (double)dsplat2_dadb_3D(vec_conv<Vec3DD>(pa), vec_conv<Vec3DD>(pb), vec_conv<Eigen::Array3d>(gradDirA), vec_conv<Eigen::Array3d>(gradDirB));
            }
        }

        double splatting_kernel_3D_isotropic(Derivative a, Derivative b, Vec3d pa, Vec3d pb, Vec3d gradDirA, Vec3d gradDirB) const {
            if (a == Derivative::None && b == Derivative::None) {
                return splatting_kernel_3D_isotropic(pa, pb);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (double)dsplat_da_3D_isotropic(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirA));
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (double)dsplat_db_3D_isotropic(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirB));
            }
            else {
                return (double)dsplat2_dadb_3D_isotropic(vec_conv<Vec3DD>(pa), vec_conv<Vec3DD>(pb), vec_conv<Eigen::Array3d>(gradDirA), vec_conv<Eigen::Array3d>(gradDirB));
            }
        }

        double splatting_kernel_1D(Derivative a, Derivative b, double pa, double pb) const {
            if (a == Derivative::None && b == Derivative::None) {
                return splatting_kernel_1D(pa, pb);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (double)dsplat_da_1D(Dual(pa), Dual(pb));
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (double)dsplat_db_1D(Dual(pa), Dual(pb));
            }
            else {
                return (double)dsplat2_dadb_1D(Dual(pa), Dual(pb));
            }
        }
    };

    class StationaryCovariance : public CovarianceFunction {
    public:

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("localScale", _kernelScale);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ CovarianceFunction::toJson(allocator), allocator,
                               "localScale", _kernelScale
            };
        }

        virtual bool isStationaryKernel() const override { return true; }
        virtual float nonStationarySplattingKernelScale(const Vec3f& p) const override { return 1.f; }
        virtual float worldSamplingSpatialScale() const override { return 1.f; }
        virtual float sparseConvNoiseLateralScale(const Vec3f& p) const override;
        virtual float sparseConvNoiseAmplitude(const Vec3f& p) const override;
        virtual float sparseConvNoiseVariance3D(const Vec3f& p, float impulseDensity, float kernelRadius, bool isIdentity, float localScale) const override;
        virtual float sparseConvNoiseVariance1D(const Vec3f& p, float impulseDensity, float kernelRadius) const override;
        virtual Eigen::Matrix3f worldToLocalMatrix(const Vec3f& p) const override;
        virtual Eigen::Matrix3f localToWorldMatrix(const Vec3f& p) const override;
        virtual Eigen::Matrix3f worldToLocalInvTransposeMatrix(const Vec3f& p) const override;
        virtual Eigen::Matrix3f localToWorldInvTransposeMatrix(const Vec3f& p) const override;

    private:
        virtual FloatD cov(FloatD absq) const = 0;
        virtual FloatDD cov(FloatDD absq) const = 0;
        virtual double cov(double absq) const = 0;

        virtual float splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
            std::cerr << "splattingKernel3DVal(Vec3f ab) not implemented!\n";
            return 0.;
        }
        virtual Vec3f splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
            std::cerr << "splattingKernel3D1stGrad(Vec3f ab) not implemented!\n";
            return Vec3f(0.);
        }

        virtual Eigen::Matrix3f splattingKernel3D2ndGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
            std::cerr << "splattingKernel3D2ndGrad(Vec3f ab) not implemented!\n";
            return Eigen::Matrix3f::Zero(1, 1);
        }

        virtual float splattingKernel1DVal(float ab, float localScale) const {
            std::cerr << "splattingKernel1DVal(float ab) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel1DVal(float ab, float localScale) const {
            std::cerr << "covarianceKernel1DVal(float ab) not implemented!\n";
            return 0.;
        }

        virtual float splattingKernel1D1stGrad(float ab, float localScale) const {
            std::cerr << "splattingKernel1D1stGrad(float ab) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel1D1stGrad(float ab, float localScale) const {
            std::cerr << "covarianceKernel1D1stGrad(float ab) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel2D2ndGrad(float ab, float localScale) const {
            std::cerr << "covarianceKernel2D2ndGrad(float ab) not implemented!\n";
            return 0.;
        }

        virtual float covarianceKernel2D2ndGradFor3DNormal(float ab, float localScale) const {
            std::cerr << "covarianceKernel2D2ndGradFor3DNormal(float ab) not implemented!\n";
            return 0.;
        }

        virtual FloatD splatting_kernel_3D(FloatD absq) const {
            std::cerr << "splatting_kernel_3D(FloatD absq) not implemented!\n";
            return 0.0;
        }
        virtual FloatDD splatting_kernel_3D(FloatDD absq) const {
            std::cerr << "splatting_kernel_3D(FloatDD absq) not implemented!\n";
            return 0.0;
        }
        virtual double splatting_kernel_3D(double absq) const {
            std::cerr << "splatting_kernel_3D(double absq) not implemented!\n";
            return 0.0;
        }

        virtual FloatD splatting_kernel_3D_isotropic(FloatD absq) const {
            std::cerr << "splatting_kernel_3D_isotropic(FloatD absq) not implemented!\n";
            return 0.0;
        }
        virtual FloatDD splatting_kernel_3D_isotropic(FloatDD absq) const {
            std::cerr << "splatting_kernel_3D_isotropic(FloatDD absq) not implemented!\n";
            return 0.0;
        }
        virtual double splatting_kernel_3D_isotropic(double absq) const {
            std::cerr << "splatting_kernel_3D_isotropic(double absq) not implemented!\n";
            return 0.0;
        }

        virtual FloatDD splatting_kernel_1D(FloatDD absq) const {
            std::cerr << "splatting_kernel_1D(FloatDD absq) not implemented!\n";
            return 0.0;
        }
        virtual Dual splatting_kernel_1D(Dual absq) const {
            std::cerr << "splatting_kernel_1D(Dual absq) not implemented!\n";
            return 0.0;
        }
        virtual double splatting_kernel_1D(double absq) const {
            std::cerr << "splatting_kernel_1D(double absq) not implemented!\n";
            return 0.0;
        }

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            FloatD absq = dist2(a, b, _aniso);
            return cov(absq);
        }

        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override {
            FloatDD absq = dist2(a, b, _aniso);
            return cov(absq);
        }

        virtual double cov(Vec3d a, Vec3d b) const override {
            double absq = dist2(a, b, _aniso);
            return cov(absq);
        }

        virtual float splattingKernel3DVal(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override {
            return splattingKernel3DVal(a-b, isCov, isIsotropic, globalScale, localScale, aniso_inv);
        }
        virtual Vec3f splattingKernel3D1stGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override {
            return splattingKernel3D1stGrad(a-b, isCov, isIsotropic, globalScale, localScale, aniso_inv);
        }

        virtual Eigen::Matrix3f splattingKernel3D2ndGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override {
            return splattingKernel3D2ndGrad(a-b, isCov, isIsotropic, globalScale, localScale, aniso_inv);
        }

        virtual float splattingKernel1DVal(float pQuery, float pCenter, float) const override {
            return splattingKernel1DVal(pQuery-pCenter, 1.0);
        }

        virtual float covarianceKernel1DVal(float pQuery, float pCenter, const Vec3f&, const Vec3f&, const Vec3f&) const override {
            return covarianceKernel1DVal(pQuery-pCenter, 1.0);
        }

        virtual float splattingKernel1D1stGrad(float pQuery, float pCenter, float) const override {
            return splattingKernel1D1stGrad(pQuery-pCenter, 1.0);
        }

        virtual float covarianceKernel1D1stGrad(float pQuery, float pCenter, const Vec3f&, const Vec3f&, const Vec3f&) const override {
            return covarianceKernel1D1stGrad(pQuery-pCenter, 1.0);
        }

        virtual float covarianceKernel2D2ndGrad(float pQuery, float pCenter, const Vec3f&, const Vec3f&, const Vec3f&) const override {
            return covarianceKernel2D2ndGrad(pQuery-pCenter, 1.0);
        }

        virtual float covarianceKernel2D2ndGradFor3DNormal(float pQuery, float pCenter, const Vec3f&, const Vec3f&, const Vec3f&) const override {
            return covarianceKernel2D2ndGradFor3DNormal(pQuery-pCenter, 1.0);
        }

        virtual FloatD splatting_kernel_3D(Vec3Diff a, Vec3Diff b) const override {
            FloatD absq = dist2(a, b);
            return splatting_kernel_3D(absq);
        }

        virtual FloatDD splatting_kernel_3D(Vec3DD a, Vec3DD b) const override {
            FloatDD absq = dist2(a, b);
            return splatting_kernel_3D(absq);
        }

        virtual double splatting_kernel_3D(Vec3d a, Vec3d b) const override {
            double absq = dist2(a, b);
            return splatting_kernel_3D(absq);
        }

        virtual FloatD splatting_kernel_3D_isotropic(Vec3Diff a, Vec3Diff b) const override {
            FloatD absq = dist2(a, b);
            return splatting_kernel_3D_isotropic(absq);
        }

        virtual FloatDD splatting_kernel_3D_isotropic(Vec3DD a, Vec3DD b) const override {
            FloatDD absq = dist2(a, b);
            return splatting_kernel_3D_isotropic(absq);
        }

        virtual double splatting_kernel_3D_isotropic(Vec3d a, Vec3d b) const override {
            double absq = dist2(a, b);
            return splatting_kernel_3D_isotropic(absq);
        }

        virtual FloatDD splatting_kernel_1D(FloatDD a, FloatDD b) const override {
            FloatDD absq = (a - b) * (a - b);
            return splatting_kernel_1D(absq);
        }

        virtual Dual splatting_kernel_1D(Dual a, Dual b) const override {
            Dual absq = (a - b) * (a - b);
            return splatting_kernel_1D(absq);
        }

        virtual double splatting_kernel_1D(double a, double b) const override {
            double absq = (a - b) * (a - b);
            return splatting_kernel_1D(absq);
        }

        virtual float sparseConvNoiseAmplitude() const {
            std::cerr << "sparseConvNoiseAmplitude() not implemented!\n";
            return 0.;
        }

        virtual float sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const {
            std::cerr << "sparseConvNoiseVariance3D() not implemented!\n";
            return 0.;
        }

        virtual float sparseConvNoiseVariance1D(float impulseDensity, float kernelRadius, float localScale) const {
            std::cerr << "sparseConvNoiseVariance1D() not implemented!\n";
            return 0.;
        }

        Vec3f transformPosDirWorldtoLocal(const Vec3f& posOrDir, const float localScale) const;

        Vec3f transformPosDirLocaltoWorld(const Vec3f& posOrDir, const float localScale) const;

        Vec3f transformGradWorldtoLocal(const Vec3f& posOrDir, const float localScale) const;

        Vec3f transformGradLocaltoWorld(const Vec3f& posOrDir, const float localScale) const;

        Eigen::Matrix3f worldToLocalMatrix() const;

        Eigen::Matrix3f localToWorldMatrix() const;

        Eigen::Matrix3f worldToLocalInvTransposeMatrix() const;

        Eigen::Matrix3f localToWorldInvTransposeMatrix() const;

        friend class SquaredExponentialCovariance;
        friend class MaternCovariance;
        friend class GaborAnisotropicCovariance;
        friend class GaborIsotropicCovariance;
        friend class NonstationaryCovariance;
        friend class GridNonstationaryCovariance;
        friend class MeanGradNonstationaryCovariance;
        friend class ProceduralNonstationaryCovariance;

        Eigen::Matrix3f _world_to_local, _local_to_world;
        Eigen::Matrix3f _world_to_local_transpose, _local_to_world_transpose;
        Vec3f _dominantEigenvector = Vec3f(0.f, 0.f, 1.f);
        float _kernelScale = 3.0;

        Vec3f dominantEigenvector() const { return _dominantEigenvector; }
    };

    class DotProductCovariance : public CovarianceFunction {
    public:

        DotProductCovariance(float sigma = 1.f, float l = 1., float p = 3., Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _p(p) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("lengthScale", _l);
            value.getField("p", _p);
            value.getField("aniso", _aniso);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                               "type", "dot_product",
                               "sigma", _sigma,
                               "lengthScale", _l,
                               "p", _p,
                               "aniso", _aniso
            };
        }


        virtual std::string id() const {
            return tinyformat::format("dp/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-p=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _p);
        }

    private:
        float _sigma, _l, _p;

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            auto d = a.dot(Vec3Diff{ _aniso.x(), _aniso.y(), _aniso.z() }.cwiseProduct(b));
            return pow(sqr(_sigma) + d / sqr(_l), _p);
        }

        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override {
            auto d = a.dot(Vec3DD{ _aniso.x(), _aniso.y(), _aniso.z() }.cwiseProduct(b));
            return pow(sqr(_sigma) + d / sqr(_l), _p);
        }

        virtual double cov(Vec3d a, Vec3d b) const override {
            auto d = a.dot(Vec3d{ _aniso.x(), _aniso.y(), _aniso.z() }.cwiseProduct(b));
            return pow(sqr(_sigma) + d / sqr(_l), _p);
        }
    };

    class SquaredExponentialCovariance : public StationaryCovariance {
    public:

        SquaredExponentialCovariance(float sigma = 1.f, float l = 1., Vec3f aniso = Vec3f(1.f), float localScale = 3.) : _sigma(sigma), _l(l) {
            _l_conv = _l * sqrt(2.f) / 2;
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;

        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        virtual std::string id() const;

        virtual bool hasAnalyticSpectralDensity() const override { return true; }

        virtual double spectral_density(double s) const {
            double norm = 1.0 / (sqrt(PI / 2) * sqr(_sigma));
            return norm * (exp(-0.5 * _l * _l * s * s) * _sigma * _sigma) / sqrt(1. / (_l * _l));
        }

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            return abs(rand_normal_2(sampler).x() / _l);
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double rad = (sqrt(2) * sqrt(-log(sampler.next1D()))) / _l;
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) *  rad;
        }

        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            Vec3d gaussian_3d = vec_conv<Vec3d>(sample_standard_normal(3, sampler));
            return gaussian_3d / _l * Vec3d(sqrt(_aniso.x()), sqrt(_aniso.y()), sqrt(_aniso.z()));
        }

        virtual float splattingKernelRadius(bool isIdentity, float localScale) const override;

        virtual float sparseConvNoiseAmplitude() const override;

        virtual Eigen::Matrix3f sparseConvNoiseOneOverSecondDerivative(const Vec3f& p_world, std::shared_ptr<Eigen::Matrix3f> aniso_inv, bool isIsotropic) const override;

        virtual float sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const override;

        virtual float sparseConvNoiseVariance1D(float impulseDensity, float kernelRadius, float localScale) const override;

        virtual Vec3f maxCorrelationDirection() const override {return _dominantEigenvector;}

    private:
        float _sigma, _l, _l_conv;
        Vec3f _l_aniso, _l_aniso_inv;
        bool _use_aniso_mtx = false;
        Eigen::Matrix3f _aniso_mtx, _cov_mtx, _cov_mtx_inv;
        float _cov_mtx_inv_determinant;
        Vec3f _dominantEigenvector;

        virtual FloatD cov(FloatD absq) const override;

        virtual FloatDD cov(FloatDD absq) const override;

        virtual double cov(double absq) const override;

        Eigen::Matrix3f getInvCovMtx(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const;

        virtual float splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Vec3f splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Eigen::Matrix3f splattingKernel3D2ndGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual float splattingKernel1DVal(float ab, float localScale) const override;

        virtual float covarianceKernel1DVal(float ab, float localScale) const override;

        virtual float splattingKernel1D1stGrad(float ab, float localScale) const override;

        virtual float covarianceKernel1D1stGrad(float ab, float localScale) const override;

        virtual float covarianceKernel2D2ndGrad(float ab, float localScale) const override;

        virtual float covarianceKernel2D2ndGradFor3DNormal(float ab, float localScale) const override;

        virtual FloatD splatting_kernel_3D(FloatD absq) const override {
            return exp(-(absq / sqr(_l)));
        }

        virtual FloatDD splatting_kernel_3D(FloatDD absq) const override {
            return exp(-(absq / sqr(_l)));
        }

        virtual double splatting_kernel_3D(double absq) const override {
            return exp(-(absq / sqr(_l)));
        }

        virtual FloatD splatting_kernel_3D_isotropic(FloatD absq) const override {
            return exp(-(absq / 2.0));
        }

        virtual FloatDD splatting_kernel_3D_isotropic(FloatDD absq) const override {
            return exp(-(absq / 2.0));
        }

        virtual double splatting_kernel_3D_isotropic(double absq) const override {
            return exp(-(absq / 2.0));
        }

        virtual FloatDD splatting_kernel_1D(FloatDD absq) const override {
            return exp(-(absq / 2.0));
        }

        virtual Dual splatting_kernel_1D(Dual absq) const override {
            return exp(-(absq / 2.0));
        }

        virtual double splatting_kernel_1D(double absq) const override {
            return exp(-(absq / 2.0));
        }
    };

    class RationalQuadraticCovariance : public StationaryCovariance {
    public:

        RationalQuadraticCovariance(float sigma = 1.f, float l = 1., float a = 1.0f, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _a(a) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            StationaryCovariance::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("a", _a);
            value.getField("lengthScale", _l);
            value.getField("aniso", _aniso);
        }

        virtual std::string id() const {
            return tinyformat::format("rq/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-a=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _a);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
                               "type", "rational_quadratic",
                               "sigma", _sigma,
                               "a", _a,
                               "lengthScale", _l,
                               "aniso", _aniso
            };
        }

        virtual bool hasAnalyticSpectralDensity() const override { return true; }

        virtual double spectral_density(double s) const {
            double norm = 1.0 / (sqrt(PI / 2.) * sqr(_sigma));
            return norm * (pow(2., 5. / 4. - _a / 2) * pow(1 / (_a * _l * _l), -(1. / 4.) - _a / 2)
                            * _sigma * _sigma * pow(abs(s), -0.5 + _a) *
                            std::cyl_bessel_k(0.5 - _a, (sqrt(2) * abs(s)) / sqrt(1. / (_a * _l * _l))))
                            / std::tgamma(_a);
        }

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double tau = rand_gamma(_a, 1 / (_l * _l), sampler);
            double l = 1 / sqrt(tau);
            return abs(rand_normal_2(sampler).x() / l);
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double rad = 2 * sqrt(rand_gamma(_a, 0.5 / (_l * _l), sampler)) * sqrt(-log(sampler.next1D()));
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * rad;
        }

        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double tau = 2 * sqrt(rand_gamma(_a, 0.5 / (_l * _l), sampler));
            Vec3d normal = vec_conv<Vec3d>(sample_standard_normal(3, sampler));
            double rad = sqrt(2) * 1/tau * normal.length();
            return vec_conv<Vec3d>(SampleWarp::uniformSphere(sampler.next2D())) * rad;
        }

    private:
        float _sigma, _a, _l;

        virtual FloatD cov(FloatD absq) const override {
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }

        virtual FloatDD cov(FloatDD absq) const override {
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }

        virtual double cov(double absq) const override {
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }
    };

    class MaternCovariance : public StationaryCovariance {
    public:

        MaternCovariance(float sigma = 1.f, float l = 1., float v = 1.0f, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _v(v) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;

        virtual std::string id() const;

        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        virtual bool hasAnalyticSpectralDensity() const override { return true; }

        virtual double spectral_density(double s) const {
            const int D = 1;
            return pow(2, D) * pow(PI, D / 2.) * std::lgamma(_v + D / 2.) *
                   pow(2 * _v, _v) / (std::lgamma(_v) * pow(_l, 2 * _v)) * pow(2 * _v / sqr(_l) + 4 * sqr(PI) * sqr(s), -(_v + D / 2.));
        }

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            return sqrt(-_v + _v * pow(1 - sampler.next1D(), -1. / _v)) / (sqrt(2) * _l * PI) * sin(PI * sampler.next1D());
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double r = sqrt(-_v + _v * pow(1 - sampler.next1D(), -1. / _v)) / (sqrt(2) * _l * PI);
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * r;
        }

        virtual float splattingKernelRadius(bool isIdentity, float localScale) const override;

        virtual float sparseConvNoiseAmplitude() const override;

        virtual float sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const override;

//        virtual float sparseConvNoiseVariance1D(float impulseDensity, float kernelRadius, float localScale) const override;

    private:
        float _sigma, _v, _l;
        Vec3f _aniso_sqrt;
        Vec3f _l_aniso_sqrt, _l_aniso_sqrt_inv;
        Vec3f _l_aniso_sqrt_filtered_zero, _l_aniso_sqrt_filtered_one;

        FloatD bessel_k(double n, FloatD x) const;

        FloatDD bessel_k(double n, FloatDD x) const;

        virtual FloatD cov(FloatD absq) const override;

        virtual FloatDD cov(FloatDD absq) const override;

        virtual double cov(double absq) const override;

        double cov(Vec3d pa, Vec3d pb) const;

        double dcov_da(Vec3d pa, Vec3d pb, Vec3d dir) const;

        double dcov2_dadb(Vec3d pa, Vec3d pb, Vec3d dira, Vec3d dirb) const;

        virtual double operator()(Derivative a, Derivative b, Vec3d pa, Vec3d pb, Vec3d gradDirA, Vec3d gradDirB) const override;

        virtual float splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Vec3f splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;
    };

    class GaborAnisotropicCovariance : public StationaryCovariance {
    public:

        GaborAnisotropicCovariance(float sigma = 1.f, float a = 1., float f = 1.0f, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _a(a), _f(f) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;

        virtual std::string id() const;

        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        virtual bool hasAnalyticSpectralDensity() const override {
            return false;
        }

        virtual float splattingKernelRadius(bool isIdentity, float localScale) const override;

        virtual float sparseConvNoiseAmplitude() const override;

        virtual float sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const override;

    private:
        float _sigma, _a, _f;
        Vec3f _omega;

        virtual FloatD cov(FloatD absq) const override;

        virtual FloatDD cov(FloatDD absq) const override;

        virtual double cov(double absq) const override;

        virtual float splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Vec3f splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;
    };

    class GaborIsotropicCovariance : public StationaryCovariance {
    public:

        GaborIsotropicCovariance(float sigma = 1.f, float a = 1., float f = 1.0f) : _sigma(sigma), _a(a), _f(f) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;

        virtual std::string id() const;

        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        virtual bool hasAnalyticSpectralDensity() const override {
            return false;
        }

        virtual float splattingKernelRadius(bool isIdentity, float localScale) const override;

        virtual float sparseConvNoiseAmplitude() const override;

        virtual float sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const override;

    private:
        float _sigma, _a, _f;

        virtual FloatD cov(FloatD absq) const override;

        virtual FloatDD cov(FloatDD absq) const override;

        virtual double cov(double absq) const override;

        virtual float splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Vec3f splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;
    };

    class PeriodicCovariance : public StationaryCovariance {
    public:

        PeriodicCovariance(float sigma = 1.f, float l = 1., float w = TWO_PI, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _w(w) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            StationaryCovariance::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("w", _w);
            value.getField("lengthScale", _l);
            value.getField("aniso", _aniso);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
                               "type", "periodic",
                               "sigma", _sigma,
                               "w", _w,
                               "lengthScale", _l,
                               "aniso", _aniso
            };
        }

        virtual std::string id() const {
            return tinyformat::format("per/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-w=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _w);
        }

    private:
        float _sigma, _w, _l;

        virtual FloatD cov(FloatD absq) const override {
            auto per = sin(PI * sqrt(absq) * _w);
            return sqr(_sigma) * exp(-2 * per * per / sqr(_l));
        }

        virtual FloatDD cov(FloatDD absq) const override {
            FloatDD ab = sqrt(absq + FLT_EPSILON);
            FloatDD per = sin(PI * ab * _w);
            return sqr(_sigma) * exp(-2 * per * per / sqr(_l)) ;
        }

        virtual double cov(double absq) const override {
            auto per = sin(PI * sqrt(absq) * _w);
            return sqr(_sigma) * exp(-2 * per * per / sqr(_l)) ;
        }
    };

    class ThinPlateCovariance : public StationaryCovariance {
    public:

        ThinPlateCovariance(float sigma = 1.f, float R = 1., Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _R(R) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            StationaryCovariance::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("R", _R);
            value.getField("aniso", _aniso);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
                               "type", "thin_plate",
                               "sigma", _sigma,
                               "R", _R,
                               "aniso", _aniso
            };
        }

        virtual std::string id() const {
            return tinyformat::format("tp/aniso=[%.4f,%.4f,%.4f]-s=%.3f-R=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _R);
        }

    private:
        float _sigma, _R;

        virtual FloatD cov(FloatD absq) const override {
            auto ab = sqrt(absq + FLT_EPSILON);
            return sqr(_sigma) * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }

        virtual FloatDD cov(FloatDD absq) const override {
            auto ab = sqrt(absq + FLT_EPSILON);
            return sqr(_sigma) * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }

        virtual double cov(double absq) const override {
            auto ab = sqrt(absq + FLT_EPSILON);
            return sqr(_sigma) * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }
    };

    class NonstationaryCovariance : public CovarianceFunction {
    public:
        NonstationaryCovariance(std::shared_ptr<StationaryCovariance> stationaryCov = nullptr) : _stationaryCov(stationaryCov), _multiResolutionGrid(false) { }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            if (auto cov = value["cov"]) {
                _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
            }
            value.getField("multiResolutionGrid", _multiResolutionGrid);
        }
        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator};
        }
        virtual void loadResources() override {};

        virtual std::string id() const {
            return tinyformat::format("ns-%s", _stationaryCov->id());
        }
        virtual Eigen::Matrix3d getCovMtx(const Vec3d p) const {
            std::cerr << "getAniso(Vec3d p) not implemented!\n";
            return Eigen::Matrix3d(1, 1);
        }
        virtual float getKernelScale(const Vec3f& p) const {
            std::cerr << "getKernelScale(const Vec3f& p) not implemented!\n";
            return 1.f;
        }
        virtual float sparseConvNoiseMaxLateralScale() const { return 1.f;}

        virtual float sparseConvNoiseMaxAnisotropyScale() const { return 1.f;}

        virtual float sparseConvNoiseLateralScale(const Vec3f& p) const override;

        virtual float nonStationarySplattingKernelScale(const Vec3f& p) const override;

        virtual float worldSamplingSpatialScale() const override;

        virtual float sparseConvNoiseAmplitude(const Vec3f& p) const override;

        virtual Eigen::Matrix3f sparseConvNoiseOneOverSecondDerivative(const Vec3f& p, std::shared_ptr<Eigen::Matrix3f>, bool isIsotropic) const override;

        virtual float splattingKernelRadius(bool isIdentity, float localScale) const override;

        virtual float sparseConvNoiseVariance3D(const Vec3f& p, float impulseDensity, float kernelRadius, bool isIdentity, float localScale) const override;

        virtual float sparseConvNoiseVariance1D(const Vec3f& p, float impulseDensity, float kernelRadius) const override;

        virtual Vec3f transformPosDirWorldtoLocal(const Vec3f& posOrDir, const float localScale) const override;

        virtual Vec3f transformPosDirLocaltoWorld(const Vec3f& posOrDir, const float localScale) const override;

        virtual Vec3f transformGradWorldtoLocal(const Vec3f& posOrDir, const float localScale) const override;

        virtual Vec3f transformGradLocaltoWorld(const Vec3f& posOrDir, const float localScale) const override;

        virtual Eigen::Matrix3f localToWorldInvTransposeMatrix(const Vec3f& p) const override;

        virtual bool useMultiResolutionGrid() const { return _multiResolutionGrid; }

    protected:
        std::shared_ptr<StationaryCovariance> _stationaryCov;
        bool _multiResolutionGrid;

    private:
        virtual std::tuple<Eigen::MatrixXd, double> computeAnisoFactor(Vec3d a, Vec3d b) const {return {Eigen::Matrix3d(1, 1), 1.f};}

        virtual float splattingKernel3DVal(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Vec3f splattingKernel3D1stGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual Eigen::Matrix3f splattingKernel3D2ndGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const override;

        virtual float splattingKernel1DVal(float pQuery, float pCenter, float localScale) const override;

        virtual float covarianceKernel1DVal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirection) const override;

        virtual float splattingKernel1D1stGrad(float pQuery, float pCenter, float localScale) const override;

        virtual float covarianceKernel1D1stGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirection) const override;

        virtual float covarianceKernel2D2ndGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirection) const override;

        virtual float covarianceKernel2D2ndGradFor3DNormal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const override;
    };

    class GridNonstationaryCovariance : public NonstationaryCovariance {
    public:

        GridNonstationaryCovariance(
                std::shared_ptr<StationaryCovariance> stationaryCov = nullptr,
                std::shared_ptr<Grid> variance = nullptr,
                std::shared_ptr<Grid> aniso = nullptr,
        float offset = 0, float scale = 1) : NonstationaryCovariance(stationaryCov), _variance(variance), _aniso(aniso), _offset(offset), _scale(scale)
        {
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual std::string id() const {
            return tinyformat::format("mean-ns-%s", _stationaryCov->id());
        }

        virtual double getVariance(const Vec3d p) const override;
        virtual float getUnscaledVariance(const Vec3d p) const override;
        virtual Eigen::Matrix3d getCovMtx(const Vec3d p) const override;
        virtual float getKernelScale(const Vec3f& p) const override;
        virtual float sparseConvNoiseMaxLateralScale() const override;
        virtual Vec3f maxCorrelationDirection() const override {return Vec3f(0.0f, 0.0f, 1.0f);}

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        FloatD sampleGrid(Vec3Diff a) const;
        FloatDD sampleGrid(Vec3DD a) const;

        std::shared_ptr<Grid> _variance;
        std::shared_ptr<Grid> _aniso;
        float _offset = 0;
        float _scale = 1;
        // Tricky way to assign different amplitude
        bool _surfVolAmpSeparate = false;
        float _surfVolAmpThresh = 1.f;
        float _surfAmpScale = 1.f;
        float _volAmpScale = 1.f;
        float _surfLsScale = 1.f;
        float _volLsScale = 1.f;
    };

    class MeanGradNonstationaryCovariance : public NonstationaryCovariance {
    public:

        MeanGradNonstationaryCovariance(
                std::shared_ptr<StationaryCovariance> stationaryCov = nullptr,
                std::shared_ptr<MeanFunction> mean = nullptr) : _stationaryCov(stationaryCov), _mean(mean)
        {
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual std::string id() const {
            return tinyformat::format("mean-ns-%s", _stationaryCov->id());
        }

        Eigen::Matrix3d localAniso(Vec3d p) const;

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        std::shared_ptr<StationaryCovariance> _stationaryCov;
        std::shared_ptr<MeanFunction> _mean;
    };

    class ProceduralNonstationaryCovariance : public NonstationaryCovariance {
    public:
        ProceduralNonstationaryCovariance(
                const std::shared_ptr<StationaryCovariance> stationaryCov = nullptr,
                const std::shared_ptr<ProceduralScalar>  variance = nullptr,
                const std::shared_ptr<ProceduralVector>  ls = nullptr,
                const std::shared_ptr<ProceduralScalar>  aniso = nullptr) : NonstationaryCovariance(stationaryCov), _variance(variance), _ls(ls), _anisoField(aniso), _anisotropyOnAxis(1.5) { }

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override;

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override;

        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override;

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;
        virtual std::string id() const;

        virtual double getVariance(const Vec3d p) const override;
        Eigen::Matrix3d getAnisoRootForSpectral(const Vec3d p) const;
        virtual Eigen::Matrix3d getCovMtx(const Vec3d p) const override;
        virtual void getNonstationaryAniso3D(const Vec3f& p, std::shared_ptr<Eigen::Matrix3f>& aniso) const override;
        virtual float getNonstationaryAniso1D(const Vec3f& p, const Vec3f& dirWorld) const override;
        virtual Eigen::Matrix3f getNonstationaryCovSplatCov3D(const Vec3f& pQueryWorld, const Vec3f& pCenterWorld) const override;
        virtual float getNonstationaryCovSplatCov1D(const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& dirLocal) const override;
        virtual float getKernelScale(const Vec3f& p) const override;
        virtual float sparseConvNoiseMaxLateralScale() const override;
        virtual float sparseConvNoiseMaxAnisotropyScale() const override;

        virtual bool isNonstationaryAnisotropicKernel() const override {
            if (_anisoField) return true;
            return false;
        }
    private:
        virtual std::tuple<Eigen::MatrixXd, double> computeAnisoFactor(Vec3d a, Vec3d b) const override;
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        std::shared_ptr<ProceduralScalar> _variance;
        std::shared_ptr<ProceduralVector> _ls;
        std::shared_ptr<ProceduralScalar> _anisoField;
        float _anisotropyOnAxis;
    };

    class NeuralNonstationaryCovariance : public NonstationaryCovariance {
    public:

        NeuralNonstationaryCovariance(
                std::shared_ptr<GPNeuralNetwork> nn = nullptr,
                float offset = 0, float scale = 1) : _nn(nn), _offset(offset), _scale(scale)
        {
        }

        virtual bool requireProjection() const { return true; }
        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual std::string id() const {
            return tinyformat::format("nn-ns");
        }

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        PathPtr _path;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        std::shared_ptr<GPNeuralNetwork> _nn;
        float _offset = 0;
        float _scale = 1;
    };

}

#endif //GPFUNCTIONS_HPP_