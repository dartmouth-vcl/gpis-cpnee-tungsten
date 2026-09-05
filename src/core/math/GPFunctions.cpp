#include "GPFunctions.hpp"
#include <math/Mat4f.hpp>
#include <math/MathUtil.hpp>
#include <math/Angle.hpp>

#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <Eigen/SparseQR>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/IterativeLinearSolvers>
#include <igl/signed_distance.h>

#include <ccomplex>
#include <fftw3.h>
#include <random>

namespace Tungsten {

    template <typename Mat, typename Vec>
    inline Mat compute_ansio_full(const float& angle, const Vec& aniso) {
        Mat vmat = Mat::Identity();
        // Set the 2x2 rotation block in the top-left corner
        vmat(0, 0) = std::cos(angle);
        vmat(0, 1) = std::sin(angle);
        vmat(1, 0) = -std::sin(angle);
        vmat(1, 1) = std::cos(angle);

        Mat smat = Mat::Identity();
        smat.diagonal() = aniso;
        auto mat = smat * vmat;
        return mat.transpose() * mat;
    }

    template <typename Mat, typename Vec>
    inline Mat compute_ansio_simplified(const Vec& grad, const Vec& aniso) {
        TangentFrameD<Mat, Vec> tf(grad);

        auto vmat = tf.toMatrix();
        Mat smat = Mat::Identity();
        smat.diagonal() = aniso;

        return vmat * smat * vmat.transpose();
    }

    double ProceduralNoise::operator()(const Vec3d p) const {
        double min = _min + _const;
        double max = _max + _const;
        switch (type) {
            case NoiseType::BottomTop:
                return sqrt(exp(lerp(log(min * min),  log(max * max), clamp(p.y() * _scale + _offset, 0.0, 1.0)))) - _const;
            case NoiseType::LeftRight:
                return sqrt(exp(lerp(log(min * min), log(max * max), clamp(p.x() * _scale + _offset, 0.0, 1.0)))) - _const;
            case NoiseType::FrontBack:
                return sqrt(exp(lerp(log(min * min), log(max * max), clamp(p.z() * _scale + _offset, 0.0, 1.0)))) - _const;
            case NoiseType::BottomTopLeftRight: {
                double min2 = _min_2 + _const;
                double max2 = _max_2 + _const;
                float bottomTop = sqrt(exp(lerp(log(min * min),  log(max * max), clamp(p.y() * _scale + _offset, 0.0, 1.0))));
                float leftRight = sqrt(exp(lerp(log(min2 * min2),  log(max2 * max2), clamp(p.x() * _scale_2 + _offset_2, 0.0, 1.0))));
                return bottomTop * leftRight - _const * _const;
            }
            case NoiseType::Sandstone:
            {
                Vec3d p_scaled = p * 0.3;
                double f = fbm(p_scaled + fbm(p_scaled + fbm(p_scaled, 2), 2), 2);
                Vec3d col = Vec3d(f * 1.9, f * 0.7, f * 0.25);
                col = std::sqrt(col*1.2) - 0.35;
                return lerp(_min, _max, clamp(col.x(), 0., 1.));
            }
            case NoiseType::Rust:
            {
                Vec3d p_scaled = p * 2;
                double f = smoothStep(0.4, 0.6, fbm(p_scaled + fbm(p_scaled*.1, 2)*0.4, 2) - fbm(p_scaled*25., 2)*0.1);
                return lerp(_min, _max, clamp(f, 0., 1.));
            }
        }
    }

    Vec3d ProceduralNoiseVec::operator()(const Vec3d p) const {
        double min = _min + _const;
        double max = _max + _const;
        switch (type) {
            case NoiseType::BottomTop:
                return Vec3d(sqrt(exp(lerp(log(min * min),  log(max * max), clamp(p.y() * _scale + _offset, 0.0, 1.0))))) - _const;
            case NoiseType::LeftRight:
                return Vec3d(sqrt(exp(lerp(log(min * min), log(max * max), clamp(p.x() * _scale + _offset, 0.0, 1.0))))) - _const;
            case NoiseType::FrontBack:
                return Vec3d(sqrt(exp(lerp(log(min * min), log(max * max), clamp(p.z() * _scale + _offset, 0.0, 1.0))))) - _const;
            case NoiseType::BottomTopLeftRight: {
                double min2 = _min_2 + _const;
                double max2 = _max_2 + _const;
                float bottomTop = sqrt(exp(lerp(log(min * min),  log(max * max), clamp(p.y() * _scale + _offset, 0.0, 1.0))));
                float leftRight = sqrt(exp(lerp(log(min2 * min2),  log(max2 * max2), clamp(p.x() * _scale_2 + _offset_2, 0.0, 1.0))));
                return Vec3d(bottomTop * leftRight - _const * _const);
            }
            case NoiseType::Sandstone:
            {
                Vec3d p_scaled = p * 0.3;
                double f = fbm(p_scaled + fbm(p_scaled + fbm(p_scaled, 10), 10), 10);
                Vec3d col = Vec3d(f * 1.9, f * 0.7, f * 0.25);
                col = std::sqrt(col*1.2) - 0.35;
                return  clamp(col * 0.2, Vec3d(0.), Vec3d(1.));
            }
            case NoiseType::Rust:
            {
                Vec3d p_scaled = p * 2;
                double f = smoothStep(0.4, 0.6, fbm(p_scaled + fbm(p_scaled*.1, 10)*0.4, 10) + fbm(p_scaled*25., 10)*0.1);
                return lerp(Vec3d(0.278,0.212,0.141), Vec3d(1.), f);
            }
        }
    }

float ProceduralNoiseVec::maxVal() const {
        switch (type) {
            case NoiseType::BottomTop:
            case NoiseType::LeftRight:
            case NoiseType::FrontBack:
                return max(_max, _min);
            case NoiseType::BottomTopLeftRight:
                return max(_max, _min) * max(_max_2, _min_2);
            case NoiseType::Sandstone:
            {
                return 1.f;
            }
            case NoiseType::Rust:
            {
                return 1.f;
            }
        }
    }




void TabulatedMean::fromJson(JsonPtr value, const Scene& scene) {
    MeanFunction::fromJson(value, scene);

    if (auto grid = value["grid"]) {
        _grid = scene.fetchGrid(grid);
    }

    value.getField("offset", _offset);
    value.getField("scale", _scale);
    value.getField("volume", _isVolume);
}

rapidjson::Value TabulatedMean::toJson(Allocator& allocator) const {
    return JsonObject{ MeanFunction::toJson(allocator), allocator,
                       "type", "tabulated",
                       "grid", *_grid,
                       "offset", _offset,
                       "scale", _scale,
                       "volume", _isVolume
    };
}

double TabulatedMean::mean(Vec3d a) const {
    Vec3f p = _grid->invNaturalTransform() * vec_conv<Vec3f>(a);
    double res = _grid->density(p);
    if(_isVolume) {
        res = -log(max(0.0001, res));
    }
    return (res + _offset) * _scale;
}

Vec3d TabulatedMean::emission(Vec3d a) const {
    Vec3f p = _grid->invNaturalTransform() * vec_conv<Vec3f>(a);
    return Vec3d(_grid->emission(p));
}

Vec3d TabulatedMean::dmean_da(Vec3d a) const {
    double eps = 0.001;
    double vals[] = {
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps)),
            mean(a)
    };

    auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[3], vals[2] - vals[3]) / eps;
    return grad;
}

void TabulatedMean::loadResources() {
    _grid->loadResources();
}


void NeuralMean::fromJson(JsonPtr value, const Scene& scene) {
    MeanFunction::fromJson(value, scene);

    if (auto path = value["network"]) _path = scene.fetchResource(path);

    value.getField("offset", _offset);
    value.getField("scale", _scale);
    value.getField("transform", _configTransform);

    _invConfigTransform = _configTransform.invert();
}

rapidjson::Value NeuralMean::toJson(Allocator& allocator) const {
    return JsonObject{ MeanFunction::toJson(allocator), allocator,
                       "type", "neural",
                       "network", *_path,
                       "offset", _offset,
                       "scale", _scale,
                       "transform", _configTransform
    };
}


double NeuralMean::mean(Vec3d a) const {
    return (_nn->mean(mult(_invConfigTransform, a)) + _offset) * _scale;
}

Vec3d NeuralMean::dmean_da(Vec3d a) const {
    double eps = 0.001;
    double vals[] = {
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps)),
            mean(a)
    };

    auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[3], vals[2] - vals[3]) / eps;
    return grad;
}

void NeuralMean::loadResources() {
    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(*_path);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
    }

    _nn = std::make_shared<GPNeuralNetwork>();
    _nn->read(*document, _path->absolute().parent());
}

void ProceduralMean::fromJson(JsonPtr value, const Scene& scene) {
    MeanFunction::fromJson(value, scene);
    value.getField("transform", _configTransform);
    _invConfigTransform = _configTransform.invert();

    std::string fnString = "knob";
    if (value.getField("func", fnString)) {
        _f = std::make_shared<ProceduralSdf>(SdfFunctions::stringToFunction(fnString));
    }
    else if(auto f = value["f"]) {
        _f = scene.fetchProceduralScalar(f);
    }

    value.getField("min", _min);
    value.getField("scale", _scale);
    value.getField("offset", _offset);
}

rapidjson::Value ProceduralMean::toJson(Allocator& allocator) const {
    return  JsonObject{ MeanFunction::toJson(allocator), allocator,
                        "type", "procedural",
                        "f", *_f,
                        "transform", _configTransform,
                        "min", _min,
                        "scale", _scale,
                        "scale", _offset,
    };
}

double ProceduralMean::mean(Vec3d a) const {
    auto p = vec_conv<Vec3f>(a);
    p = _invConfigTransform.transformPoint(p);
    float m = (*_f)(Vec3d(p));
    m *= _scale;
    return max(_min, m + _offset);
}

Vec3d ProceduralMean::dmean_da(Vec3d a) const {
    double eps = 0.001f;
    double vals[] = {
            mean(a),
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps))
    };
    return Vec3d(vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]) / eps;
}

void MeshSdfMean::fromJson(JsonPtr value, const Scene& scene) {
    MeanFunction::fromJson(value, scene);
    if (auto path = value["file"]) _path = scene.fetchResource(path);
    value.getField("transform", _configTransform);
    value.getField("signed", _signed);
    value.getField("min", _min);
    _invConfigTransform = _configTransform.invert();
}

rapidjson::Value MeshSdfMean::toJson(Allocator& allocator) const {
    return JsonObject{ MeanFunction::toJson(allocator), allocator,
                       "type", "mesh",
                       "file", *_path,
                       "transform", _configTransform,
                       "signed", _signed,
                       "min", _min,
    };
}

double MeshSdfMean::mean(Vec3d a) const {
    // perform a closest point query
    Eigen::MatrixXd V_vis(1, 3);
    V_vis(0, 0) = a.x();
    V_vis(1, 0) = a.y();
    V_vis(2, 0) = a.z();

    Eigen::VectorXd S_vis;
    igl::signed_distance_fast_winding_number(V_vis, V, F, tree, fwn_bvh, S_vis);

    return max((double)S_vis(0), _min);
}

Vec3d MeshSdfMean::color(Vec3d a) const {
    Eigen::RowVector3d P = vec_conv<Eigen::RowVector3d>(a);

    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::RowVector3d closest_point;
    tree.squared_distance(V, F, P, sqrD, I, closest_point);

    Eigen::VectorXi closestFace = F.row(I[0]);

    Eigen::RowVector3d
            vA = V.row(closestFace[0]),
            vB = V.row(closestFace[1]),
            vC = V.row(closestFace[2]);

    Eigen::RowVector3d L;
    igl::barycentric_coordinates(
            closest_point,
            vA, vB, vC,
            L);

    Vec3f colA = _colors[closestFace[0]];
    Vec3f colB = _colors[closestFace[1]];
    Vec3f colC = _colors[closestFace[2]];

    Vec3d result = vec_conv<Vec3d>(colA * L[0] + colB * L[1] + colC * L[2]);

    if (std::isinf(result) || std::isnan(result) || (colA + colB + colC).sum() < 0.05f) {
        return vec_conv<Vec3d>(colA + colB + colC) / 3;
    }

    return result;
}

Vec3d MeshSdfMean::shell_embedding(Vec3d a) const {
    Eigen::RowVector3d P = vec_conv<Eigen::RowVector3d>(a);

    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::RowVector3d closest_point;
    tree.squared_distance(V, F, P, sqrD, I, closest_point);

    Eigen::VectorXi closestFace = F.row(I[0]);

    Eigen::RowVector3d
            vA = V.row(closestFace[0]),
            vB = V.row(closestFace[1]),
            vC = V.row(closestFace[2]);

    Eigen::RowVector3d L;
    igl::barycentric_coordinates(
            closest_point,
            vA, vB, vC,
            L);

    Vec2f uvA = _uvs[closestFace[0]];
    Vec2f uvB = _uvs[closestFace[1]];
    Vec2f uvC = _uvs[closestFace[2]];

    Vec2d uv = Vec2d(uvA * L[0] + uvB * L[1] + uvC * L[2]) * _bounds.diagonal().length();

    double w = igl::fast_winding_number(fwn_bvh, 2, P);
    //0.5 is on surface
    double dist = sqrt(sqrD(0)) * (1. - 2. * std::abs(w));

    return Vec3d(uv.x(), uv.y(), dist);
}

Vec3d MeshSdfMean::dmean_da(Vec3d a) const {
    double eps = 0.001f;
    double vals[] = {
            mean(a),
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps))
    };

    return Vec3d(vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]) / eps;
}

void MeshSdfMean::loadResources() {

    std::vector<Vertex> _verts;
    std::vector<TriangleI> _tris;

    _bounds = Box3d();

    if (_path && MeshIO::load(*_path, _verts, _tris)) {

        V.resize(_verts.size(), 3);
        _colors.resize(_verts.size());
        _uvs.resize(_verts.size());

        for (int i = 0; i < _verts.size(); i++) {
            Vec3f tpos = _configTransform * _verts[i].pos();
            V(i, 0) = tpos.x();
            V(i, 1) = tpos.y();
            V(i, 2) = tpos.z();

            _colors[i] = _verts[i].color();

            if (_colors[i].sum() < 0.001) {
                _colors[i] = Vec3f(0.215684f, 0.262744f, 0.031373f);
            }

            _uvs[i] = _verts[i].uv();

            //Vec3f tnorm = _configTransform.transformVector(_verts[i].normal());

            _bounds.grow(vec_conv<Vec3d>(tpos));
        }

        F.resize(_tris.size(), 3);

        // specify the triangle indices
        for (int i = 0; i < _tris.size(); i++) {
            F(i, 0) = _tris[i].v0;
            F(i, 1) = _tris[i].v1;
            F(i, 2) = _tris[i].v2;
        }

        tree.init(V, F);
        igl::fast_winding_number(V, F, 2, fwn_bvh);
    }
    else {
        FAIL("Failed to load mesh.");
    }
}




void CovarianceFunction::loadResources() {}

double CovarianceFunction::spectral_density(double s) const {
    double max_t = 10;
    double dt = max_t / pow(2, 12);
    double max_w = PI / dt;

    double bin_c = s / max_w * discreteSpectralDensity.size();
    size_t bin = clamp(size_t(bin_c), size_t(0), discreteSpectralDensity.size() - 1);
    size_t n_bin = clamp(size_t(bin_c) + 1, size_t(0), discreteSpectralDensity.size() - 1);

    double bin_frac = bin_c - bin;

    return lerp(discreteSpectralDensity[bin], discreteSpectralDensity[n_bin], bin_frac);
}

double CovarianceFunction::sample_spectral_density(PathSampleGenerator& sampler, Vec3d p) const {
    return 0;
}

Vec2d CovarianceFunction::sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p) const {
    return Vec2d(0.);
}

Vec3d CovarianceFunction::sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p) const {
    return Vec3d(0.);
}

FloatD CovarianceFunction::dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const {
    Eigen::Array3d zd = Eigen::Array3d::Zero();
    auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a, b); }, autodiff::along(dirA, zd), at(a, b));
    return covDiff[1];
}

FloatD CovarianceFunction::dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const {
    return dcov_da(b, a, dirB);
}

FloatDD CovarianceFunction::dcov2_dadb(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const {
    Eigen::Matrix3d hess = autodiff::hessian([&](auto a, auto b) { return cov(a, b); }, wrt(a, b), at(a, b)).block(3, 0, 3, 3);
    double res = dirA.transpose().matrix() * hess * dirB.matrix();
    return res;
}

FloatD CovarianceFunction::dsplat_da_3D(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const {
    Eigen::Array3d zd = Eigen::Array3d::Zero();
    auto splatDiff = autodiff::derivatives([&](auto a, auto b) { return splatting_kernel_3D(a, b); }, autodiff::along(dirA, zd), at(a, b));
    return splatDiff[1];
}

FloatD CovarianceFunction::dsplat_db_3D(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const {
    return dsplat_da_3D(b, a, dirB);
}

FloatDD CovarianceFunction::dsplat2_dadb_3D(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const {
    Eigen::Matrix3d hess = autodiff::hessian([&](auto a, auto b) { return splatting_kernel_3D(a, b); }, wrt(a, b), at(a, b)).block(3, 0, 3, 3);
    double res = dirA.transpose().matrix() * hess * dirB.matrix();
    return res;
}

FloatD CovarianceFunction::dsplat_da_3D_isotropic(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const {
    Eigen::Array3d zd = Eigen::Array3d::Zero();
    auto splatDiff = autodiff::derivatives([&](auto a, auto b) { return splatting_kernel_3D_isotropic(a, b); }, autodiff::along(dirA, zd), at(a, b));
    return splatDiff[1];
}

FloatD CovarianceFunction::dsplat_db_3D_isotropic(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const {
    return dsplat_da_3D_isotropic(b, a, dirB);
}

FloatDD CovarianceFunction::dsplat2_dadb_3D_isotropic(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const {
    Eigen::Matrix3d hess = autodiff::hessian([&](auto a, auto b) { return splatting_kernel_3D_isotropic(a, b); }, wrt(a, b), at(a, b)).block(3, 0, 3, 3);
    double res = dirA.transpose().matrix() * hess * dirB.matrix();
    return res;
}

FloatD CovarianceFunction::dsplat_da_1D(Dual a, Dual b) const {
    auto splatDiff = autodiff::derivative([&](auto a, auto b) { return splatting_kernel_1D(a, b); }, wrt(a), at(a, b));
    return splatDiff;
}

FloatD CovarianceFunction::dsplat_db_1D(Dual a, Dual b) const {
    return dsplat_da_1D(b, a);
}

FloatDD CovarianceFunction::dsplat2_dadb_1D(FloatDD a, FloatDD b) const {
    auto hess = autodiff::hessian([&](auto a, auto b) { return splatting_kernel_1D(a, b); }, wrt(a, b), at(a, b));
    return hess(0, 1);
}

Vec4f CovarianceFunction::splattingKernel3D(Vec3f pa, Vec3f pb, bool isCov, bool isIsotropic, float globalScale, const Vec3f& p_world) const {
    // Put the logic here to prevent computing aniso_inv twice
    float localScale = nonStationarySplattingKernelScale(p_world);
    std::shared_ptr<Eigen::Matrix3f> aniso_inv;
    getNonstationaryAniso3D(p_world, aniso_inv);
    if (aniso_inv)
        *aniso_inv = aniso_inv->inverse();
    float val = splattingKernel3DVal(pa, pb, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    Vec3f grad = splattingKernel3D1stGrad(pa, pb, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    return Vec4f(val, grad.x(), grad.y(), grad.z());
}

Vec4f CovarianceFunction::splattingKernel3DGrad(Vec3f pa, Vec3f pb, Vec3f coeff, bool isCov, bool isIsotropic, float globalScale, const Vec3f& p_world) const {
    float localScale = nonStationarySplattingKernelScale(p_world);
    std::shared_ptr<Eigen::Matrix3f> aniso_inv;
    getNonstationaryAniso3D(p_world, aniso_inv);
    if (aniso_inv)
        *aniso_inv = aniso_inv->inverse();
    Vec3f grad1st = splattingKernel3D1stGrad(pa, pb, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    Eigen::Matrix3f grad2nd = splattingKernel3D2ndGrad(pa, pb, isCov, isIsotropic, globalScale, localScale, aniso_inv);

    Vec4f vec_x = Vec4f(grad1st.x(), grad2nd(0, 0), grad2nd(0, 1), grad2nd(0, 2));
    Vec4f vec_y = Vec4f(grad1st.y(), grad2nd(1, 0), grad2nd(1, 1), grad2nd(1, 2));
    Vec4f vec_z = Vec4f(grad1st.z(), grad2nd(2, 0), grad2nd(2, 1), grad2nd(2, 2));
    return vec_x * coeff.x() + vec_y * coeff.y() + vec_z * coeff.z();
}

Vec2f CovarianceFunction::splattingKernel1D(float pQuery, float pCenter, const Vec3f& pCenterWorld, const Vec3f& rayDirectionWorld) const {
    float localScale = nonStationarySplattingKernelScale(pCenterWorld);
    localScale *= sqrt(getNonstationaryAniso1D(pCenterWorld, rayDirectionWorld));
    float val = splattingKernel1DVal(pQuery, pCenter, localScale);
    float grad = splattingKernel1D1stGrad(pQuery, pCenter, localScale);
    return Vec2f(val, grad);
}

Vec2f CovarianceFunction::covarianceKernel1D(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float val = covarianceKernel1DVal(pQuery, pCenter, pQueryWorld, pCenterWorld, rayDirectionLocal);
    float grad = covarianceKernel1D1stGrad(pQuery, pCenter, pQueryWorld, pCenterWorld, rayDirectionLocal);
    return Vec2f(val, grad);
}

Vec2f CovarianceFunction::covarianceKernel1DGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float grad1st = covarianceKernel1D1stGrad(pQuery, pCenter, pQueryWorld, pCenterWorld, rayDirectionLocal);
    float grad2nd = covarianceKernel2D2ndGrad(pQuery, pCenter, pQueryWorld, pCenterWorld, rayDirectionLocal);
    return Vec2f(grad1st, grad2nd);
}

float CovarianceFunction::covarianceKernel1DGradFor3DNormal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float grad2nd = covarianceKernel2D2ndGradFor3DNormal(pQuery, pCenter, pQueryWorld, pCenterWorld, rayDirectionLocal);
    return grad2nd;
}




float StationaryCovariance::sparseConvNoiseLateralScale(const Vec3f& p) const {
    return 1.0;
}

float StationaryCovariance::sparseConvNoiseAmplitude(const Vec3f& p) const {
    return sparseConvNoiseAmplitude();
}

float StationaryCovariance::sparseConvNoiseVariance3D(const Vec3f& p, float impulseDensity, float kernelRadius, bool isIdentity, float globalScale) const {
    return sparseConvNoiseVariance3D(impulseDensity, kernelRadius, isIdentity, globalScale, 1.0);
}

float StationaryCovariance::sparseConvNoiseVariance1D(const Vec3f& p, float impulseDensity, float kernelRadius) const {
    return sparseConvNoiseVariance1D(impulseDensity, kernelRadius, 1.0);
}

Eigen::Matrix3f StationaryCovariance::worldToLocalMatrix(const Vec3f& p) const { return worldToLocalMatrix(); }

Eigen::Matrix3f StationaryCovariance::localToWorldMatrix(const Vec3f& p) const { return localToWorldMatrix(); }

Eigen::Matrix3f StationaryCovariance::worldToLocalInvTransposeMatrix(const Vec3f& p) const { return worldToLocalInvTransposeMatrix(); }

Eigen::Matrix3f StationaryCovariance::localToWorldInvTransposeMatrix(const Vec3f& p) const { return localToWorldInvTransposeMatrix(); }

Vec3f StationaryCovariance::transformPosDirWorldtoLocal(const Vec3f& posOrDir, const float localScale) const {
    return to_vec3f(_world_to_local * to_eigen3f(posOrDir) / localScale);
}

Vec3f StationaryCovariance::transformPosDirLocaltoWorld(const Vec3f& posOrDir, const float localScale) const {
    return to_vec3f(_local_to_world * to_eigen3f(posOrDir) * localScale);
}

Vec3f StationaryCovariance::transformGradWorldtoLocal(const Vec3f& posOrDir, const float localScale) const { return to_vec3f(_local_to_world_transpose * to_eigen3f(posOrDir) * localScale); }

Vec3f StationaryCovariance::transformGradLocaltoWorld(const Vec3f& posOrDir, const float localScale) const { return to_vec3f(_world_to_local_transpose * to_eigen3f(posOrDir) / localScale); }

Eigen::Matrix3f StationaryCovariance::worldToLocalMatrix() const { return _world_to_local; }

Eigen::Matrix3f StationaryCovariance::localToWorldMatrix() const { return _local_to_world; }

Eigen::Matrix3f StationaryCovariance::worldToLocalInvTransposeMatrix() const { return _local_to_world_transpose; }

Eigen::Matrix3f StationaryCovariance::localToWorldInvTransposeMatrix() const { return _world_to_local_transpose; }




void SquaredExponentialCovariance::fromJson(JsonPtr value, const Scene& scene) {
    StationaryCovariance::fromJson(value, scene);
    value.getField("sigma", _sigma);
    value.getField("lengthScale", _l);
    _l_conv = _l * sqrt(2.f) / 2;
    value.getField("aniso", _aniso);
    value.getField("useAnisoMtx", _use_aniso_mtx);
    value.getField("localAniso", _localAniso);
    if (!_use_aniso_mtx) {
        _l_aniso = Vec3f(_l_conv) * _aniso;
        _l_aniso_inv = Vec3f(1.0) / _l_aniso;
        _l_aniso_inv = filterWithZero(_l_aniso_inv);
        _local_to_world.diagonal() << to_eigen3f(_l_aniso);
        _world_to_local.diagonal() << to_eigen3f(_l_aniso_inv);

        int maxAxis = _l_aniso.x() >= _l_aniso.y() ? (_l_aniso.x() >= _l_aniso.z() ? 0 : 2) : (_l_aniso.y() >= _l_aniso.z() ? 1 : 2);
        _dominantEigenvector = Vec3f(maxAxis == 0, maxAxis == 1, maxAxis == 2);
    }
    else {
        value.getField("anisoMtx", _aniso_mtx);
        _local_to_world = _l_conv * _aniso_mtx;
        _world_to_local = _local_to_world.inverse();
        _cov_mtx_inv = _world_to_local.transpose() * _world_to_local; // Σ^TΣ -> X^T(Σ^TΣ)X = (ΣX)^T(ΣX)
        // _cov_mtx =  _cov_mtx_inv.inverse(); // _local_to_world * _world_to_local.transpose()
        _cov_mtx_inv_determinant = _cov_mtx_inv.determinant();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(_cov_mtx_inv, Eigen::ComputeEigenvectors);
        _dominantEigenvector = to_vec3f(solver.eigenvectors().col(0));
        std::cout << _dominantEigenvector<< std::endl;
    }

    _local_to_world_transpose = _local_to_world.transpose();
    _world_to_local_transpose = _world_to_local.transpose();
}

rapidjson::Value SquaredExponentialCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
    "type", "squared_exponential",
    "sigma", _sigma,
    "lengthScale", _l,
    "aniso", _aniso,
    "useAnisoMtx", _use_aniso_mtx,
    "anisoMtx", _aniso_mtx,
    };
}

std::string SquaredExponentialCovariance::id() const {
    return tinyformat::format("se/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l);
}

float SquaredExponentialCovariance::splattingKernelRadius(bool isIdentity, float localScale) const {
    float scale_factor = _kernelScale;
    if (isIdentity)
        return scale_factor;
    float mtx_factor;
    if (!_use_aniso_mtx) {
        mtx_factor = max(max(_l_aniso.x(), _l_aniso.y()), _l_aniso.z());
    }
    else {
        auto end_pt = _local_to_world.col(0) + _local_to_world.col(1) + _local_to_world.col(2);
        mtx_factor = max(max(end_pt[0], end_pt[1]), end_pt[2]);
    }
    return scale_factor * localScale * mtx_factor;
}

float SquaredExponentialCovariance::sparseConvNoiseAmplitude() const { return _sigma; }

Eigen::Matrix3f SquaredExponentialCovariance::sparseConvNoiseOneOverSecondDerivative(const Vec3f& p_world, std::shared_ptr<Eigen::Matrix3f> aniso_inv, bool isIsotropic) const {
    Eigen::Matrix3f invCovMtx;
    if (aniso_inv) {
        *aniso_inv = aniso_inv->inverse();
        Eigen::Matrix3f invSigma = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invSigma.diagonal() << to_eigen3f(_l_aniso_inv);
            else
                invSigma = _world_to_local;
        }
        invCovMtx = invSigma.transpose() * (*aniso_inv) * invSigma;
    }
    else {
        invCovMtx = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invCovMtx.diagonal() << to_eigen3f(_l_aniso_inv.cwiseProduct(_l_aniso_inv));
            else
                invCovMtx = _cov_mtx_inv; // = _world_to_local.transpose() * _world_to_local;
        }
    }

    invCovMtx *= 0.25f; // 0.5f for covariance, 0.5f for the 1/2 in the squared exponential form
    Eigen::Matrix3f covKernelSecondGrad = - 2.f * invCovMtx;
    return covKernelSecondGrad.inverse();
}

float SquaredExponentialCovariance::sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const {
    double impulseDensityUnitArea = impulseDensity / (kernelRadius * kernelRadius * kernelRadius);
    double covDeterminantSqrt = 1.0;
    if (!isIdentity) {
        if (!_use_aniso_mtx)
            covDeterminantSqrt = _l_aniso.x() * _l_aniso.y() * _l_aniso.z();
        else
            covDeterminantSqrt = 1.0 / sqrt(_cov_mtx_inv_determinant);
        covDeterminantSqrt *= pow(globalScale, 3);
    }
    covDeterminantSqrt *= pow(localScale, 3);
    double integralKernelSquared = pow(M_PI, 1.5) * covDeterminantSqrt;
    return impulseDensityUnitArea * integralKernelSquared;
}

float SquaredExponentialCovariance::sparseConvNoiseVariance1D(float impulseDensity, float kernelRadius, float localScale) const {
    double impulseDensityUnitArea = impulseDensity / kernelRadius;
    double integralKernelSquared = sqrt(M_PI) * localScale;
    return impulseDensityUnitArea * integralKernelSquared;
}

FloatD SquaredExponentialCovariance::cov(FloatD absq) const {
    return sqr(_sigma) * exp(-absq / (2 * sqr(_l)));
}

FloatDD SquaredExponentialCovariance::cov(FloatDD absq) const {
    return sqr(_sigma) * exp(-absq / (2 * sqr(_l)));
}

double SquaredExponentialCovariance::cov(double absq) const {
    return sqr(_sigma) * exp(-absq / (2 * sqr(_l)));
}

Eigen::Matrix3f SquaredExponentialCovariance::getInvCovMtx(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    Eigen::Matrix3f invCovMtx;
    if (aniso_inv) {
        Eigen::Matrix3f invSigma = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invSigma.diagonal() << to_eigen3f(_l_aniso_inv);
            else
                invSigma = _world_to_local;
            invSigma /= globalScale;
        }
        invCovMtx = invSigma.transpose() * (*aniso_inv) * invSigma;
    }
    else {
        invCovMtx = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invCovMtx.diagonal() << to_eigen3f(_l_aniso_inv.cwiseProduct(_l_aniso_inv));
            else
                invCovMtx = _cov_mtx_inv; // = _world_to_local.transpose() * _world_to_local;
            invCovMtx /= sqr(globalScale);
        }
    }
    if (isCov)
        invCovMtx *= 0.5f;
    invCovMtx /= sqr(localScale);
    invCovMtx *= 0.5f;
    return invCovMtx;
}

float SquaredExponentialCovariance::splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    Eigen::Matrix3f invCovMtx = getInvCovMtx(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    float absq = dist2_ab(to_eigen3f(ab), invCovMtx);
    return exp(-absq);
}

Vec3f SquaredExponentialCovariance::splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    float f = splattingKernel3DVal(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    Eigen::Matrix3f invCovMtx = getInvCovMtx(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    float dfdx = -2.f * to_eigen3f(ab).dot(invCovMtx.col(0)) * f;
    float dfdy = -2.f * to_eigen3f(ab).dot(invCovMtx.col(1)) * f;
    float dfdz = -2.f * to_eigen3f(ab).dot(invCovMtx.col(2)) * f;
    return Vec3f(dfdx, dfdy, dfdz);
}

Eigen::Matrix3f SquaredExponentialCovariance::splattingKernel3D2ndGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    float f = splattingKernel3DVal(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    Eigen::Matrix3f invCovMtx = getInvCovMtx(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    float dot_product_x = -2.f * to_eigen3f(ab).dot(invCovMtx.col(0));
    float dot_product_y = -2.f * to_eigen3f(ab).dot(invCovMtx.col(1));
    float dot_product_z = -2.f * to_eigen3f(ab).dot(invCovMtx.col(2));
    Eigen::Matrix3f grad2nd = Eigen::Matrix3f::Zero(3, 3);
    grad2nd(0, 0) = dot_product_x * dot_product_x - 2.f * invCovMtx(0, 0);
    grad2nd(1, 1) = dot_product_y * dot_product_y - 2.f * invCovMtx(1, 1);
    grad2nd(2, 2) = dot_product_z * dot_product_z - 2.f * invCovMtx(2, 2);
    grad2nd(0, 1) = grad2nd(1, 0) = dot_product_x * dot_product_y - 2.f * invCovMtx(0, 1);
    grad2nd(0, 2) = grad2nd(2, 0) = dot_product_x * dot_product_z - 2.f * invCovMtx(0, 2);
    grad2nd(1, 2) = grad2nd(2, 1) = dot_product_y * dot_product_z - 2.f * invCovMtx(1, 2);
    return grad2nd * f;
}

float SquaredExponentialCovariance::splattingKernel1DVal(float ab, float localScale) const {
    float denominator = 2.0 * sqr(localScale);
    return exp(-(ab * ab / denominator));
}

float SquaredExponentialCovariance::covarianceKernel1DVal(float ab, float localScale) const {
    return splattingKernel1DVal(ab, localScale * sqrt(2));
}

float SquaredExponentialCovariance::splattingKernel1D1stGrad(float ab, float localScale) const {
    float f = splattingKernel1DVal(ab, localScale);
    float denominator = 2.0 * sqr(localScale);
    return -2.0 * ab / denominator * f;
}

float SquaredExponentialCovariance::covarianceKernel1D1stGrad(float ab, float localScale) const {
    return splattingKernel1D1stGrad(ab, localScale * sqrt(2));
}

float SquaredExponentialCovariance::covarianceKernel2D2ndGrad(float ab, float localScale) const {
    float f = covarianceKernel1DVal(ab, localScale);
    float denominator = 4.0 * sqr(localScale);
    float grad1st = -2.0 * ab / denominator;
    return (grad1st * grad1st - 2.0 / denominator) * f;
}

float SquaredExponentialCovariance::covarianceKernel2D2ndGradFor3DNormal(float ab, float localScale) const {
    float f = covarianceKernel1DVal(ab, localScale);
    float denominator = 4.0 * sqr(localScale);
    return - 2.0 / denominator * f;
}




void MaternCovariance::fromJson(JsonPtr value, const Scene& scene) {
    StationaryCovariance::fromJson(value, scene);
    value.getField("sigma", _sigma);
    value.getField("v", _v);
    value.getField("lengthScale", _l);
    value.getField("aniso", _aniso);

    _aniso_sqrt = Vec3f(sqrt(_aniso.x()), sqrt(_aniso.y()), sqrt(_aniso.z()));
    _l_aniso_sqrt = Vec3f(_l).cwiseDivision(_aniso_sqrt);
    _l_aniso_sqrt_filtered_zero = filterWithZero(_l_aniso_sqrt);
}

std::string MaternCovariance::id() const {
    return tinyformat::format("mat/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-v=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _v);
}

rapidjson::Value MaternCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
    "type", "matern",
    "sigma", _sigma,
    "v", _v,
    "lengthScale", _l,
    "aniso", _aniso
    };
}

FloatD MaternCovariance::bessel_k(double n, FloatD x) const {
    FloatD result;
    result[0] = boost::math::cyl_bessel_k(n, x.val());
    result[1] = x[1] * (-boost::math::cyl_bessel_k(n - 1, x.val()) - n / x.val() * boost::math::cyl_bessel_k(n, x.val()));
    return result;
}

FloatDD MaternCovariance::bessel_k(double n, FloatDD x) const {
    FloatDD result;
    result.val.val = boost::math::cyl_bessel_k(n, x.val.val);
    result.val.grad = x.val.val * (-boost::math::cyl_bessel_k(n - 1, x.val.val) - n / x.val.val * boost::math::cyl_bessel_k(n, x.val.val));
    return result;
}

FloatD MaternCovariance::cov(FloatD absq) const {
    if (absq < 1e-2) {
        return sqr(_sigma);
    }
    auto r_scl = sqrt(2 * _v * absq) / _l;
    return sqr(_sigma) * pow(2, 1 - _v) / std::lgamma(_v) * pow(r_scl, _v) * bessel_k(_v, r_scl);
}

FloatDD MaternCovariance::cov(FloatDD absq) const {
    if (absq < 1e-2) {
        return sqr(_sigma);
    }
    auto r_scl = sqrt(2 * _v * absq) / _l;
    return sqr(_sigma) * pow(2, 1 - _v) / std::lgamma(_v) * pow(r_scl, _v) * bessel_k(_v, r_scl);
}

double MaternCovariance::cov(double absq) const {
    if (absq < 1e-2) {
        return sqr(_sigma);
    }
    auto r_scl = sqrt(2. * _v * absq) / _l;
    return sqr(_sigma) * pow(2, 1. - _v) / std::lgamma(_v) * pow(r_scl, _v) * boost::math::cyl_bessel_k(_v, r_scl);
}

double MaternCovariance::cov(Vec3d pa, Vec3d pb) const {
    double absq = dist2(pa, pb, Vec3f(1.0));
    double abLen = sqrt(absq);
    if (_v == 0.5) {
        return sqr(_sigma) * exp(-abLen / _l);
    }
    else if (_v == 1.5) {
        return sqr(_sigma) * (1.0 + sqrt(3.0) * abLen / _l) * exp(-sqrt(3.0) * abLen / _l);
    }
    else if (_v == 2.5) {
        return sqr(_sigma) * (1.0 + sqrt(5.0) * abLen / _l + 5.0 * absq / (3.0 * sqr(_l))) * exp(-sqrt(5.0) * abLen / _l);
    }
    else {
        std::cerr << "Matern kernel only implemented for v = 0.5, 1.5, 2.5!\n";
        return 0.f;
    }
}

double MaternCovariance::dcov_da(Vec3d pa, Vec3d pb, Vec3d dir) const {
    double absq = dist2(pa, pb, Vec3f(1.0));
    double abLen = sqrt(absq);
     Vec3f ab = Vec3f(pb - pa);
    if (_v == 0.5)  {
        if (abLen == 0)
            return 0.f;
        return - sqr(_sigma) * exp(-abLen / _l) / _l * ab.dot(Vec3f(dir)) / abLen;
    }
    else if (_v == 1.5) {
        return - sqr(_sigma) * 3.0 * ab.dot(Vec3f(dir)) * exp(-sqrt(3.0) * abLen / _l) / sqr(_l);
    }
    else if (_v == 2.5) {
        return - sqr(_sigma) * 5.0 * ab.dot(Vec3f(dir)) * (_l + sqrt(5.0) * abLen) * exp(-sqrt(5.0) * abLen / _l) / (3 * pow(_l, 3));
    }
    else {
        std::cerr << "Matern kernel only implemented for v = 0.5, 1.5, 2.5!\n";
        return 0.f;
    }
}

double MaternCovariance::dcov2_dadb(Vec3d pa, Vec3d pb, Vec3d dira, Vec3d dirb) const {
    double absq = dist2(pa, pb, Vec3f(1.0));
    double abLen = sqrt(absq);
     Vec3f ab = Vec3f(pb - pa);
    if (_v == 0.5)  {
        // Note: gradient undefined at pa == pb, so function space sampling won't generate a "correct" appearance
        if (abLen < 1e-10)
            return 0.f;
        float term1 = ab.dot(Vec3f(dira)) * ab.dot(Vec3f(dirb)) * abLen;
        float term2 = (dira.x() * dirb.y() + dirb.x() * dira.y()) * ab.x() * ab.y() + (dira.x() * dirb.z() + dirb.x() * dira.z()) * ab.x() * ab.z() + (dira.y() * dirb.z() + dirb.y() * dira.z()) * ab.y() * ab.z();
        float term3 = dira.x() * dirb.x() * (sqr(ab.y()) + sqr(ab.z())) + dira.y() * dirb.y() * (sqr(ab.x()) + sqr(ab.z())) + dira.z() * dirb.z() * (sqr(ab.x()) + sqr(ab.y()));
        return sqr(_sigma) * exp(-abLen / _l) / (sqr(_l) * pow(abLen, 3)) * (term1 + _l * (term2 - term3));
    }
    else if (_v == 1.5) {
        if (abLen == 0) {
            return sqr(_sigma) * 3.0 * dira.dot(dirb) / sqr(_l);
        }
        float term1 = 3.0 * sqrt(3.0) * ab.dot(Vec3f(dira)) * ab.dot(Vec3f(dirb));
        float term2 = 3.0 * _l * dira.dot(dirb) * abLen;
        return sqr(_sigma) * exp(-sqrt(3.0) * abLen / _l) / (pow(_l, 3) * abLen) * (term2 - term1);
    }
    else if (_v == 2.5) {
        float term1 = dira.dot(dirb) * (sqr(_l) + _l * sqrt(5.0) * abLen);
        float term2 = 5.0 * ab.dot(Vec3f(dira)) * ab.dot(Vec3f(dirb));
        return sqr(_sigma) * 5.0 * exp(-sqrt(5.0) * abLen / _l) / (3.0 * pow(_l, 4)) * (term1 - term2);
    }
    else {
        std::cerr << "Matern kernel only implemented for v = 0.5, 1.5, 2.5!\n";
        return 0.f;
    }
}

double MaternCovariance::operator()(Derivative a, Derivative b, Vec3d pa, Vec3d pb, Vec3d gradDirA, Vec3d gradDirB) const {
    if (a == Derivative::None && b == Derivative::None) {
        return cov(pa, pb);
    }
    else if (a == Derivative::First && b == Derivative::None) {
        return dcov_da(pa, pb, gradDirA);
    }
    else if (a == Derivative::None && b == Derivative::First) {
        return dcov_da(pb, pa, gradDirB);
    }
    else {
        return dcov2_dadb(pa, pb, gradDirA, gradDirB);
    }
}

float MaternCovariance::splattingKernelRadius(bool isIdentity, float localScale) const {
    float scale_factor = _kernelScale;
    if (isIdentity)
        return scale_factor;
    return scale_factor * localScale * sqrt(2) / 2 * max(max(_l_aniso_sqrt_filtered_zero.x(), _l_aniso_sqrt_filtered_zero.y()), _l_aniso_sqrt_filtered_zero.z());
}

float MaternCovariance::sparseConvNoiseAmplitude() const { return _sigma; }

float MaternCovariance::sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const {
    double impulseDensityUnitArea = impulseDensity / (kernelRadius * kernelRadius * kernelRadius);
    double integralKernelSquared;
    if (_v == 0.5) {
        integralKernelSquared = 2.0 * M_PI * _l;
    }
    else if (_v == 1.5) {
        integralKernelSquared = pow(M_PI * _l, 3) / (24 * sqrt(3));
    }
    else if (_v == 2.5) {
        integralKernelSquared = M_PI * pow(_l, 3) / (5 * sqrt(5));
    }
    else {
        std::cerr << "Matern kernel only implemented for v = 0.5, 1.5, 2.5!\n";
        return 0.f;
    }
    return impulseDensityUnitArea * integralKernelSquared;
}

float MaternCovariance::splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f>) const {
    double abLen = ab.length();
    if (_v == 0.5) {
        return exp(-abLen / _l) / abLen;
    }
    else if (_v == 1.5) {
        auto r_scl = sqrt(3.) * abLen / _l;
        return boost::math::cyl_bessel_k(0, r_scl);
    }
    else if (_v == 2.5) {
        return exp(-sqrt(5.) * abLen / _l);
    }
    else {
        std::cerr << "Matern kernel only implemented for v = 0.5, 1.5, 2.5!\n";
        return 0.f;
    }
}

Vec3f MaternCovariance::splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f>) const {
    double abLen = ab.length();
    if (_v == 0.5) {
        return - (float)(exp(- abLen / _l) * (1 / pow(abLen, 3) - 1 /(pow(abLen, 2) * _l))) * ab;
    }
    else if (_v == 1.5) {
        auto r_scl = sqrt(3.) * abLen / _l;
        return -(float)(boost::math::cyl_bessel_k(1, r_scl) * sqrt(3.) / _l / abLen) * ab;
    }
    else if (_v == 2.5) {
        return -(float)(exp(-sqrt(5.) * abLen / _l) * sqrt(5.) / _l / abLen) * ab;
    }
    else {
        std::cerr << "Matern kernel only implemented for v = 0.5, 1.5, 2.5!\n";
        return Vec3f(0.f);
    }
}




void GaborAnisotropicCovariance::fromJson(JsonPtr value, const Scene& scene) {
    StationaryCovariance::fromJson(value, scene);
    value.getField("sigma", _sigma);
    value.getField("a_inv", _a);
    _a = 1.0 / _a;
    value.getField("f_inv", _f);
    _f = 1.0 / _f;
    value.getField("omega", _omega);
    _omega.normalize();
}

std::string GaborAnisotropicCovariance::id() const {
    return tinyformat::format("gaborAniso/omega=[%.4f,%.4f,%.4f]-s=%.3f-a=%.3f-f=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _a, _f);
}

rapidjson::Value GaborAnisotropicCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
                       "type", "gabor_aniso",
                       "sigma", _sigma,
                       "a_inv", 1.0 / _a,
                       "f_inv", 1.0 / _f,
                       "omega", _omega
    };
}

FloatD GaborAnisotropicCovariance::cov(FloatD absq) const {
    std::cerr << "GaborAnisotropicCovariance::cov() not implemented!\n";
    return 0.f;
}

FloatDD GaborAnisotropicCovariance::cov(FloatDD absq) const {
    std::cerr << "GaborAnisotropicCovariance::cov() not implemented!\n";
    return 0.f;
}

double GaborAnisotropicCovariance::cov(double absq) const {
    std::cerr << "GaborAnisotropicCovariance::cov() not implemented!\n";
    return 0.f;
}

float GaborAnisotropicCovariance::splattingKernelRadius(bool isIdentity, float localScale) const {
    float scale_factor = _kernelScale;
    return scale_factor * sqrt(2) / 2 * 1.0 / _a;
}

float GaborAnisotropicCovariance::sparseConvNoiseAmplitude() const { return _sigma; }

float GaborAnisotropicCovariance::sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const {
    double impulseDensityUnitArea = impulseDensity / (kernelRadius * kernelRadius * kernelRadius);
    double integralKernelSquared = pow(1.0 / _a, 3) * (1 + exp(- 2.0 * M_PI * sqr(_f / _a))) / (4 * sqrt(2));
    return impulseDensityUnitArea * integralKernelSquared;
}

float GaborAnisotropicCovariance::splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f>) const {
    return exp(- M_PI * sqr(_a) * ab.lengthSq()) * std::cos(2.f * M_PI * _f * _omega.dot(ab));
}

Vec3f GaborAnisotropicCovariance::splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f>) const {
    float A = exp(- M_PI * sqr(_a) * ab.lengthSq());
    float B = std::cos(2.f * M_PI * _f * _omega.dot(ab));
    Vec3f grad = - (float)(A * std::sin(2.f * M_PI * _f * _omega.dot(ab)) * 2.f * M_PI * _f) * _omega -
            (float)(B * A * 2.f * M_PI * sqr(_a)) * ab;
    return grad;
}




void GaborIsotropicCovariance::fromJson(JsonPtr value, const Scene& scene) {
    StationaryCovariance::fromJson(value, scene);
    value.getField("sigma", _sigma);
    value.getField("a_inv", _a);
    _a = 1.0 / _a;
    value.getField("f_inv", _f);
    _f = 1.0 / _f;
}

std::string GaborIsotropicCovariance::id() const {
    return tinyformat::format("gaborIso/s=%.3f-a=%.3f-f=%.3f", _sigma, _a, _f);
}

rapidjson::Value GaborIsotropicCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ StationaryCovariance::toJson(allocator), allocator,
                       "type", "gabor_iso",
                       "sigma", _sigma,
                       "a_inv", 1.0 / _a,
                       "f_inv", 1.0 / _f,
    };
}

FloatD GaborIsotropicCovariance::cov(FloatD absq) const {
    std::cerr << "GaborIsotropicCovariance::cov() not implemented!\n";
    return 0.f;
}

FloatDD GaborIsotropicCovariance::cov(FloatDD absq) const {
    std::cerr << "GaborIsotropicCovariance::cov() not implemented!\n";
    return 0.f;
}

double GaborIsotropicCovariance::cov(double absq) const {
    std::cerr << "GaborIsotropicCovariance::cov() not implemented!\n";
    return 0.f;
}

float GaborIsotropicCovariance::splattingKernelRadius(bool isIdentity, float localScale) const {
    float scale_factor = _kernelScale;
    return scale_factor * sqrt(2) / 4 * 1.0 / _a;
}

float GaborIsotropicCovariance::sparseConvNoiseAmplitude() const { return _sigma; }

float GaborIsotropicCovariance::sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const {
    double impulseDensityUnitArea = impulseDensity / (kernelRadius * kernelRadius * kernelRadius);
    double integralKernelSquared = 2 * sqrt(2) * M_PI * sqr(_f) / _a * (1 - exp(- 2 * M_PI * _f/ sqr(_a)));
    return impulseDensityUnitArea * integralKernelSquared;
}

float GaborIsotropicCovariance::splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f>) const {
    float r = ab.length();
    return exp(- M_PI * sqr(_a * r)) * 2 * _f / r * std::sin(2 * M_PI * _f * r);
}

Vec3f GaborIsotropicCovariance::splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f>) const {
    float r = ab.length();
    Vec3f grad = (float)(2 * _f * exp(- M_PI * sqr(_a) * ab.lengthSq()) * (- std::sin(2 * M_PI * _f * r) / pow(r, 3) - 2 * M_PI * sqr(_a) * std::sin(2 * M_PI * _f * r) / r + 2 * M_PI * _f * std::cos(2 * M_PI * _f * r) / sqr(r))) * ab;
    return grad;
}




float NonstationaryCovariance::sparseConvNoiseLateralScale(const Vec3f& p) const {
    return getKernelScale(p);
}

float NonstationaryCovariance::nonStationarySplattingKernelScale(const Vec3f& p) const {
    if (_multiResolutionGrid)
        return 1.;
    else {
        return sparseConvNoiseLateralScale(p) / sparseConvNoiseMaxLateralScale();
    }
}

float NonstationaryCovariance::worldSamplingSpatialScale() const {
    return sparseConvNoiseMaxLateralScale();
}

float NonstationaryCovariance::sparseConvNoiseAmplitude(const Vec3f& p) const {
    return getVariance(Vec3d(p)) * _stationaryCov->sparseConvNoiseAmplitude();
}

Eigen::Matrix3f NonstationaryCovariance::sparseConvNoiseOneOverSecondDerivative(const Vec3f& p_world, std::shared_ptr<Eigen::Matrix3f>, bool isIsotropic) const {
    std::shared_ptr<Eigen::Matrix3f> aniso;
    getNonstationaryAniso3D(p_world, aniso);
    return _stationaryCov->sparseConvNoiseOneOverSecondDerivative(p_world, aniso, isIsotropic);
}

float NonstationaryCovariance::splattingKernelRadius(bool isIdentity, float localScale) const {
    localScale *= _multiResolutionGrid ? 1.0 : sparseConvNoiseMaxLateralScale();
    localScale *= sparseConvNoiseMaxAnisotropyScale();
    return _stationaryCov->splattingKernelRadius(isIdentity, localScale);
}

float NonstationaryCovariance::sparseConvNoiseVariance3D(const Vec3f& p, float impulseDensity, float kernelRadius, bool isIdentity, float globalScale) const {
    float localScale = _multiResolutionGrid ? 1.0 : nonStationarySplattingKernelScale(p);
    return _stationaryCov->sparseConvNoiseVariance3D(impulseDensity, kernelRadius, isIdentity, globalScale, localScale);
}

float NonstationaryCovariance::sparseConvNoiseVariance1D(const Vec3f& p, float impulseDensity, float kernelRadius) const {
    float localScale = nonStationarySplattingKernelScale(p);
    return _stationaryCov->sparseConvNoiseVariance1D(impulseDensity, kernelRadius, localScale);
}

Vec3f NonstationaryCovariance::transformPosDirWorldtoLocal(const Vec3f& posOrDir, const float globalScale) const {
    float scale = _multiResolutionGrid ? globalScale : sparseConvNoiseMaxLateralScale();
    return _stationaryCov->transformPosDirWorldtoLocal(posOrDir, scale);
}

Vec3f NonstationaryCovariance::transformPosDirLocaltoWorld(const Vec3f& posOrDir, const float globalScale) const {
    float scale = _multiResolutionGrid ? globalScale : sparseConvNoiseMaxLateralScale();
    return _stationaryCov->transformPosDirLocaltoWorld(posOrDir, scale);
}

Vec3f NonstationaryCovariance::transformGradWorldtoLocal(const Vec3f& posOrDir, const float globalScale) const {
    float scale = _multiResolutionGrid ? globalScale : sparseConvNoiseMaxLateralScale();
    return _stationaryCov->transformGradWorldtoLocal(posOrDir, scale);
}

Vec3f NonstationaryCovariance::transformGradLocaltoWorld(const Vec3f& posOrDir, const float globalScale) const {
    float scale = _multiResolutionGrid ? globalScale : sparseConvNoiseMaxLateralScale();
    return _stationaryCov->transformGradLocaltoWorld(posOrDir, scale);
}

Eigen::Matrix3f NonstationaryCovariance::localToWorldInvTransposeMatrix(const Vec3f& p) const {
    float scale = _multiResolutionGrid ? 1.0 : sparseConvNoiseMaxLateralScale();
    return _stationaryCov->localToWorldInvTransposeMatrix(p) / scale;
}

float NonstationaryCovariance::splattingKernel3DVal(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    return _stationaryCov->splattingKernel3DVal(a-b, isCov, isIsotropic, globalScale, localScale, aniso_inv);
}

Vec3f NonstationaryCovariance::splattingKernel3D1stGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    return _stationaryCov->splattingKernel3D1stGrad(a-b, isCov, isIsotropic, globalScale, localScale, aniso_inv);
}

Eigen::Matrix3f NonstationaryCovariance::splattingKernel3D2ndGrad(Vec3f a, Vec3f b, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    return _stationaryCov->splattingKernel3D2ndGrad(a-b, isCov, isIsotropic, globalScale, localScale, aniso_inv);
}

float NonstationaryCovariance::splattingKernel1DVal(float pQuery, float pCenter, float localScale) const {
    return _stationaryCov->splattingKernel1DVal(pQuery - pCenter, localScale);
}

float NonstationaryCovariance::covarianceKernel1DVal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float d = pQuery - pCenter;
    float anisoABavgLocal = getNonstationaryCovSplatCov1D(pQueryWorld, pCenterWorld, rayDirectionLocal);
    return
        _stationaryCov->covarianceKernel1DVal(d, anisoABavgLocal);
}

float NonstationaryCovariance::splattingKernel1D1stGrad(float pQuery, float pCenter, float localScale) const {
    return _stationaryCov->splattingKernel1D1stGrad(pQuery - pCenter, localScale);
}

float NonstationaryCovariance::covarianceKernel1D1stGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float d = pQuery - pCenter;
    float anisoABavgLocal = getNonstationaryCovSplatCov1D(pQueryWorld, pCenterWorld, rayDirectionLocal);
    return
        _stationaryCov->covarianceKernel1D1stGrad(d, anisoABavgLocal);
}

float NonstationaryCovariance::covarianceKernel2D2ndGrad(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float d = pQuery - pCenter;
    float anisoABavgLocal = getNonstationaryCovSplatCov1D(pQueryWorld, pCenterWorld, rayDirectionLocal);
    return
        _stationaryCov->covarianceKernel2D2ndGrad(d, anisoABavgLocal);
}

float NonstationaryCovariance::covarianceKernel2D2ndGradFor3DNormal(float pQuery, float pCenter, const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& rayDirectionLocal) const {
    float d = pQuery - pCenter;
    float anisoABavgLocal = getNonstationaryCovSplatCov1D(pQueryWorld, pCenterWorld, rayDirectionLocal);
    return
        _stationaryCov->covarianceKernel2D2ndGradFor3DNormal(d, anisoABavgLocal);
}





void GridNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
    NonstationaryCovariance::fromJson(value, scene);

    if (auto variance = value["grid"]) {
        _variance = scene.fetchGrid(variance);
    }
    else if (auto variance = value["variance"]) {
        _variance = scene.fetchGrid(variance);
    }

    if (auto aniso = value["ansio"]) {
        _aniso = scene.fetchGrid(aniso);
    }

    value.getField("offset", _offset);
    value.getField("scale", _scale);
    value.getField("surf_vol_amp_separate", _surfVolAmpSeparate);
    value.getField("surf_vol_amp_thresh", _surfVolAmpThresh);
    value.getField("surf_amp_scale", _surfAmpScale);
    value.getField("vol_amp_scale", _volAmpScale);
    value.getField("surf_ls_scale", _surfLsScale);
    value.getField("vol_ls_scale", _volLsScale);}

rapidjson::Value GridNonstationaryCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ NonstationaryCovariance::toJson(allocator), allocator,
                       "type", "nonstationary",
                       "cov", *_stationaryCov,
                       "variance", *_variance,
                       "offset", _offset,
                       "scale", _scale,
                       "surf_vol_amp_separate", _surfVolAmpSeparate,
                       "surf_vol_amp_thresh", _surfVolAmpThresh,
                       "surf_amp_scale", _surfAmpScale,
                       "vol_amp_scale", _volAmpScale,
                       "surf_ls_scale", _surfLsScale,
                       "vol_ls_scale", _volLsScale
    };
}

void GridNonstationaryCovariance::loadResources() {
    _variance->loadResources();
    _stationaryCov->loadResources();

    if (_aniso) {
        _aniso->loadResources();
    }
}

float GridNonstationaryCovariance::getUnscaledVariance(const Vec3d p) const {
    if (_variance) {
        return (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(p)) + _offset) * _scale;
    }
    return 1.f;
}

double GridNonstationaryCovariance::getVariance(const Vec3d p) const {
    float amplitude = getUnscaledVariance(p);
    if (!_surfVolAmpSeparate)
        return amplitude;
    if (amplitude < _surfVolAmpThresh) {
        return amplitude * _surfAmpScale;
    }
    else {
        return amplitude * _volAmpScale;
    }
}

Eigen::Matrix3d GridNonstationaryCovariance::getCovMtx(const Vec3d p) const {
    return Eigen::Matrix3d::Identity();
}

float GridNonstationaryCovariance::getKernelScale(const Vec3f& p) const {
    if(_surfVolAmpSeparate) {
        float amplitude = getUnscaledVariance(Vec3d(p));
        if (amplitude < _surfVolAmpThresh) {
            return _surfLsScale;
        }
        else {
            return _volLsScale;
        }
    }
    return 1.f;
}

float GridNonstationaryCovariance::sparseConvNoiseMaxLateralScale() const {
    if(_surfVolAmpSeparate) {
        return max(_surfLsScale, _volLsScale);
    }
    return 1.f;
}

FloatD GridNonstationaryCovariance::sampleGrid(Vec3Diff a) const {
    FloatD result = 0;
    Vec3f ap = vec_conv<Vec3f>(from_diff(a));
    result[0] = _variance->density(ap);
    return result;

    /**/
    float eps = 0.001f;
    float vals[] = {
            _variance->density(ap + Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap + Vec3f(0.f, eps, 0.f)),
            _variance->density(ap + Vec3f(0.f, 0.f, eps)),
            _variance->density(ap - Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap - Vec3f(0.f, eps, 0.f)),
            _variance->density(ap - Vec3f(0.f, 0.f, eps))
    };
    auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);

    result[1] = grad.dot({ (float)a.x()[1], (float)a.y()[1] , (float)a.z()[1] });
    result[2] = 0; // linear interp
    return result;
}

FloatDD GridNonstationaryCovariance::sampleGrid(Vec3DD a) const {
    FloatDD result = 0;
    Vec3f ap = vec_conv<Vec3f>(from_diff(a));
    result.val = _variance->density(ap);
    return result;

    /**/
    float eps = 0.001f;
    float vals[] = {
            _variance->density(ap + Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap + Vec3f(0.f, eps, 0.f)),
            _variance->density(ap + Vec3f(0.f, 0.f, eps)),
            _variance->density(ap - Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap - Vec3f(0.f, eps, 0.f)),
            _variance->density(ap - Vec3f(0.f, 0.f, eps))
    };
    auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);

    result.grad.val = grad.dot({ a.x().grad.val, a.y().grad.val , a.z().grad.val });
    result.grad.grad = 0; // linear interp
    return result;
}

FloatD GridNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {

    FloatD sigmaA = (sampleGrid(mult(_variance->invNaturalTransform(), a)) + _offset) * _scale;
    FloatD sigmaB = (sampleGrid(mult(_variance->invNaturalTransform(), b)) + _offset) * _scale;
    return sigmaA * sigmaB * _stationaryCov->cov(a, b);

    Mat3Diff anisoA = Mat3Diff::Identity();
    Mat3Diff anisoB = Mat3Diff::Identity();

    FloatD detAnisoA = anisoA.determinant();
    FloatD detAnisoB = anisoB.determinant();

    Mat3Diff anisoABavg = 0.5 * (anisoA + anisoB);
    FloatD detAnisoABavg = anisoABavg.determinant();

    FloatD ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

    Vec3Diff d = b - a;
    FloatD dsq = d.transpose() * anisoABavg.inverse() * d;
    return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
}

FloatDD GridNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {

    auto sigmaA = (sampleGrid(mult(_variance->invNaturalTransform(), a)) + _offset) * _scale;
    auto sigmaB = (sampleGrid(mult(_variance->invNaturalTransform(), b)) + _offset) * _scale;
    return sigmaA * sigmaB * _stationaryCov->cov(a, b);

    auto anisoA = Mat3DD::Identity();
    auto anisoB = Mat3DD::Identity();

    auto detAnisoA = anisoA.determinant();
    auto detAnisoB = anisoB.determinant();

    auto anisoABavg = 0.5 * (anisoA + anisoB);
    auto detAnisoABavg = anisoABavg.determinant();

    auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

    auto d = b - a;
    auto dsq = d.transpose() * anisoABavg.inverse() * d;
    return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
}

double GridNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
    double sigmaA = (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(a)) + _offset) * _scale;
    double sigmaB = (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(b)) + _offset) * _scale;
    return sigmaA * sigmaB * _stationaryCov->cov(a, b);

    Eigen::Matrix3f anisoA = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f anisoB = Eigen::Matrix3f::Identity();

    double detAnisoA = anisoA.determinant();
    double detAnisoB = anisoB.determinant();

    Eigen::Matrix3f anisoABavg = 0.5 * (anisoA + anisoB);
    double detAnisoABavg = anisoABavg.determinant();

    double ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

    Eigen::Vector3f d = vec_conv<Eigen::Vector3f>(b - a);
    double dsq = d.transpose() * anisoABavg.inverse() * d;
    return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
}







double ProceduralNonstationaryCovariance::sample_spectral_density(PathSampleGenerator& sampler, Vec3d p) const {
    if (!_ls) {
        return _stationaryCov->sample_spectral_density(sampler, p);
    }
    else {
        return _stationaryCov->sample_spectral_density(sampler, p) / ((*_ls)(p)[0]);
    }
}

Vec2d ProceduralNonstationaryCovariance::sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p) const {
    if (!_ls) {
        return _stationaryCov->sample_spectral_density_2d(sampler, p);
    }
    else if (!_anisoField) {
        auto spec = _stationaryCov->sample_spectral_density_2d(sampler, p);
        spec = spec.length() * spec.normalized() / ((*_ls)(p)).xy();
        return spec;
    }
    else {
        Eigen::Matrix3d anisoMat = getAnisoRootForSpectral(p).inverse();
        auto spec = _stationaryCov->sample_spectral_density_2d(sampler, p);
        spec = spec.length() * mult((Eigen::Matrix2d)anisoMat.block(0,0,2,2), spec.normalized());
        return spec;
    }
}

Vec3d ProceduralNonstationaryCovariance::sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p) const {
    if (!_ls) {
        return _stationaryCov->sample_spectral_density_3d(sampler, p);
    }
    else if (!_anisoField) {
        auto spec = _stationaryCov->sample_spectral_density_3d(sampler, p);
        spec = spec.length() * spec.normalized() / ((*_ls)(p));
        return spec;
    }
    else {
        Eigen::Matrix3d anisoMat = getAnisoRootForSpectral(p).inverse();
        auto spec = _stationaryCov->sample_spectral_density_3d(sampler, p);
        spec = spec.length() * mult(anisoMat, spec.normalized());
        return spec;
    }
}


void ProceduralNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
    NonstationaryCovariance::fromJson(value, scene);

    if (auto variance = value["var"]) {
        _variance = scene.fetchProceduralScalar(variance);
    }

    if (auto variance = value["ls"]) {
        _ls = scene.fetchProceduralVector(variance);
    }

    if (auto aniso = value["aniso"]) {
        _anisoField = scene.fetchProceduralScalar(aniso);
    }
}

rapidjson::Value ProceduralNonstationaryCovariance::toJson(Allocator& allocator) const {
    auto obj = JsonObject{ NonstationaryCovariance::toJson(allocator), allocator,
                           "type", "proc_nonstationary",
                           "cov", *_stationaryCov,
                           "multiResolutionGrid", _multiResolutionGrid,
    };

    if (_variance) {
        obj.add("var", *_variance);
    }

    if (_ls) {
        obj.add("ls", *_ls);
    }

    if (_anisoField) {
        obj.add("aniso", *_anisoField);
    }

    return obj;
}

void ProceduralNonstationaryCovariance::loadResources() {
    if (_variance) _variance->loadResources();
    if (_ls) _ls->loadResources();
    if (_anisoField) _anisoField->loadResources();
}

std::string ProceduralNonstationaryCovariance::id() const {
    return tinyformat::format("pns-%s", _stationaryCov->id());
}

double ProceduralNonstationaryCovariance::getVariance(Vec3d p) const {
    if (_variance) return (*_variance)(p);
    return 1;
}

// For covariance function
Eigen::Matrix3d ProceduralNonstationaryCovariance::getAnisoRootForSpectral(Vec3d p) const {
    if (_anisoField && _ls) {
        Vec3d dir = Vec3d(1., 0., 0.); // (*_anisoField)(p).normalized();
        Vec3d ls = (*_ls)(p);
        return compute_ansio_simplified<Eigen::Matrix3d>(vec_conv<Eigen::Vector3d>(dir), vec_conv<Eigen::Vector3d>(ls));
    }
    else if (_ls) {
        Vec3d ls = (*_ls)(p);
        Eigen::Matrix3d anisoA = Eigen::Matrix3d::Identity();
        anisoA.diagonal().array() *= vec_conv<Eigen::Array3d>(ls);
        return anisoA;
    }

    return Eigen::Matrix3d::Identity();
}

Eigen::Matrix3d ProceduralNonstationaryCovariance::getCovMtx(Vec3d p) const {
    if (_anisoField && _ls) {
        float angle = (*_anisoField)(p) * PI_HALF;
        Vec3d ls = (*_ls)(p);
        Vec3d anisotropyAxis =  Vec3d(_anisotropyOnAxis, 1.0 / _anisotropyOnAxis, 0.0);
        return compute_ansio_full<Eigen::Matrix3d>(angle, vec_conv<Eigen::Vector3d>(anisotropyAxis * ls));
    }
    else if(_ls) {
        Vec3d ls = (*_ls)(p);
        Eigen::Matrix3d anisoA = Eigen::Matrix3d::Identity();
        anisoA.diagonal().array() *= vec_conv<Eigen::Array3d>(ls*ls);
        return anisoA;
    }

    return Eigen::Matrix3d::Identity();
}

// For convolution kernel
void ProceduralNonstationaryCovariance::getNonstationaryAniso3D(const Vec3f& p_world, std::shared_ptr<Eigen::Matrix3f>& aniso) const {
    if (_anisoField) {
        // The anisotropy is hard-coded here.
        // (Should be simple) future work: specify non-stationary anisotropy matrix similarly as SGGX
        float angle = (*_anisoField)(Vec3d(p_world)) * PI_HALF;
        Vec3f anisotropyAxis = Vec3f(_anisotropyOnAxis, 1.f / _anisotropyOnAxis, 1.f);
        aniso = std::make_shared<Eigen::Matrix3f>(compute_ansio_full<Eigen::Matrix3f>(angle, vec_conv<Eigen::Vector3f>(anisotropyAxis)));
    }
    else
        aniso = nullptr;
    return;
}

float ProceduralNonstationaryCovariance::getNonstationaryAniso1D(const Vec3f& p_world, const Vec3f& dirWorld) const {
    if (!_anisoField)
        return 1.0;
    std::shared_ptr<Eigen::Matrix3f> aniso_inv;
    getNonstationaryAniso3D(p_world, aniso_inv);
    *aniso_inv = aniso_inv->inverse();
    auto dir_eigen = to_eigen3f(dirWorld);
    return 1.0 / (dir_eigen.dot((*aniso_inv) * dir_eigen));
}

Eigen::Matrix3f ProceduralNonstationaryCovariance::getNonstationaryCovSplatCov3D(const Vec3f& pQueryWorld, const Vec3f& pCenterWorld) const {
    if (!_anisoField) {
        float scale = 0.5 * (sqr(nonStationarySplattingKernelScale(pQueryWorld)) + sqr(nonStationarySplattingKernelScale(pCenterWorld)));
        return Eigen::Matrix3f::Identity() * scale;
    }
    else {
        std::shared_ptr<Eigen::Matrix3f> aniso1, aniso2;
        getNonstationaryAniso3D(pQueryWorld, aniso1);
        getNonstationaryAniso3D(pCenterWorld, aniso2);
        float scale1 = nonStationarySplattingKernelScale(pQueryWorld);
        float scale2 = nonStationarySplattingKernelScale(pCenterWorld);
        return 0.5 * (sqr(scale1) * (*aniso1) + sqr(scale2) * (*aniso2));
    }
}

float ProceduralNonstationaryCovariance::getNonstationaryCovSplatCov1D(const Vec3f& pQueryWorld, const Vec3f& pCenterWorld, const Vec3f& dirLocal) const {
    if (!_anisoField) {
        float scale = 0.5 * (sqr(nonStationarySplattingKernelScale(pQueryWorld)) + sqr(nonStationarySplattingKernelScale(pCenterWorld)));
        return sqrt(scale);
    }
    else {
        Eigen::Matrix3f cov = getNonstationaryCovSplatCov3D(pQueryWorld, pCenterWorld);
        Vec3f dirWorld = to_vec3f(_stationaryCov->_local_to_world * to_eigen3f(dirLocal)).normalized();
        auto dir_eigen = to_eigen3f(dirWorld);
        return 1.0 / sqrt(dir_eigen.dot(cov.inverse() * dir_eigen));
    }
}

float ProceduralNonstationaryCovariance::getKernelScale(const Vec3f& p) const {
    if(_ls) {
        Vec3f ls = Vec3f((*_ls)(Vec3d(p)));
        return max(max(ls.x(), ls.y()), ls.z());
    }
    return 1.f;
}

float ProceduralNonstationaryCovariance::sparseConvNoiseMaxLateralScale() const {
    if(_ls)
        return _ls->maxVal();
    return 1.f;
}

float ProceduralNonstationaryCovariance::sparseConvNoiseMaxAnisotropyScale() const {
    if(_anisoField)
        return max(_anisotropyOnAxis, 1.f / _anisotropyOnAxis);
    return 1.f;
}

std::tuple<Eigen::MatrixXd, double> ProceduralNonstationaryCovariance::computeAnisoFactor(Vec3d a, Vec3d b) const {
    auto anisoA = getCovMtx(a);
    auto anisoB = getCovMtx(b);

    auto detAnisoA = anisoA.determinant();
    auto detAnisoB = anisoB.determinant();

    auto anisoABavg = 0.5 * (anisoA + anisoB);
    auto detAnisoABavg = (anisoABavg).determinant();

    auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);
    return {anisoABavg, ansioFac};
}

FloatD ProceduralNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {
    auto sigmaA = getVariance(from_diff(a));
    auto sigmaB = getVariance(from_diff(b));

    if (!_ls) {
        return sigmaA * sigmaB * _stationaryCov->cov(a, b);
    }

    auto [anisoABavg, anisoFac] = computeAnisoFactor(from_diff(a), from_diff(b));

    auto d = vec_conv<Vec3Diff>(b - a);
    FloatD dsq = d.transpose() * anisoABavg.inverse() * d;
    return sigmaA * sigmaB * anisoFac * _stationaryCov->cov(dsq);
}

FloatDD ProceduralNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {
    auto sigmaA = getVariance(from_diff(a));
    auto sigmaB = getVariance(from_diff(b));

    if (!_ls) {
    return sigmaA * sigmaB * _stationaryCov->cov(a, b);
    }

    auto [anisoABavg, anisoFac] = computeAnisoFactor(from_diff(a), from_diff(b));

    auto d = vec_conv<Vec3DD>(b - a);
    FloatDD dsq = d.transpose() * anisoABavg.inverse() * d;
    return sigmaA * sigmaB * anisoFac * _stationaryCov->cov(dsq);
}

double ProceduralNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
    auto sigmaA = getVariance(a);
    auto sigmaB = getVariance(b);

    if (!_ls) {
    return sigmaA * sigmaB * _stationaryCov->cov(a, b);
    }

    auto [anisoABavg, anisoFac] = computeAnisoFactor(a, b);

    auto d = vec_conv<Eigen::Vector3d>(b - a);
    auto dsq = d.transpose() * anisoABavg.inverse() * d;
    return sigmaA * sigmaB * anisoFac * _stationaryCov->cov(dsq);
}




void MeanGradNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
    NonstationaryCovariance::fromJson(value, scene);

    if (auto cov = value["cov"]) {
        _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
    }

    if (auto mean = value["mean"]) {
        _mean = std::dynamic_pointer_cast<MeanFunction>(scene.fetchMeanFunction(mean));
    }

    value.getField("aniso", _aniso);
}

rapidjson::Value MeanGradNonstationaryCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ NonstationaryCovariance::toJson(allocator), allocator,
        "type", "mg_nonstationary",
        "cov",*_stationaryCov,
        "mean", *_mean,
        "aniso", _aniso,
    };
}

void MeanGradNonstationaryCovariance::loadResources() {
    _stationaryCov->loadResources();
    if (_mean)
        _mean->loadResources();
}

Eigen::Matrix3d MeanGradNonstationaryCovariance::localAniso(Vec3d p) const {
    return compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(p).normalized()),
        vec_conv<Eigen::Vector3d>(_aniso));
}


FloatD MeanGradNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {
    Eigen::Matrix3d anisoA = compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(a))),
        vec_conv<Eigen::Vector3d>(_aniso));
    Eigen::Matrix3d anisoB = compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(b))),
        vec_conv<Eigen::Vector3d>(_aniso));

    auto detAnisoA = anisoA.determinant();
    auto detAnisoB = anisoB.determinant();

    Eigen::Matrix3d anisoABavg = 0.5 * (anisoA + anisoB);
    auto detAnisoABavg = anisoABavg.determinant();

    auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

    auto d = b - a;
    auto dsq = d.transpose() * anisoABavg.inverse() * d;
    return ansioFac * _stationaryCov->cov(dsq);
}

FloatDD MeanGradNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {
    Eigen::Matrix3d anisoA = compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(a))),
        vec_conv<Eigen::Vector3d>(_aniso));
    Eigen::Matrix3d anisoB = compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(b))),
        vec_conv<Eigen::Vector3d>(_aniso));

    auto detAnisoA = anisoA.determinant();
    auto detAnisoB = anisoB.determinant();

    Eigen::Matrix3d anisoABavg = 0.5 * (anisoA + anisoB);
    auto detAnisoABavg = anisoABavg.determinant();

    auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

    auto d = b - a;
    auto dsq = d.transpose() * anisoABavg.inverse() * d;
    return ansioFac * _stationaryCov->cov(dsq);
}

double MeanGradNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
    auto anisoA = compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(a)),
        vec_conv<Eigen::Vector3d>(_aniso));
    auto anisoB = compute_ansio_simplified<Eigen::Matrix3d>(
        vec_conv<Eigen::Vector3d>(_mean->dmean_da(b)),
        vec_conv<Eigen::Vector3d>(_aniso));

    auto detAnisoA = anisoA.determinant();
    auto detAnisoB = anisoB.determinant();

    Eigen::Matrix3d anisoABavg = 0.5 * (anisoA + anisoB);
    auto detAnisoABavg = anisoABavg.determinant();

    float ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

    Eigen::Vector3d d = vec_conv<Eigen::Vector3d>(b - a);
    double dsq = d.transpose() * anisoABavg.inverse() * d;
    return ansioFac * _stationaryCov->cov(dsq);
}





void NeuralNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
    NonstationaryCovariance::fromJson(value, scene);

    if (auto path = value["network"]) {
        _path = scene.fetchResource(path);
    }

    value.getField("scale", _scale);
    value.getField("transform", _configTransform);
    _invConfigTransform = _configTransform.invert();
}

rapidjson::Value NeuralNonstationaryCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ NonstationaryCovariance::toJson(allocator), allocator,
                       "type", "nonstationary",
                       "network", *_path,
                       "scale", _scale,
                       "transform", _configTransform
    };
}

void NeuralNonstationaryCovariance::loadResources() {
    CovarianceFunction::loadResources();

    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(*_path);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
    }

    _nn = std::make_shared<GPNeuralNetwork>();
    _nn->read(*document, _path->absolute().parent());
}

FloatD NeuralNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {
    return _nn->cov(mult(_invConfigTransform, a), mult(_invConfigTransform, b)) * _scale;
}

FloatDD NeuralNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {
    return _nn->cov(mult(_invConfigTransform, a), mult(_invConfigTransform, b)) * _scale;
}

double NeuralNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
    return _nn->cov(mult(_invConfigTransform, a), mult(_invConfigTransform, b)) * _scale;
}

}