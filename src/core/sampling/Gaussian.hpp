#ifndef GAUSSIAN_HPP_
#define GAUSSIAN_HPP_

#include "math/MathUtil.hpp"
#include "sampling/PathSampleGenerator.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace Tungsten {

    //#define SPARSE_COV
#ifdef SPARSE_COV
    using CovMatrix = Eigen::SparseMatrix<double>;
#else
    using CovMatrix = Eigen::MatrixXd;
#endif

    CovMatrix project_to_psd(const CovMatrix& in);

    // Box muller transform
    Vec2d rand_normal_2(UniformSampler& sampler);
    Vec2d rand_normal_2(PathSampleGenerator& sampler);
    double rand_truncated_normal(double mean, double sigma, double a, PathSampleGenerator& sampler);
    double rand_gamma(double shape, double mean, PathSampleGenerator& samples);

    // Normal distribution utilities
    /// @brief Evaluates the standard normal (0,1) CDF.
    double normal_01_cdf(double x);
    /// @brief Inverts the standard normal (0,1) CDF. 0 < p < 1.
    double normal_01_cdf_inv(double p);
    /// @brief Evaluates the moment of given order of the standard normal (0,1) PDF.
    double normal_01_moment(int order);
    /// @brief Evaluates the standard normal (0,1) PDF.
    double normal_01_pdf(double x);
    /// @brief Samples the standard normal (0,1) distribution via Box-Muller.
    double normal_01_sample(Vec2d rv2);

    /// @brief Evaluates the normal (mu, sigma) CDF.
    double normal_ms_cdf(double x, double mu, double sigma);
    /// @brief Inverts the normal (mu, sigma) CDF.
    double normal_ms_cdf_inv(double cdf, double mu, double sigma);
    /// @brief Evaluates the moment of given order of the normal (mu, sigma) PDF.
    double normal_ms_moment(int order, double mu, double sigma);
    /// @brief Evaluates the central moment of given order of the normal (mu, sigma) PDF.
    double normal_ms_moment_central(int order, double sigma);
    /// @brief Evaluates the normal (mu, sigma) PDF.
    double normal_ms_pdf(double x, double mu, double sigma);
    /// @brief Samples the normal (mu, sigma) distribution.
    double normal_ms_sample(double mu, double sigma, Vec2d rv2);

    /// @brief Evaluates the truncated normal CDF on [a, b].
    double truncated_normal_ab_cdf(double x, double mu, double sigma, double a, double b);
    /// @brief Inverts the truncated normal CDF on [a, b].
    double truncated_normal_ab_cdf_inv(double cdf, double mu, double sigma, double a, double b);
    /// @brief Returns the mean of the truncated normal PDF on [a, b].
    double truncated_normal_ab_mean(double mu, double sigma, double a, double b);
    /// @brief Evaluates the moment of given order of the truncated normal PDF on [a, b].
    double truncated_normal_ab_moment(int order, double mu, double sigma, double a, double b);
    /// @brief Evaluates the truncated normal PDF on [a, b].
    double truncated_normal_ab_pdf(double x, double mu, double sigma, double a, double b);
    /// @brief Samples the truncated normal distribution on [a, b] via inverse CDF.
    double truncated_normal_ab_sample(double mu, double sigma, double a, double b, double rv);
    /// @brief Returns the variance of the truncated normal PDF on [a, b].
    double truncated_normal_ab_variance(double mu, double sigma, double a, double b);

    /// @brief Evaluates the PDF of the double-sided tails of a normal (mu, sigma).
    /// a < b; the support is (-inf, a] union [b, +inf), and the PDF is 0 on (a, b).
    double truncated_normal_ab_tails_pdf(double x, double mu, double sigma, double a, double b);
    /// @brief Samples the double-sided tails of a normal (mu, sigma), picking a tail
    /// proportionally to its mass. a < b; the sample lies in (-inf, a] union [b, +inf).
    double truncated_normal_ab_tails_sample(double mu, double sigma, double a, double b, double rv);

    /// @brief Evaluates the lower truncated normal CDF on [a, +inf).
    double truncated_normal_a_cdf(double x, double mu, double sigma, double a);
    /// @brief Inverts the lower truncated normal CDF on [a, +inf).
    double truncated_normal_a_cdf_inv(double cdf, double mu, double sigma, double a);
    /// @brief Returns the mean of the lower truncated normal PDF on [a, +inf).
    double truncated_normal_a_mean(double mu, double sigma, double a);
    /// @brief Evaluates the moment of given order of the lower truncated normal PDF on [a, +inf).
    double truncated_normal_a_moment(int order, double mu, double sigma, double a);
    /// @brief Evaluates the lower truncated normal PDF on [a, +inf).
    double truncated_normal_a_pdf(double x, double mu, double sigma, double a);
    /// @brief Samples the lower truncated normal distribution on [a, +inf) via inverse CDF.
    double truncated_normal_a_sample(double mu, double sigma, double a, double rv);
    /// @brief Returns the variance of the lower truncated normal PDF on [a, +inf).
    double truncated_normal_a_variance(double mu, double sigma, double a);

    /// @brief Evaluates the upper truncated normal CDF on (-inf, b].
    double truncated_normal_b_cdf(double x, double mu, double sigma, double b);
    /// @brief Inverts the upper truncated normal CDF on (-inf, b].
    double truncated_normal_b_cdf_inv(double cdf, double mu, double sigma, double b);
    /// @brief Returns the mean of the upper truncated normal PDF on (-inf, b].
    double truncated_normal_b_mean(double mu, double sigma, double b);
    /// @brief Evaluates the moment of given order of the upper truncated normal PDF on (-inf, b].
    double truncated_normal_b_moment(int order, double mu, double sigma, double b);
    /// @brief Evaluates the upper truncated normal PDF on (-inf, b].
    double truncated_normal_b_pdf(double x, double mu, double sigma, double b);
    /// @brief Samples the upper truncated normal distribution on (-inf, b] via inverse CDF.
    double truncated_normal_b_sample(double mu, double sigma, double b, double rv);
    /// @brief Returns the variance of the upper truncated normal PDF on (-inf, b].
    double truncated_normal_b_variance(double mu, double sigma, double b);

    struct Constraint {
        int startIdx, endIdx;
        float minV, maxV;
    };

    Eigen::VectorXd sample_standard_normal(int n, UniformSampler& sampler);

    Eigen::VectorXd sample_standard_normal(int n, PathSampleGenerator& sampler);

    struct MultivariateNormalDistribution {
        Eigen::VectorXd mean;

        Eigen::BDCSVD<Eigen::MatrixXd> svd;
        //Eigen::LLT<Eigen::MatrixXd> chol;

        double sqrt2PiN;

        Eigen::MatrixXd normTransform;

        MultivariateNormalDistribution(const Eigen::VectorXd& _mean, const CovMatrix& _cov);

        double eval(const Eigen::VectorXd& x) const;

        Eigen::MatrixXd sample(const Constraint* constraints, int numConstraints,
            int samples, PathSampleGenerator& sampler) const;
    };

}

#endif /* GAUSSIAN_HPP_ */
