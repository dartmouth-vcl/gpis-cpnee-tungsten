#include "Gaussian.hpp"

#include "math/Angle.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <random>
#include <cmath>

namespace Tungsten {
    CovMatrix project_to_psd(const CovMatrix& in) {

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(in);
        auto eps = 1e6 * std::numeric_limits<double>::epsilon() * eigs.eigenvalues()[0];

        CovMatrix result = eigs.eigenvectors()
            * eigs.eigenvalues().array().max(eps).matrix() * eigs.eigenvectors();
        //result.diagonal().array() += 1e-6;
        return result;
    }

    // The following codes are adapted from https://people.sc.fsu.edu/~jburkardt/cpp_src/truncated_normal/truncated_normal.cpp.
    //  Licensing:
    //
    //    This code is distributed under the MIT license.
    //
    //  Modified:
    //
    //    02 October 2012
    //
    //  Author:
    //
    //    John Burkardt
    //

    // Box muller transform
    Vec2d rand_normal_2(UniformSampler& sampler) {
        double u1 = sampler.next1D();
        double u2 = sampler.next1D();

        double r = sqrt(-2 * log(1. - u1));
        double x = cos(2 * PI * u2);
        double y = sin(2 * PI * u2);
        double z1 = r * x;
        double z2 = r * y;

        return Vec2d(z1, z2);

    }

    Vec2d rand_normal_2(PathSampleGenerator& sampler) {
        double u1 = sampler.next1D();
        double u2 = sampler.next1D();

        double r = sqrt(-2 * log(1. - u1));
        double x = cos(2 * PI * u2);
        double y = sin(2 * PI * u2);
        double z1 = r * x;
        double z2 = r * y;

        return Vec2d(z1, z2);

    }

    double rand_gamma(double shape, double mean, PathSampleGenerator& sampler) {
        double scale = mean / shape;
        // Not ideal
        std::mt19937 rnd(sampler.nextDiscrete(1 << 16));
        std::gamma_distribution<> gamma_dist(shape, scale);
        return gamma_dist(rnd);
    }

    double rand_truncated_normal(double mean, double sigma, double a, PathSampleGenerator& sampler) {
        if (abs(a - mean) < 0.000001) {
            return abs(mean + sigma * rand_normal_2(sampler).x());
        }

        if (a < mean) {
            while (true) {
                double x = mean + sigma * rand_normal_2(sampler).x();
                if (x >= a) {
                    return x;
                }
            }
        }

        double a_bar = (a - mean) / sigma;
        double x_bar;

        for(int i = 0; i < 1000; i++) {
            double u = sampler.next1D();
            x_bar = sqrt(a_bar * a_bar - 2 * log(1 - u));
            double v = sampler.next1D();

            if (v < x_bar / a_bar) {
                break;
            }
        }

        return sigma * x_bar + mean;
    }

    Eigen::VectorXd sample_standard_normal(int n, UniformSampler& sampler) {
        Eigen::VectorXd result(n);
        // We're always getting two samples, so make use of that
        for (int i = 0; i < result.size() / 2; i++) {
            Vec2d norm_samp = rand_normal_2(sampler);
            result(i * 2) = norm_samp.x();
            result(i * 2 + 1) = norm_samp.y();
        }

        // Fill up the last one for an uneven number of samples
        if (result.size() % 2) {
            Vec2d norm_samp = rand_normal_2(sampler);
            result(result.size() - 1) = norm_samp.x();
        }
        return result;
    }

    Eigen::VectorXd sample_standard_normal(int n, PathSampleGenerator& sampler) {
        Eigen::VectorXd result(n);
        // We're always getting two samples, so make use of that
        for (int i = 0; i < result.size() / 2; i++) {
            Vec2d norm_samp = rand_normal_2(sampler);
            result(i * 2) = norm_samp.x();
            result(i * 2 + 1) = norm_samp.y();
        }

        // Fill up the last one for an uneven number of samples
        if (result.size() % 2) {
            Vec2d norm_samp = rand_normal_2(sampler);
            result(result.size() - 1) = norm_samp.x();
        }
        return result;
    }

    MultivariateNormalDistribution::MultivariateNormalDistribution(const Eigen::VectorXd& _mean, const CovMatrix& _cov) : mean(_mean) {
#if 0
        svd = Eigen::BDCSVD<Eigen::MatrixXd>(_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

        if (svd.info() != Eigen::Success) {
            std::cerr << "SVD for MVN computations failed!\n";
        }

        double logDetCov = 0;
        for (int i = 0; i < svd.nonzeroSingularValues(); i++) {
            logDetCov += log(svd.singularValues()(i));
        }
        sqrt2PiN = std::exp(logDetCov);

        // Compute the square root of the PSD matrix
        normTransform = svd.matrixU() * svd.singularValues().array().max(0).sqrt().matrix().asDiagonal() * svd.matrixV().transpose();

#else

#ifdef SPARSE_COV
        Eigen::SimplicialLLT<CovMatrix> chol(_cov);
#else
        Eigen::LLT<Eigen::MatrixXd> chol(_cov.triangularView<Eigen::Lower>());
#endif

        // We can only use the cholesky decomposition if 
        // the covariance matrix is symmetric, pos-definite.
        // But a covariance matrix might be pos-semi-definite.
        // In that case, we'll go to an EigenSolver
        if (chol.info() == Eigen::Success) {
            // Use cholesky solver
            normTransform = chol.matrixL();
        }
        else
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(_cov);

            if (eigs.info() != Eigen::ComputationInfo::Success) {
                std::cerr << "Matrix square root failed!\n";
            }

            normTransform = eigs.eigenvectors()
                * eigs.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
        }
#endif

    }

    double MultivariateNormalDistribution::eval(const Eigen::VectorXd& x) const {
        Eigen::VectorXd diff = x - mean;

        double quadform = diff.transpose() * svd.solve(diff);

        double inv_sqrt_2pi = 0.3989422804014327;
        double normConst = pow(inv_sqrt_2pi, x.rows()) * pow(sqrt2PiN, -.5);
        return normConst * exp(-.5 * quadform);
    }

    Eigen::MatrixXd MultivariateNormalDistribution::sample(const Constraint* constraints, int numConstraints,
        int samples, PathSampleGenerator& sampler) const {

        // Generate a vector of standard normal variates with the same dimension as the mean
        Eigen::VectorXd z = Eigen::VectorXd(mean.size());
        Eigen::MatrixXd sample(mean.size(), samples);

        int numTries = 0;
        for (int j = 0; j < samples; /*only advance sample idx if the sample passes all constraints*/) {

            numTries++;

            // We're always getting two samples, so make use of that
            for (int i = 0; i < mean.size() / 2; i++) {
                Vec2d norm_samp = rand_normal_2(sampler); // { (float)random_standard_normal(sampler), (float)random_standard_normal(sampler) };
                z(i * 2) = norm_samp.x();
                z(i * 2 + 1) = norm_samp.y();
            }

            // Fill up the last one for an uneven number of samples
            if (mean.size() % 2) {
                Vec2d norm_samp = rand_normal_2(sampler);
                z(mean.size() - 1) = norm_samp.x();
            }

            Eigen::VectorXd currSample = mean + normTransform * z;

            // Check constraints
            bool passedConstraints = true;
            for (int cIdx = 0; cIdx < numConstraints; cIdx++) {
                const Constraint& con = constraints[cIdx];

                for (int i = con.startIdx; i <= con.endIdx; i++) {
                    if (currSample(i) < con.minV || currSample(i) > con.maxV) {
                        passedConstraints = false;
                        break;
                    }
                }

                if (!passedConstraints) {
                    break;
                }
            }

            if (passedConstraints || numTries > 100000) {
                if (numTries > 100000) {
                    std::cout << "Constraint not satisfied. " << mean(0) << "\n";
                }
                sample.col(j) = currSample;
                j++;
                numTries = 0;
            }
        }

        return sample;
    }

    // ---------------------------------------------------------------
    // Normal and truncated normal distribution utilities
    // Adapted from:
    //   Norman Johnson, Samuel Kotz, Narayanaswamy Balakrishnan,
    //   Continuous Univariate Distributions, Second edition, Wiley, 1994.
    //   https://people.sc.fsu.edu/~jburkardt/cpp_src/truncated_normal/truncated_normal.html
    // ---------------------------------------------------------------

    static double poly_value_horner(int m, double c[], double x) {
        double value = c[m];
        for (int i = m - 1; 0 <= i; i--)
            value = value * x + c[i];
        return value;
    }

    // Binomial coefficient C(n, k), accumulated to avoid overflow and roundoff
    static double choose(int n, int k) {
        int mn = k < n - k ? k : n - k;
        int mx = k < n - k ? n - k : k;

        if (mn < 0)
            return 0.0;
        if (mn == 0)
            return 1.0;

        double value = mx + 1;
        for (int i = 2; i <= mn; i++)
            value = (value * (mx + i)) / i;

        return value;
    }

    // Double factorial n!! = n * (n - 2) * (n - 4) * ..., and 1 for n < 1
    static double factorial2(int n) {
        double value = 1.0;
        for (int i = n; 1 < i; i -= 2)
            value *= i;
        return value;
    }

    // The i-th power of -1
    static double mop(int i) {
        return (i % 2) == 0 ? 1.0 : -1.0;
    }

    double normal_01_cdf(double x) {
        double a1 = 0.398942280444;
        double a2 = 0.399903438504;
        double a3 = 5.75885480458;
        double a4 = 29.8213557808;
        double a5 = 2.62433121679;
        double a6 = 48.6959930692;
        double a7 = 5.92885724438;
        double b0 = 0.398942280385;
        double b1 = 3.8052E-08;
        double b2 = 1.00000615302;
        double b3 = 3.98064794E-04;
        double b4 = 1.98615381364;
        double b5 = 0.151679116635;
        double b6 = 5.29330324926;
        double b7 = 4.8385912808;
        double b8 = 15.1508972451;
        double b9 = 0.742380924027;
        double b10 = 30.789933034;
        double b11 = 3.99019417011;
        double cdf;
        double q;
        double y;

        if (std::fabs(x) <= 1.28) {
            y = 0.5 * x * x;
            q = 0.5 - std::fabs(x) * (a1 - a2 * y / (y + a3 - a4 / (y + a5
                + a6 / (y + a7))));
        } else if (std::fabs(x) <= 12.7) {
            y = 0.5 * x * x;
            q = std::exp(-y) * b0 / (std::fabs(x) - b1
                + b2 / (std::fabs(x) + b3
                    + b4 / (std::fabs(x) - b5
                        + b6 / (std::fabs(x) + b7
                            - b8 / (std::fabs(x) + b9
                                + b10 / (std::fabs(x) + b11))))));
        } else {
            q = 0.0;
        }

        if (x < 0.0)
            cdf = q;
        else
            cdf = 1.0 - q;

        return cdf;
    }

    double normal_01_cdf_inv(double p) {
        double a[8] = {
            3.3871328727963666080,     1.3314166789178437745E+2,
            1.9715909503065514427E+3,  1.3731693765509461125E+4,
            4.5921953931549871457E+4,  6.7265770927008700853E+4,
            3.3430575583588128105E+4,  2.5090809287301226727E+3 };
        double b[8] = {
            1.0,                       4.2313330701600911252E+1,
            6.8718700749205790830E+2,  5.3941960214247511077E+3,
            2.1213794301586595867E+4,  3.9307895800092710610E+4,
            2.8729085735721942674E+4,  5.2264952788528545610E+3 };
        double c[8] = {
            1.42343711074968357734,     4.63033784615654529590,
            5.76949722146069140550,     3.64784832476320460504,
            1.27045825245236838258,     2.41780725177450611770E-1,
            2.27238449892691845833E-2,  7.74545014278341407640E-4 };
        double const1 = 0.180625;
        double const2 = 1.6;
        double d[8] = {
            1.0,                        2.05319162663775882187,
            1.67638483018380384940,     6.89767334985100004550E-1,
            1.48103976427480074590E-1,  1.51986665636164571966E-2,
            5.47593808499534494600E-4,  1.05075007164441684324E-9 };
        double e[8] = {
            6.65790464350110377720,     5.46378491116411436990,
            1.78482653991729133580,     2.96560571828504891230E-1,
            2.65321895265761230930E-2,  1.24266094738807843860E-3,
            2.71155556874348757815E-5,  2.01033439929228813265E-7 };
        double f[8] = {
            1.0,                        5.99832206555887937690E-1,
            1.36929880922735805310E-1,  1.48753612908506148525E-2,
            7.86869131145613259100E-4,  1.84631831751005468180E-5,
            1.42151175831644588870E-7,  2.04426310338993978564E-15 };
        double q;
        double r;
        double split1 = 0.425;
        double split2 = 5.0;
        double value;

        if (p <= 0.0) {
            return -HUGE_VAL;
        }
        if (1.0 <= p) {
            return HUGE_VAL;
        }

        q = p - 0.5;

        if (std::fabs(q) <= split1) {
            r = const1 - q * q;
            value = q * poly_value_horner(7, a, r)
                / poly_value_horner(7, b, r);
        } else {
            if (q < 0.0)
                r = p;
            else
                r = 1.0 - p;

            if (r <= 0.0) {
                value = HUGE_VAL;
            } else {
                r = std::sqrt(-std::log(r));

                if (r <= split2) {
                    r = r - const2;
                    value = poly_value_horner(7, c, r)
                        / poly_value_horner(7, d, r);
                } else {
                    r = r - split2;
                    value = poly_value_horner(7, e, r)
                        / poly_value_horner(7, f, r);
                }
            }

            if (q < 0.0)
                value = -value;
        }

        return value;
    }

    double normal_01_moment(int order) {
        if (order % 2)
            return 0.0;

        return factorial2(order - 1);
    }

    double normal_01_pdf(double x) {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
    }

    double normal_01_sample(Vec2d rv2) {
        return std::sqrt(-2.0 * std::log(rv2.x())) * std::cos(2.0 * PI * rv2.y());
    }

    double normal_ms_cdf(double x, double mu, double sigma) {
        double y = (x - mu) / sigma;
        return normal_01_cdf(y);
    }

    double normal_ms_cdf_inv(double cdf, double mu, double sigma) {
        if (cdf < 0.0 || 1.0 < cdf) {
            std::cerr << "normal_ms_cdf_inv: CDF out of range [0,1]: " << cdf << std::endl;
        }
        double x2 = normal_01_cdf_inv(cdf);
        return mu + sigma * x2;
    }

    double normal_ms_moment(int order, double mu, double sigma) {
        double value = 0.0;
        for (int j = 0; j <= order / 2; j++) {
            value += choose(order, 2 * j) * factorial2(2 * j - 1)
                * std::pow(mu, order - 2 * j) * std::pow(sigma, 2 * j);
        }
        return value;
    }

    double normal_ms_moment_central(int order, double sigma) {
        if (order % 2)
            return 0.0;

        return factorial2(order - 1) * std::pow(sigma, order);
    }

    double normal_ms_pdf(double x, double mu, double sigma) {
        double y = (x - mu) / sigma;
        return std::exp(-0.5 * y * y) / (sigma * std::sqrt(2.0 * PI));
    }

    double normal_ms_sample(double mu, double sigma, Vec2d rv2) {
        return mu + sigma * normal_01_sample(rv2);
    }

    double truncated_normal_ab_cdf(double x, double mu, double sigma, double a, double b) {
        if (x < a)
            return 0.0;
        if (x > b)
            return 1.0;

        double alpha = (a - mu) / sigma;
        double beta  = (b - mu) / sigma;
        double xi    = (x - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double beta_cdf  = normal_01_cdf(beta);
        double xi_cdf    = normal_01_cdf(xi);

        return (xi_cdf - alpha_cdf) / (beta_cdf - alpha_cdf);
    }

    double truncated_normal_ab_cdf_inv(double cdf, double mu, double sigma, double a, double b) {
        if (cdf < 0.0 || 1.0 < cdf) {
            std::cerr << "truncated_normal_ab_cdf_inv: CDF out of range [0,1]: " << cdf << std::endl;
        }

        double alpha = (a - mu) / sigma;
        double beta  = (b - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double beta_cdf  = normal_01_cdf(beta);

        double xi_cdf = (beta_cdf - alpha_cdf) * cdf + alpha_cdf;
        double xi     = normal_01_cdf_inv(xi_cdf);

        return mu + sigma * xi;
    }

    double truncated_normal_ab_mean(double mu, double sigma, double a, double b) {
        double alpha = (a - mu) / sigma;
        double beta  = (b - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double beta_cdf  = normal_01_cdf(beta);

        double alpha_pdf = normal_01_pdf(alpha);
        double beta_pdf  = normal_01_pdf(beta);

        return mu + sigma * (alpha_pdf - beta_pdf) / (beta_cdf - alpha_cdf);
    }

    double truncated_normal_ab_moment(int order, double mu, double sigma, double a, double b) {
        if (order < 0) {
            std::cerr << "truncated_normal_ab_moment: order must be non-negative: " << order << std::endl;
            return 0.0;
        }
        if (sigma <= 0.0) {
            std::cerr << "truncated_normal_ab_moment: sigma must be positive: " << sigma << std::endl;
            return 0.0;
        }
        if (b <= a) {
            std::cerr << "truncated_normal_ab_moment: empty interval [" << a << ", " << b << "]" << std::endl;
            return 0.0;
        }

        double a_h   = (a - mu) / sigma;
        double a_pdf = normal_01_pdf(a_h);
        double a_cdf = normal_01_cdf(a_h);

        double b_h   = (b - mu) / sigma;
        double b_pdf = normal_01_pdf(b_h);
        double b_cdf = normal_01_cdf(b_h);

        if (a_cdf == 0.0 || b_cdf == 0.0) {
            std::cerr << "truncated_normal_ab_moment: PDF/CDF ratio fails, CDF too small: "
                << a_cdf << ", " << b_cdf << std::endl;
            return 0.0;
        }

        double moment = 0.0;
        double irm2 = 0.0;
        double irm1 = 0.0;

        for (int r = 0; r <= order; r++) {
            double ir;
            if (r == 0) {
                ir = 1.0;
            } else if (r == 1) {
                ir = -(b_pdf - a_pdf) / (b_cdf - a_cdf);
            } else {
                ir = (r - 1) * irm2
                    - (std::pow(b_h, r - 1) * b_pdf - std::pow(a_h, r - 1) * a_pdf)
                    / (b_cdf - a_cdf);
            }

            moment += choose(order, r) * std::pow(mu, order - r) * std::pow(sigma, r) * ir;

            irm2 = irm1;
            irm1 = ir;
        }

        return moment;
    }

    double truncated_normal_ab_pdf(double x, double mu, double sigma, double a, double b) {
        if (x < a || x > b)
            return 0.0;

        double alpha = (a - mu) / sigma;
        double beta  = (b - mu) / sigma;
        double xi    = (x - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double beta_cdf  = normal_01_cdf(beta);
        double xi_pdf    = normal_01_pdf(xi);

        return xi_pdf / (beta_cdf - alpha_cdf) / sigma;
    }

    double truncated_normal_ab_sample(double mu, double sigma, double a, double b, double rv) {
        double alpha = (a - mu) / sigma;
        double beta  = (b - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double beta_cdf  = normal_01_cdf(beta);

        double xi_cdf = alpha_cdf + rv * (beta_cdf - alpha_cdf);
        double xi     = normal_01_cdf_inv(xi_cdf);

        return mu + sigma * xi;
    }

    double truncated_normal_ab_variance(double mu, double sigma, double a, double b) {
        double alpha = (a - mu) / sigma;
        double beta  = (b - mu) / sigma;

        double alpha_pdf = normal_01_pdf(alpha);
        double beta_pdf  = normal_01_pdf(beta);

        double alpha_cdf = normal_01_cdf(alpha);
        double beta_cdf  = normal_01_cdf(beta);

        double ratio = (alpha_pdf - beta_pdf) / (beta_cdf - alpha_cdf);

        return sigma * sigma * (1.0
            + (alpha * alpha_pdf - beta * beta_pdf) / (beta_cdf - alpha_cdf)
            - ratio * ratio);
    }

    double truncated_normal_a_cdf(double x, double mu, double sigma, double a) {
        if (x < a)
            return 0.0;

        double alpha = (a - mu) / sigma;
        double xi    = (x - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double xi_cdf    = normal_01_cdf(xi);

        return (xi_cdf - alpha_cdf) / (1.0 - alpha_cdf);
    }

    double truncated_normal_a_cdf_inv(double cdf, double mu, double sigma, double a) {
        if (cdf < 0.0 || 1.0 < cdf) {
            std::cerr << "truncated_normal_a_cdf_inv: CDF out of range [0,1]: " << cdf << std::endl;
        }

        double alpha = (a - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);

        double xi_cdf = (1.0 - alpha_cdf) * cdf + alpha_cdf;
        double xi     = normal_01_cdf_inv(xi_cdf);

        return mu + sigma * xi;
    }

    double truncated_normal_a_mean(double mu, double sigma, double a) {
        double alpha = (a - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double alpha_pdf = normal_01_pdf(alpha);

        return mu + sigma * alpha_pdf / (1.0 - alpha_cdf);
    }

    double truncated_normal_a_moment(int order, double mu, double sigma, double a) {
        return mop(order) * truncated_normal_b_moment(order, -mu, sigma, -a);
    }

    double truncated_normal_a_pdf(double x, double mu, double sigma, double a) {
        if (x < a)
            return 0.0;

        double alpha = (a - mu) / sigma;
        double xi    = (x - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);
        double xi_pdf    = normal_01_pdf(xi);

        return xi_pdf / (1.0 - alpha_cdf) / sigma;
    }

    double truncated_normal_a_sample(double mu, double sigma, double a, double rv) {
        double alpha = (a - mu) / sigma;

        double alpha_cdf = normal_01_cdf(alpha);

        double xi_cdf = alpha_cdf + rv * (1.0 - alpha_cdf);
        double xi     = normal_01_cdf_inv(xi_cdf);

        return mu + sigma * xi;
    }

    double truncated_normal_a_variance(double mu, double sigma, double a) {
        double alpha = (a - mu) / sigma;

        double alpha_pdf = normal_01_pdf(alpha);
        double alpha_cdf = normal_01_cdf(alpha);

        double ratio = alpha_pdf / (1.0 - alpha_cdf);

        return sigma * sigma * (1.0 + alpha * ratio - ratio * ratio);
    }

    double truncated_normal_b_cdf(double x, double mu, double sigma, double b) {
        if (x > b)
            return 1.0;

        double beta = (b - mu) / sigma;
        double xi   = (x - mu) / sigma;

        double beta_cdf = normal_01_cdf(beta);
        double xi_cdf   = normal_01_cdf(xi);

        return xi_cdf / beta_cdf;
    }

    double truncated_normal_b_cdf_inv(double cdf, double mu, double sigma, double b) {
        if (cdf < 0.0 || 1.0 < cdf) {
            std::cerr << "truncated_normal_b_cdf_inv: CDF out of range [0,1]: " << cdf << std::endl;
        }

        double beta = (b - mu) / sigma;

        double beta_cdf = normal_01_cdf(beta);

        double xi_cdf = beta_cdf * cdf;
        double xi     = normal_01_cdf_inv(xi_cdf);

        return mu + sigma * xi;
    }

    double truncated_normal_b_mean(double mu, double sigma, double b) {
        double beta = (b - mu) / sigma;

        double beta_cdf = normal_01_cdf(beta);
        double beta_pdf = normal_01_pdf(beta);

        return mu - sigma * beta_pdf / beta_cdf;
    }

    double truncated_normal_b_moment(int order, double mu, double sigma, double b) {
        if (order < 0) {
            std::cerr << "truncated_normal_b_moment: order must be non-negative: " << order << std::endl;
            return 0.0;
        }

        double h     = (b - mu) / sigma;
        double h_pdf = normal_01_pdf(h);
        double h_cdf = normal_01_cdf(h);

        if (h_cdf == 0.0) {
            std::cerr << "truncated_normal_b_moment: PDF/CDF ratio fails, CDF too small: " << h_cdf << std::endl;
            return 0.0;
        }

        double f = h_pdf / h_cdf;

        double moment = 0.0;
        double irm2 = 0.0;
        double irm1 = 0.0;

        for (int r = 0; r <= order; r++) {
            double ir;
            if (r == 0) {
                ir = 1.0;
            } else if (r == 1) {
                ir = -f;
            } else {
                ir = -std::pow(h, r - 1) * f + (r - 1) * irm2;
            }

            moment += choose(order, r) * std::pow(mu, order - r) * std::pow(sigma, r) * ir;

            irm2 = irm1;
            irm1 = ir;
        }

        return moment;
    }

    double truncated_normal_b_pdf(double x, double mu, double sigma, double b) {
        if (x > b)
            return 0.0;

        double beta = (b - mu) / sigma;
        double xi   = (x - mu) / sigma;

        double beta_cdf = normal_01_cdf(beta);
        double xi_pdf   = normal_01_pdf(xi);

        return xi_pdf / beta_cdf / sigma;
    }

    double truncated_normal_b_sample(double mu, double sigma, double b, double rv) {
        double beta = (b - mu) / sigma;

        double beta_cdf = normal_01_cdf(beta);

        double xi_cdf = rv * beta_cdf;
        double xi     = normal_01_cdf_inv(xi_cdf);

        return mu + sigma * xi;
    }

    double truncated_normal_b_variance(double mu, double sigma, double b) {
        double beta = (b - mu) / sigma;

        double beta_pdf = normal_01_pdf(beta);
        double beta_cdf = normal_01_cdf(beta);

        double ratio = beta_pdf / beta_cdf;

        return sigma * sigma * (1.0 - beta * ratio - ratio * ratio);
    }

    double truncated_normal_ab_tails_pdf(double x, double mu, double sigma, double a, double b) {
        if (a < x && x < b)
            return 0.0;

        double probA = normal_ms_cdf(a, mu, sigma);
        double probB = 1.0 - normal_ms_cdf(b, mu, sigma);

        double sum = probA + probB;
        if (sum <= 0.0)
            return 0.0;

        return normal_ms_pdf(x, mu, sigma) / sum;
    }

    double truncated_normal_ab_tails_sample(double mu, double sigma, double a, double b, double rv) {
        double probA = normal_ms_cdf(a, mu, sigma);
        double probB = 1.0 - normal_ms_cdf(b, mu, sigma);

        double sum = probA + probB;
        if (sum <= 0.0) {
            // std::cerr << "truncated_normal_ab_tails_sample: both tails have vanishing mass: "
            //     << a << ", " << b << std::endl;
            return rv < 0.5 ? a : b;
        }

        // Pick a tail proportionally to its mass, then rescale the variate to
        // cover [0, 1) again within that tail.
        probA /= sum;

        if (rv < probA) {
            // Sample the left tail, x <= a. That tail is bounded from above, so it
            // is the upper-truncated ("_b_") sampler that applies here, with a as
            // its upper limit.
            return truncated_normal_b_sample(mu, sigma, a, rv / probA);
        }

        // Sample the right tail, x >= b, bounded from below by b.
        return truncated_normal_a_sample(mu, sigma, b, (rv - probA) / (1.0 - probA));
    }
}