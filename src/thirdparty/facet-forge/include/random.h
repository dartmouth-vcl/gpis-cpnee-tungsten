/*
 * Copyright (c) <2023> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <vector.h>
#include <random>
#include <math.h>

//////////////////////////////////////////////////////////////////////////////////
// Random numbers
//////////////////////////////////////////////////////////////////////////////////

#ifndef M_PI
const double M_PI = 3.141592653f;
#endif 

extern std::mt19937 g_mt;
typedef std::gamma_distribution<> D_gamma;

inline double RandomReal()
{
    std::uniform_real_distribution<double> r(0., 1.);
    return r(g_mt);
}

inline double RandomReal(double a, double b)
{
    std::uniform_real_distribution<double> rd(0., 1.);
    double r = rd(g_mt);
    return r * b + (1.0 - r) * a;
}

inline double RandomGauss()
{
    return sqrt(2.0) * cos(2 * M_PI * RandomReal()) * sqrt(-log(RandomReal()));
}


// random Gamma variate for general shape parameter a > 0.0
inline double RandomGamma(const double a)
{
    D_gamma r(a, 1.0);
    return r(g_mt);
}

//////////////////////////////////////////////////////////////////////////////////
// Random Directions
//////////////////////////////////////////////////////////////////////////////////

inline Vector2 diskSample2D(const double radius)
{
    const double phi = RandomReal(0.0, 2 * M_PI);
    return radius * sqrt(RandomReal()) * Vector2(cos(phi), sin(phi));
}

inline Vector3 diskSample(const double radius)
{
    const double phi = RandomReal(0.0, 2 * M_PI);
    return radius * sqrt(RandomReal()) * Vector3(cos(phi), sin(phi), 0.0);
}

inline Vector3 isotropicDir()
{
    const double w = RandomReal(-1.0, 1.0);
    const double p = RandomReal(0.0, 2.0 * M_PI);
    const double s = sqrt(1.0 - w * w);
    return Vector3(w, s * cos(p), s * sin(p));
}

inline Vector3 lambertDir()
{
    const double w = sqrt(RandomReal());
    const double p = RandomReal(0.0, 2.0 * M_PI);
    const double s = sqrt(1.0 - w * w);
    return Vector3(s * cos(p), s * sin(p), w); // z axis is the normal
}

// sample a Lambertian direction about normal n:
inline Vector3 lambertDir(const Vector3 &n)
{
    const Vector3 local(lambertDir());
    Vector3 x, y;
    buildOrthonormalBasis(x, y, n);
    return x * local.x + y * local.y + n * local.z;
}
