##############################################################################
# Functions used to detect a change in the parameters of a SIRV distribution
# Authored by Ammar Mian, 28/09/2018
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2018 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import warnings
from generic_functions import *




##############################################################################
# Gaussian Statistics
##############################################################################
def Natural_Geodesic_statistic(X, Args):
    """ Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * Args = unused
        Outputs:
            * the statistic given the observations in input"""

    (p, N, T) = 𝐗.shape


    λ = 0
    for t_1 in range(T):
        𝚺_t_1 = SCM(𝐗[:, :, t_1])
        i𝚺_t_1_sqm = np.linalg.inv(sp.linalg.sqrtm(𝚺_t_1))
        for t_2 in range(T):
            if t_2 != t_1:
                𝚺_t_2 = SCM(𝐗[:, :, t_2])
                λ = λ + np.linalg.norm( sp.linalg.logm( i𝚺_t_1_sqm @ 𝚺_t_2 @ i𝚺_t_1_sqm ), 'fro')

    return np.real(λ)


def Wasserstein_Gaussian_statistic(X, Args):
    """ Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * Args = Unused
        Outputs:
            * the statistic given the observations in input"""

    (p, N, T) = 𝐗.shape

    λ = 0
    for t_1 in range(T):
        𝚺_t_1 = SCM(𝐗[:, :, t_1])
        𝚺_t_1_sqm = sp.linalg.sqrtm(𝚺_t_1)
        for t_2 in range(T):
            if t_2 != t_1:
                𝚺_t_2 = SCM(𝐗[:, :, t_2])
                λ = λ + np.trace(𝚺_t_1) + np.trace(𝚺_t_2) - 2*np.trace( sp.linalg.sqrtm(𝚺_t_1_sqm@𝚺_t_2@𝚺_t_1_sqm) )

    return np.real(λ)


def Kullback_Leibler_divergence(A, B):
    """ Inputs:
            * A, B = covariance matrices
        Outputs:
            * the divergence"""

    p = A.shape[0]
    λ = np.trace(np.linalg.inv(B)@A) + np.log(np.abs(np.linalg.det(B))) - np.log(np.abs(np.linalg.det(A))) - p
    return np.real(λ)


def Kullback_Leibler_statistic(𝐗, args=None):
    """ Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = Unused
        Outputs:
            * the Kullback-Leibler statistic"""

    (p, N, T) = 𝐗.shape
    λ = 0
    for t_1 in range(T):
        𝚺_t_1 = SCM(𝐗[:, :, t_1])
        for t_2 in range(T):
            if t_2 > t_1:
                𝚺_t_2 = SCM(𝐗[:, :, t_2])
                λ = λ +  0.5*(Kullback_Leibler_divergence(𝚺_t_1,𝚺_t_2) + Kullback_Leibler_divergence(𝚺_t_2,𝚺_t_1))
    return np.real(λ)


def complex_Hotelling_Lawley_trace_statistic(𝐗, args=None):
    """ V. Akbari, S. N. Anfinsen, A. P. Doulgeris, T. Eltoft, G. Moser and S. B. Serpico, 
        "Polarimetric SAR Change Detection With the Complex Hotelling–Lawley Trace Statistic," in IEEE Transactions on Geoscience and Remote Sensing, vol. 54, no. 7, pp. 3953-3966, July 2016.
        doi: 10.1109/TGRS.2016.2532320
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = Unused
        Outputs:
            * the CHLT statistic given the observations in input"""

    (p, N, T) = 𝐗.shape
    λ = 0
    for t_1 in range(T):
        𝚺_t_1 = SCM(𝐗[:, :, t_1])
        i𝚺_t_1 = np.linalg.inv(𝚺_t_1)
        for t_2 in range(T):
            if t_2 != t_1:
                𝚺_t_2 = SCM(𝐗[:, :, t_2])
                λ = λ + np.trace(i𝚺_t_1@𝚺_t_2)
    return np.real(λ)


def covariance_equality_glrt_gaussian_statistic(𝐗, args=None):
    """ GLRT statistic for detecting a change of covariance matrix in a multivariate Gaussian Time Series.
        At each time, Ni.i.d samples are available. A description of the statistic can be found in:
        D. Ciuonzo, V. Carotenuto and A. De Maio, 
        "On Multiple Covariance Equality Testing with Application to SAR Change Detection," 
        in IEEE Transactions on Signal Processing, vol. 65, no. 19, pp. 5078-5091, 1 Oct.1, 2017.
        doi: 10.1109/TSP.2017.2712124
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = None
        Outputs:
            * the GLRT statistic given the observations in input"""

    (p, N, T) = 𝐗.shape
    S = 0
    logDenominator = 0
    for t in range(0, T):
        St = SCM(𝐗[:, :, t])
        logDenominator = logDenominator + N * np.log(np.abs(np.linalg.det(St)))
        S = S + St / T
    logNumerator = N * T * np.log(np.abs(np.linalg.det(S)))
    if args is not None:
        if args=='log':
            return np.real(logNumerator - logDenominator)
    return np.exp(np.real(logNumerator - logDenominator))


def covariance_equality_t1_gaussian_statistic(𝐗, args=None):
    """ t1 statistic for detecting a change of covariance matrix in a multivariate Gaussian Time Series.
        At each time, Ni.i.d samples are available. A description of the statistic can be found in:
        D. Ciuonzo, V. Carotenuto and A. De Maio, 
        "On Multiple Covariance Equality Testing with Application to SAR Change Detection," 
        in IEEE Transactions on Signal Processing, vol. 65, no. 19, pp. 5078-5091, 1 Oct.1, 2017.
        doi: 10.1109/TSP.2017.2712124
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
        Outputs:
            * the t1 statistic given the observations in input"""

    (N, K, M) = 𝐗.shape

    Sigma_10 = SCM(𝐗.reshape((N, K*M)))
    iSigma_10 = np.linalg.inv(Sigma_10)
    t1 = 0
    for t in range(0, M):
        Sigma_m1 = SCM(𝐗[:, :, t])
        S = (iSigma_10 @ Sigma_m1)
        t1 = t1 + np.trace( S @ S )/M;

    if args is not None:
        if args=='log':
            return np.log(np.real(t1))
    return np.real(t1)


def covariance_equality_Wald_gaussian_statistic(𝐗, args=None):
    """ Wald statistic for detecting a change of covariance matrix in a multivariate Gaussian Time Series.
        At each time, Ni.i.d samples are available. A description of the statistic can be found in:
        D. Ciuonzo, V. Carotenuto and A. De Maio, 
        "On Multiple Covariance Equality Testing with Application to SAR Change Detection," 
        in IEEE Transactions on Signal Processing, vol. 65, no. 19, pp. 5078-5091, 1 Oct.1, 2017.
        doi: 10.1109/TSP.2017.2712124
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
        Outputs:
            * the Wald statistic given the observations in input"""

    (N, K, M) = 𝐗.shape
    L = 0
    O = 0
    Q = 0
    Sigma_11 = SCM(𝐗[:, :, 0])
    for m in range(0,M):
        Sigma_m1 = SCM(𝐗[:, :, m])
        iSigma_m1 = np.linalg.inv(Sigma_m1)
        if m != 0:
            S = np.eye(N) - Sigma_11@iSigma_m1
            L = L + K*np.trace(S@S)
            Q = Q + K*(iSigma_m1 - iSigma_m1@Sigma_11@iSigma_m1)
        O = O + K*np.kron(iSigma_m1.T, iSigma_m1)
    
    if args is not None:
        if args=='log':
            return np.log(np.real(L - vec(Q).conj().T @ (np.linalg.inv(O)@vec(Q))))
    return np.real(L - vec(Q).conj().T @ (np.linalg.inv(O)@vec(Q)))


##############################################################################
# Robust Statistics
##############################################################################

def MLL(X, tol, iter_max):
    """ Case L=1 """
    p, N = X.shape

    (𝚺, δ, iteration) = tyler_estimator_covariance(X, tol, iter_max)
    τ = np.abs(1/p * np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗))

    # Estimate Gamma parameters on τ
    ν, loc, μ = sp.stats.gamma.fit(τ, floc=0)

    temp = 0
    for i in range(N):
        temp = temp + np.log(sp.special.kv(ν-p, 2*np.sqrt(τ[i]*ν/μ)))



    result = N * (ν + p)/2 * (np.log(ν) - np.log(μ)) + \
            (ν-p)/2 * p * np.sum(τ) + temp - N*np.log(np.abs(np.linalg.det(𝚺))) - \
            N * np.log(sp.special.gamma(ν))
    return result
     


def Similarity_measure_Meng_Liu(X, Args):
    """ From M. Liu, H. Zhang, C. Wang and F. Wu, 
    "Change Detection of Multilook Polarimetric SAR Images Using Heterogeneous Clutter Models," 
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 52, no. 12, pp. 7483-7494, Dec. 2014.
    doi: 10.1109/TGRS.2014.2310451
        Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * Args = tol, iterMax for Tyler
        Outputs:
            * the statistic given the observations in input"""

    p, N, T = X.shape
    tol, iter_max = Args

    λ = 0
    for t_1 in range(T):
        for t_2 in range(T):
            if t_2 > t_1:
                λ = λ + MLL(X[:,:,t_1], tol, iter_max) + MLL(X[:,:,t_2], tol, iter_max) - \
                        MLL(np.hstack([X[:,:,t_1],X[:,:,t_2]]), tol, iter_max)

    return np.real(λ)



def Natural_student_t_distance(X, Args):
    """ Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * Args = d=degrees of freedom and tol, iterMax for MLE
        Outputs:
            * the statistic given the observations in input"""

    d, tol, iter_max = Args
    (p, N, T) = 𝐗.shape

    alpha = (d+p)/(d+p+1)
    beta = alpha - 1

    λ = 0
    for t_1 in range(T):
        (𝚺_t_1, δ, niter) = student_t_estimator_covariance_mle(𝐗[:,:,t_1], d, tol, iter_max)
        for t_2 in range(T):
            if t_2 != t_1:
                (𝚺_t_2, δ, niter) = student_t_estimator_covariance_mle(𝐗[:,:,t_2], d, tol, iter_max)
                w = sp.linalg.eig(𝚺_t_1, 𝚺_t_2, right=False)
                λ = λ + alpha*np.sum(np.log(w)**2) + beta*np.sum(np.log(w))**2

    return np.real(λ)




def Wasserstein_robust_statistic(X, Args):
    """ Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * Args = tol, iterMax for Tyler
        Outputs:
            * the statistic given the observations in input"""

    tol, iter_max = Args
    (p, N, T) = 𝐗.shape


    λ = 0
    for t_1 in range(T):
        (𝚺_t_1, δ, iteration) = tyler_estimator_covariance(𝐗[:,:,t_1], tol, iter_max)
        τ_t_1 = (1/p) * np.diagonal(𝐗[:,:,t_1].conj().T@np.linalg.inv(𝚺_t_1)@𝐗[:,:,t_1])
        𝚺_t_1 = np.mean(τ_t_1)*𝚺_t_1
        sqrtm_𝚺_t_1 = sp.linalg.sqrtm(𝚺_t_1)
        for t_2 in range(T):
            if t_2 != t_1:
                
                (𝚺_t_2, δ, iteration) = tyler_estimator_covariance(𝐗[:,:,t_2], tol, iter_max)                
                τ_t_2 = (1/p) * np.diagonal(𝐗[:,:,t_2].conj().T@np.linalg.inv(𝚺_t_2)@𝐗[:,:,t_2])
                𝚺_t_2 = np.mean(τ_t_2)*𝚺_t_2

                
                λ = λ + np.trace(𝚺_t_1) + np.trace(𝚺_t_2) - 2*np.trace( sp.linalg.sqrtm(sqrtm_𝚺_t_1@𝚺_t_2@sqrtm_𝚺_t_1) )

    return np.real(λ)



def student_t_glrt_statistic_d_known(X, Args):
    """ GLRT test for testing a change in the covariance of a multivariate
        Student-t distribution when the degree of freefom is known.
        Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * Args = d=degree of freedom and tol, iterMax for Tyler, scale
        Outputs:
            * the statistic given the observations in input"""

    d, tol, iter_max, scale = Args
    (p, N, T) = 𝐗.shape

    # Estimating 𝚺_0 using all the observations
    (𝚺_0, δ, niter) = student_t_estimator_covariance_mle(𝐗.reshape((p,T*N)), d, tol, iter_max)
    i𝚺_0 = np.linalg.inv(𝚺_0)

    # Some initialisation
    log_numerator_determinant_terms = T*N*np.log(np.abs(np.linalg.det(𝚺_0)))
    log_denominator_determinant_terms = 0
    log𝛕_0 = 0
    log𝛕_t = 0
    # Iterating on each date to compute the needed terms
    for t in range(0,T):
        # Estimating 𝚺_t
        (𝚺_t, δ, iteration) = student_t_estimator_covariance_mle(𝐗[:,:,t], d, tol, iter_max)

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
                                            N*np.log(np.abs(np.linalg.det(𝚺_t)))

        # Computing quadratic terms
        log𝛕_0 =  log𝛕_0 + np.log(d + np.diagonal(𝐗[:,:,t].conj().T@i𝚺_0@𝐗[:,:,t]))
        log𝛕_t = log𝛕_t + np.log(d + np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_t)@𝐗[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = (d+p)*np.sum(log𝛕_0)
    log_denominator_quadtratic_terms = (d+p)*np.sum(log𝛕_t)

    # Final expression of the statistic
    if scale=='linear':
        λ = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        λ = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    return λ


def scale_and_shape_equality_robust_statistic(𝐗, args):
    """ GLRT test for testing a change in the scale or/and shape of 
        a deterministic SIRV model.
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = tol, iter_max for Tyler, scale
        Outputs:
            * the statistic given the observations in input"""

    tol, iter_max, scale = args
    (p, N, T) = 𝐗.shape

    # Estimating 𝚺_0 using all the observations
    (𝚺_0, δ, niter) = tyler_estimator_covariance_matandtext(𝐗, tol, iter_max)
    i𝚺_0 = np.linalg.inv(𝚺_0)

    # Some initialisation
    log_numerator_determinant_terms = T*N*np.log(np.abs(np.linalg.det(𝚺_0)))
    log_denominator_determinant_terms = 0
    𝛕_0 = 0
    log𝛕_t = 0
    # Iterating on each date to compute the needed terms
    for t in range(0,T):
        # Estimating 𝚺_t
        (𝚺_t, δ, iteration) = tyler_estimator_covariance(𝐗[:,:,t], tol, iter_max)

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
                                            N*np.log(np.abs(np.linalg.det(𝚺_t)))

        # Computing texture estimation
        𝛕_0 =  𝛕_0 + np.diagonal(𝐗[:,:,t].conj().T@i𝚺_0@𝐗[:,:,t]) / T
        log𝛕_t = log𝛕_t + np.log(np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_t)@𝐗[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = T*p*np.sum(np.log(𝛕_0))
    log_denominator_quadtratic_terms = p*np.sum(log𝛕_t)

    # Final expression of the statistic
    if scale=='linear':
        λ = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        λ = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    return λ


def shape_equality_robust_statistic(𝐗, args):
    """ GLRT test for testing a change in the shape of 
        a deterministic SIRV model.
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = tol, iter_max for Tyler, scale
        Outputs:
            * the statistic given the observations in input"""

    tol, iter_max, scale = args
    (p, N, T) = 𝐗.shape

    # Estimating 𝚺_0 using all the observations
    (𝚺_0, δ, niter) = tyler_estimator_covariance(𝐗.reshape((p,T*N)), tol, iter_max)
    i𝚺_0 = np.linalg.inv(𝚺_0)

    # Some initialisation
    log_numerator_determinant_terms = T*N*np.log(np.abs(np.linalg.det(𝚺_0)))
    log_denominator_determinant_terms = 0
    log𝛕_0 = 0
    log𝛕_t = 0
    # Iterating on each date to compute the needed terms
    for t in range(0,T):
        # Estimating 𝚺_t
        (𝚺_t, δ, iteration) = tyler_estimator_covariance(𝐗[:,:,t], tol, iter_max)

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
                                            N*np.log(np.abs(np.linalg.det(𝚺_t)))

        # Computing texture estimation
        log𝛕_0 =  log𝛕_0 + np.log(np.diagonal(𝐗[:,:,t].conj().T@i𝚺_0@𝐗[:,:,t]))
        log𝛕_t = log𝛕_t + np.log(np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_t)@𝐗[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = p*np.sum(log𝛕_0)
    log_denominator_quadtratic_terms = p*np.sum(log𝛕_t)

    # Final expression of the statistic
    if scale=='linear':
        λ = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        λ = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    return λ


def scale_equality_robust_statistic(𝐗, args):
    """ GLRT test for testing a change in the scale of 
        a deterministic SIRV model.
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = tol, iter_max for Tyler, scale
        Outputs:
            * the statistic given the observations in input"""

    tol, iter_max, scale = args
    (p, N, T) = 𝐗.shape

    # Estimating 𝚺_t under H0 regime using all the observations
    (𝚺_0, 𝛅, niter) = tyler_estimator_covariance_text(𝐗, tol, iter_max)

    # Some initialisation
    log_numerator_determinant_terms = 0
    log_denominator_determinant_terms = 0
    𝛕_0 = 0
    log𝛕_t = 0
    # Iterating on each date to compute the needed terms
    for t in range(0,T):

        # Estimating 𝚺_t under H1 regime
        (𝚺_t, δ, iteration) = tyler_estimator_covariance(𝐗[:,:,t], tol, iter_max)

        # Computing determinant add adding it to log_numerator_determinant_terms
        log_numerator_determinant_terms = log_numerator_determinant_terms + \
                                        N*np.log(np.abs(np.linalg.det(𝚺_0[:,:,t])))

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
                                            N*np.log(np.abs(np.linalg.det(𝚺_t)))

        # Computing texture estimation
        𝛕_0 =  𝛕_0 + np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_0[:,:,t])@𝐗[:,:,t]) / T
        log𝛕_t = log𝛕_t + np.log(np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_t)@𝐗[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = T*p*np.sum(np.log(𝛕_0))
    log_denominator_quadtratic_terms = p*np.sum(log𝛕_t)

    # Final expression of the statistic
    if scale=='linear':
        λ = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        λ = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    return λ

##############################################################################
# Some Functions
##############################################################################
def tyler_estimator_covariance_matandtext(𝐗, tol=0.0001, iter_max=20):
    """ A function that computes the Modified Tyler Fixed Point Estimator for 
    covariance matrix estimation under problem MatAndText.
        Inputs:
            * 𝐗 = a matrix of size p*N*T with each saptial observation along column dimension and time
                observation along third dimension.
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    (p, N, T) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and iteration < iter_max:

        # Compute the textures for each pixel using all the dates avalaibe
        τ = 0
        i𝚺 = np.linalg.inv(𝚺)
        for t in range(0, T):
            τ = τ + np.diagonal(𝐗[:,:,t].conj().T@i𝚺@𝐗[:,:,t])

        # Computing expression of the estimator
        𝚺_new = 0
        for t in range(0, T):
            𝐗_bis = 𝐗[:,:,t] / np.sqrt(τ)
            𝚺_new = 𝚺_new + (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p*𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')

        # Updating 𝚺
        𝚺 = 𝚺_new
        iteration = iteration + 1

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)


def tyler_estimator_covariance_text(𝐗, tol=0.0001, iter_max=20):
    """ A function that computes the Modified Tyler Fixed Point Estimator for
    covariance matrix estimation under problem TextGen.
        Inputs:
            * 𝐗 = a matrix of size p*N*T with each saptial observation along column dimension and time
                observation along third dimension.
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = array of size (p,p,T) to the different estimates
            * 𝛅 = the final distance between two iterations for each estimate
            * iteration = number of iterations til convergence """

    (p, N, T) = 𝐗.shape
    𝛅 = np.inf*np.ones(T) # Distance between two iterations for each t
    𝚺 = np.tile(np.eye(p).reshape(p,p,1), (1,1,T)) # Initialise all estimates to identity
    iteration = 0

    # Recursive algorithm
    while (np.max(𝛅)>tol) and iteration < iter_max:

        # Compute the textures for each pixel using all the dates avalaibe
        τ = 0
        for t in range(0, T):
            i𝚺_t = np.linalg.inv(𝚺[:,:,t])
            τ = τ + np.diagonal(𝐗[:,:,t].conj().T@i𝚺_t@𝐗[:,:,t])

        # Computing expression of the estimator
        𝚺_new = np.zeros((p,p,T)).astype(complex)
        for t in range(0, T):
            𝐗_bis = 𝐗[:,:,t] / np.sqrt(τ)
            𝚺_new[:,:,t] = (T*p/N) * 𝐗_bis@𝐗_bis.conj().T

            # Imposing trace constraint: Tr(𝚺) = p
            𝚺_new[:,:,t] = p*𝚺_new[:,:,t]/np.trace(𝚺_new[:,:,t])

            # Condition for stopping
            𝛅[t] = np.linalg.norm(𝚺_new[:,:,t] - 𝚺[:,:,t], 'fro') / \
                     np.linalg.norm(𝚺[:,:,t], 'fro')
        
        # Updating 𝚺
        𝚺 = 𝚺_new
        iteration = iteration + 1

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, 𝛅, iteration)