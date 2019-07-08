##############################################################################
# Some General use functions
# Authored by Ammar Mian, 17/06/2018
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
import scipy.special
import warnings

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def enable_latex_infigures():
    """ A function that allows to enable latex in figures"""
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    (p, p) = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = np.random.multivariate_normal(mu, Gamma_2r, N).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y


def multivariate_complex_t_samples(mean, covariance, N, df, pseudo_covariance=0):
    """ A function to generate multivariate complex t distributed vectors using the
    definition with a product of a multivaraite normal with an inverse chi2 distributed samples. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * df = degrees of freedom of the chi-squared distribution
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""


    if df == np.inf:
        x = 1
    else:
        x = np.random.chisquare(df, N)/df
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z/np.sqrt(x)[None,:] 


def multivariate_complex_K_samples(mean, covariance, N, mu, b, pseudo_covariance=0):
    """ A function to generate multivariate complex K distributed vectors using the
    definition provided at page 27 of the Pd.d thesis:
    "Detection en environement non Gaussien", Emanuelle Jay. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * mu = Shape parameter
        * b = Scale parameter
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""

    x = np.random.gamma(mu, 2/(b**2), N)
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z*np.sqrt(x)[None,:]   


def multivariate_complex_Cauchy_samples(mean, covariance, N, mu, b, pseudo_covariance=0):
    """ A function to generate multivariate complex Cauchy distributed vectors using the
    definition provided at page 26 of the Pd.d thesis:
    "Detection en environement non Gaussien", Emanuelle Jay. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * mu = Shape parameter
        * b = Scale parameter
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""

    x = np.random.gamma(mu, 2/(b**2), N)
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z/np.sqrt(x)[None,:]    


def multivariate_complex_Laplace_samples(mean, covariance, N, beta, pseudo_covariance=0):
    """ A function to generate multivariate complex Cauchy distributed vectors using the
    definition provided at page 27 of the Pd.d thesis:
    "Detection en environement non Gaussien", Emanuelle Jay. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * beta = Scale parameter
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""

    x = np.random.exponential(beta, N)
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z*np.sqrt(x)[None,:]    


def SCM(x, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""

    (p, N) = x.shape
    return (x @ x.conj().T) / N


def tyler_estimator_covariance(𝐗, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p*𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)


def tyler_estimator_covariance_normalisedet(𝐗, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # # Imposing det constraint: det(𝚺) = 1 DOT NOT WORK HERE
        # 𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    # Imposing det constraint: det(𝚺) = 1
    𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)


def student_t_estimator_covariance_mle(𝐗, d, tol=0.001, iter_max=20):
    """ A function that computes the MLE for covariance matrix estimation for a student t distribution
        when the degree of freedom is known
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = d + np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = ((d+p)/N) * 𝐗_bis@𝐗_bis.conj().T

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)

def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows), dtype=v.dtype)
    result[np.triu_indices(rows)] = v
    result = result + result.conj().T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result


def plot_Pauli_SAR(image, figsize=(9,6)):
    import matplotlib.pyplot as plt
    """ 1st dimension =HH, 2nd dimnension = HV, 3rd dimension=VV"""
    R = np.abs(image[:,:,0] - image[:,:,2])
    G = np.abs(image[:,:,1])
    B = np.abs(image[:,:,0] + image[:,:,2])

    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w')
    plt.imshow(np.stack([R,G,B], axis=2), aspect='auto')
    plt.title(r'Image at $t=%d$')
    plt.axis('off')
    return fig


def plot_sar_image_span_plotly(image, colormap='Greys'):

    import plotly.graph_objs as go
    from plotly.offline import iplot

    if len(image.shape)>2:
        trace = go.Heatmap(z=20*np.log10(np.sum(np.abs(image)**2, axis=2)), 
                   colorscale=colormap)
    else:
        trace = go.Heatmap(z=20*np.log10(np.abs(image)**2), 
                   colorscale=colormap)

    data=[trace]
    fig = go.Figure(data=data)
    iplot(fig)
    return fig


