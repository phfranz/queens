#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Gaussian likelihood."""

import warnings

import numpy as np
import scipy as sp

from queens.distributions.normal import Normal
from queens.distributions.truncated_normal import TruncatedNormal
from queens.models.likelihoods._likelihood import Likelihood
from queens.utils.exceptions import InvalidOptionError
from queens.utils.logger_settings import log_init_args
from queens.utils.numpy_linalg import add_nugget_to_diagonal


class CustomProduct(Likelihood):
    r"""Gaussian likelihood model with fixed or dynamic noise.

    The noise can be modelled by a full covariance matrix, independent variances or a unified
    variance for all observations. If the noise is chosen to be dynamic, a MAP estimate of the
    covariance, independent variances or unified variance is computed using a Jeffrey's prior.
    Jeffrey's prior is defined as :math:`\pi_J(\Sigma) = |\Sigma|^{-(p+2)/2}`, where :math:`\Sigma`
    is the covariance matrix of shape :math:`p \times p` (see [1])

    References:
        [1]: Sun, Dongchu, and James O. Berger. "Objective Bayesian analysis for the multivariate
             normal model." Bayesian Statistics 8 (2007): 525-562.

    Attributes:
        nugget_noise_variance (float): Lower bound for the likelihood noise parameter
        noise_type (str): String encoding the type of likelihood noise model:
                                     Fixed or MAP estimate with Jeffreys prior
        noise_var_iterative_averaging (obj): Iterative averaging object
        normal_distribution (obj): Underlying normal distribution object

    Returns:
        Instance of Gaussian Class
    """

    @log_init_args
    def __init__(
        self,
        forward_model,
        n_observations,
        n_nf,
        n_sensors,
        noise_type,
        noise_value_nf=None,
        noise_value_ms=None,
        nugget_noise_variance=0,
        noise_var_iterative_averaging=None,
        y_obs=None,
        experimental_data_reader=None,
    ):

        """Initialize likelihood model.

        Args:
            forward_model (obj): Forward model on which the likelihood model is based
            noise_type (str): String encoding the type of likelihood noise model:
                                Fixed or MAP estimate with Jeffreys prior
            noise_value (array_like): Likelihood (co)variance value
            nugget_noise_variance (float): Lower bound for the likelihood noise parameter
            noise_var_iterative_averaging (obj): Iterative averaging object
            y_obs (array_like): Vector with observations
            experimental_data_reader (obj): Experimental data reader
        """
        # if y_obs is not None and experimental_data_reader is not None:
        #     warnings.warn(
        #         "You provided 'y_obs' and 'experimental_data_reader' to Gaussian. "
        #         "Only provided 'y_obs' is used."
        #     )
        # if y_obs is None:
        #     if experimental_data_reader is None:
        #         raise InvalidOptionError(
        #             "You must either provide 'y_obs' or an "
        #             "'experimental_data_reader' for Gaussian."
        #         )
        #     y_obs = experimental_data_reader.get_experimental_data()[0]

        super().__init__(forward_model, y_obs)

        y_obs_1_dim = y_obs[:,:n_observations].size
        #y_obs_2_dim = y_obs[:,n_observations:].size

        if noise_value_nf is None and noise_type.startswith("fixed"):
            raise InvalidOptionError(f"You have to provide a 'noise_value' for {noise_type}.")

        if noise_type == "fixed_variance":
            covariance = noise_value_nf * np.eye(y_obs_1_dim)
        elif noise_type == "fixed_variance_vector":
            covariance = np.diag(noise_value_nf)
        elif noise_type == "fixed_covariance_matrix":
            covariance = noise_value_nf
        elif noise_type in [
            "MAP_jeffrey_variance",
            "MAP_jeffrey_variance_vector",
            "MAP_jeffrey_covariance_matrix",
        ]:
            covariance = np.eye(y_obs_1_dim)
        else:
            raise NotImplementedError

        self.cov = covariance
        self.nugget_noise_variance = nugget_noise_variance
        self.noise_type = noise_type
        self.noise_value_TN = noise_value_ms
        self.noise_var_iterative_averaging = noise_var_iterative_averaging
        self.number_eigenfrequencies = n_nf
        self.n_sensors = n_sensors
        self.n_observations = n_observations
        self.noise_std_ms = np.sqrt(noise_value_ms)*(1+1/np.sqrt(self.n_sensors)) # treated as standard deviation here!

        #normal_distribution = Normal(self.y_obs[:,:n_observations].T, covariance) # use here the transposed so that the "mean value vector" contains stacked pairs of observations
        #die observations können hier einfach als mean genommen werden 
        # anders als in der Formel, denn die Subtraktion wird quadriert, oder?
        truncated_normal_distribution = TruncatedNormal(1, self.noise_value_TN*(1+1/np.sqrt(n_sensors)), a_trunc=0.0, b_trunc=1.0)
        
        #self.normal_distribution = normal_distribution
        self.trunc_normal_distribution = truncated_normal_distribution

    def _computeMAC(self, vector1, vector2):
        """
        Compute the Modal Assurance Criterion between two mode shape vectors.
        
        If vector1 is a 2D array, compute MAC between each row of vector1 and vector2,
        returning a 1D array of MAC values.
        
        Parameters:
        - vector1: 1D or 2D numpy array (if 2D, shape = (n_vectors, vector_length))
        - vector2: 1D numpy array
        
        Return: MAC value(s) as float (if vector1 is 1D) or 1D numpy array (if vector1 is 2D)
        """

        vector2_norm = np.linalg.norm(vector2)
    
        if vector1.ndim == 1:
            # both vectors
            return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * vector2_norm)
        
        elif vector1.ndim == 2:
            # matrix and vector
            vector1_norms = np.linalg.norm(vector1, axis=1)
            dot_products = np.dot(vector1, vector2)
            return dot_products / (vector1_norms * vector2_norm)
                    
        else:
            raise ValueError("vector1 must be either 1D or 2D numpy array")

        
    def _evaluate(self, samples):
        """Evaluate likelihood with current set of input samples.

        Args:
            samples (np.array): Input samples

        Returns:
            dict: log-likelihood values at input samples
        """
        self.response = self.forward_model.evaluate(samples) # result.shape_ 10x96: 10 chains with 8 ef + 11 sensors x 8 mode shapes = 96 elements

        # --- Evaluate first part of log-likelihood --- 

        if self.noise_type.startswith("MAP"):
            self.update_covariance(self.response["result"])

        #Gaussian_log_probabilities = self.normal_distribution.logpdf(self.response["result"][:,:self.number_eigenfrequencies]) # should be of shape n_chains x 1

        log_likelihood = np.zeros([len(samples)]) #should give an array of size: n_chains

        # iterate over chains
        for n in range(len(samples)): 
            
            # such that number of variable in multivariate normal distribution fits the number of features per sample.
            log_likelihood[n] += sp.stats.multivariate_normal.logpdf(self.y_obs[:,:self.n_observations].T, mean=self.response["result"][n,0:8], cov=self.cov).sum(axis=0)
     
        # --- Evaluate second part of log-likelihood --- 

        mode_shapes_predicted = self.response["result"][:,self.number_eigenfrequencies:] #shape: 10 chains x 88 modal coordinates (= 8 mode shapes per chain à 11 sensors)

        # compute MAC between predicted and measured mode shapes

        truncated_Gaussian_log_probabilities = np.zeros([samples.shape[0]]) #1D truncated Gaussian probabilities
        
        for obs in range(self.n_observations):

            for shape in range(self.number_eigenfrequencies):

                vec_1 = mode_shapes_predicted[:, shape*self.n_sensors:(shape+1)*self.n_sensors] #iterate over predicted shapes, shape: n_chains x n_sensors

                vec_2 = self.y_obs[shape, self.n_observations+obs*self.n_sensors:self.n_observations+(obs+1)*self.n_sensors] #iterate over data matrix of y_obs

                mac = self._computeMAC(vec_1, vec_2) # should be of shape n_chains x 1

                # compute now probability that MAC is at 1, given the model parameters
                
                a, b = (0 - mac) / self.noise_std_ms, (1 - mac) / self.noise_std_ms

                truncated_Gaussian_log_probabilities+= sp.stats.truncnorm.logpdf(1, a, b, mac, self.noise_std_ms) # should be of shape n_chains x 1

        #test = 10*truncated_Gaussian_log_probabilities

        return {"result": log_likelihood + 5*truncated_Gaussian_log_probabilities}


    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        # shape convention: num_samples x jacobian_shape
        log_likelihood_grad = self.normal_distribution.grad_logpdf(self.response["result"])
        upstream_gradient = upstream_gradient * log_likelihood_grad
        gradient = self.forward_model.grad(samples, upstream_gradient)
        return gradient

    def update_covariance(self, y_model):
        """Update covariance matrix of the gaussian likelihood.

        Args:
            y_model (np.ndarray): Forward model output with shape (samples, outputs)
        """
        dist = y_model - self.y_obs.reshape(1, -1)
        num_samples, dim_y = y_model.shape
        if self.noise_type == "MAP_jeffrey_variance":
            covariance = np.eye(dim_y) / (dim_y * (num_samples + dim_y + 2)) * np.sum(dist**2)
        elif self.noise_type == "MAP_jeffrey_variance_vector":
            covariance = np.diag(1 / (num_samples + dim_y + 2) * np.sum(dist**2, axis=0))
        else:
            covariance = 1 / (num_samples + dim_y + 2) * np.dot(dist.T, dist)

        # If iterative averaging is desired
        if self.noise_var_iterative_averaging:
            covariance = self.noise_var_iterative_averaging.update_average(covariance)

        covariance = add_nugget_to_diagonal(covariance, self.nugget_noise_variance)
        self.normal_distribution.update_covariance(covariance)
