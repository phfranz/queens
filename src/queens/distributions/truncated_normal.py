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
"""Normal distribution."""

import numpy as np
import scipy.linalg
import scipy.stats

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args
from queens.utils.numpy_array import at_least_2d
from queens.utils.numpy_linalg import safe_cholesky


class TruncatedNormal(Continuous):
    """1D - Truncated Normal distribution.

    Attributes:
        low_chol (np.ndarray): Lower-triangular Cholesky factor of covariance matrix.
        precision (np.ndarray): Precision matrix corresponding to covariance matrix.
        logpdf_const (float): Constant for evaluation of log pdf.
    """

    @log_init_args
    def __init__(self, mean, covariance, a_trunc=0., b_trunc=1.):
        """Initialize univariate truncated normal distribution.

        Args:
            mean (array_like): mean of the distribution
            covariance (array_like): covariance of the distribution
            a (float): lower bound
            b (float): upper bound
        """
        #mean = np.array(mean).reshape(-1)
       
        if np.isscalar(mean) is False:
            raise ValueError(
                f"Provided mean is not a scalar. "
                f"Provided mean shape: {mean.shape}"
            )

        if np.isscalar(covariance) is False:
            raise ValueError(
                f"Provided mean is not a scalar. "
                f"Provided mean shape: {covariance.shape}"
            )
        
        dimension = 1

        #low_chol, precision, logpdf_const = self._calculate_distribution_parameters(covariance)
        super().__init__(mean, covariance, dimension)
        self.mean = mean
        self.precision = 1/covariance
        self.a = (a_trunc - self.mean) / np.sqrt(1/self.precision)
        self.b = (b_trunc - self.mean) / np.sqrt(1/self.precision)

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): cdf at evaluated positions
        """
        # cdf = scipy.stats.truncnorm.cdf(
        #     x.reshape(-1, self.dimension), a=self.a, b=self.b, mean=self.mean, cov=self.covariance
        # ).reshape(-1)
        # return cdf

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        # uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        # samples = self.mean + np.dot(self.low_chol, uncorrelated_vector).T
        # return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        #dist = x.reshape(-1, self.dimension) - self.mean

        # TODO: check if this works for multiple chains

        #x = np.tile(x, self.dimension//x.shape[1]).reshape(-1, self.dimension)

        logpdf = scipy.stats.truncnorm.logpdf(x, self.a, self.b, self.mean, np.sqrt(1/self.precision))

        #log_sum = logpdf.sum(axis=-1)
        
        return logpdf


    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        # x = x.reshape(-1, self.dimension)
        # grad_logpdf = np.dot(self.mean.reshape(1, -1) - x, self.precision)
        # return grad_logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        # self.check_1d()
        # ppf = scipy.stats.norm.ppf(
        #     quantiles, loc=self.mean, scale=self.covariance ** (1 / 2)
        # ).reshape(-1)
        # return ppf

    def update_covariance(self, covariance):
        """Update covariance and dependent distribution parameters.

        Args:
            covariance (np.ndarray): Covariance matrix
        """
        # low_chol, precision, logpdf_const = self._calculate_distribution_parameters(covariance)
        # self.covariance = covariance
        # self.low_chol = low_chol
        # self.precision = precision
        # self.logpdf_const = logpdf_const

    @staticmethod
    def _calculate_distribution_parameters(covariance):
        """Calculate covariance dependent distribution parameters.

        Args:
            covariance (np.ndarray): Covariance matrix

        Returns:
            low_chol (np.ndarray): lower-triangular Cholesky factor of covariance matrix
            precision (np.ndarray): Precision matrix corresponding to covariance matrix
            logpdf_const (float): Constant for evaluation of log pdf
        """
        # dimension = covariance.shape[0]
        # low_chol = safe_cholesky(covariance)

        # # precision matrix Q and determinant of cov matrix
        # chol_inv = np.linalg.inv(low_chol)
        # precision = np.dot(chol_inv.T, chol_inv)

        # # constant needed for pdf
        # logpdf_const = -1 / 2 * (np.log(2.0 * np.pi) * dimension + np.linalg.slogdet(covariance)[1])
        # return low_chol, precision, logpdf_const
