import sys
import glob
import numpy as np
import skimage
import pdb

from sklearn.mixture import GMM

def _gradients(descriptors, n_modes, covariance_type, random_state,
               tol, min_covar, n_iter, n_init, params, init_params, verbose):

    n_descriptors = descriptors.shape[0]
    gmm = GMM(n_components=n_modes, covariance_type=covariance_type,
              random_state=random_state, tol=tol,
              min_covar=min_covar, n_iter=n_iter, n_init=n_init, params=params,
              init_params=init_params, verbose=verbose)

    gmm.fit(descriptors)

    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covars_

    posterior_prob = gmm.predict_proba(descriptors)
    descriptors_tiled = np.tile(descriptors, (n_modes, 1, 1)).transpose((1, 0, 2))

    s0 = posterior_prob.sum(axis=0)
    s1 = posterior_prob[:, :, np.newaxis] * descriptors_tiled
    s1 = s1.sum(axis=0)
    s2 = posterior_prob[:, :, np.newaxis] * (descriptors_tiled ** 2)
    s2 = s2.sum(axis=0)

    grad_alpha = (s0 - n_descriptors * weights) / np.sqrt(weights)
    s0 = s0[:, np.newaxis]
    weights = weights[:, np.newaxis]
    grad_mu = (s1 - means * s0) / (np.sqrt(weights) * covars)

    grad_sigma = (s2 - 2 * means * s1 + (means ** 2 - covars ** 2) * s0) / \
                 (np.sqrt(2 * weights) * (covars ** 2))

    return grad_alpha, grad_mu, grad_sigma


def _power_normalization(fisher):
    normalized = np.sign(fisher) * np.sqrt(np.abs(fisher))
    return normalized


def _l2_normalization(fisher):
    normalized = fisher / np.sqrt(np.dot(fisher, fisher))
    return normalized


def fisher_vector(descriptors, n_modes=1, covariance_type='diag',
                  random_state=None, tol=0.001, min_covar=0.001, n_iter=100,
                  n_init=1, params='wmc', init_params='wmc', verbose=0,
                  normalize=True):

    grad_alpha, grad_mu, grad_sigma = _gradients(descriptors,
                                                 n_modes=n_modes,
                                                 covariance_type=covariance_type,
                                                 random_state=random_state,
                                                 tol=tol,
                                                 min_covar=min_covar,
                                                 n_iter=n_iter,
                                                 n_init=n_init,
                                                 params='wmc',
                                                 init_params='wmc',
                                                 verbose=verbose)

    grad_mu = np.ravel(grad_mu)
    grad_sigma =np.ravel(grad_sigma)
    fisher = np.hstack([grad_alpha, grad_mu, grad_sigma])

    assert fisher.shape[0] == n_modes * (2 * descriptors.shape[1] + 1)

    if normalize:
        fisher = _power_normalization(fisher)
        fisher = _l2_normalization(fisher)

    return fisher







