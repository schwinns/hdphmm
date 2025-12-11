# Functions to compute the likelihood of a trajectory given a fitted HDP-HMM model

import numpy as np

def compute_likelihood(ihmm, iter, trajectory):
    """Compute the likelihood on a trajectory given the fitted model.
    
    Parameters
    ----------
    ihmm : InfiniteHMM
        Fitted HDP-HMM model
    iter : int
        Iteration number of the fitted model to use
    trajectory : np.ndarray
        Trajectory data, shape (T, dimensions) where T is the number of timesteps
        and dimensions matches the model's dimensionality
        
    Returns
    -------
    likelihood : np.ndarray
        Likelihood of being in each state at each time point, shape (max_states, Ks, T)
    log_likelihood_normalized : np.ndarray
        Log likelihood normalized for numerical stability, same shape as likelihood
    """
    
    if ihmm.observation_model != 'AR':
        raise TypeError('compute_likelihood_new_data only implemented for AR observation model')
    
    # Validate input dimensions
    if trajectory.ndim != 2:
        raise ValueError(f"Expected 2D array (T, dimensions), got shape {trajectory.shape}")
    if trajectory.shape[1] != ihmm.dimensions:
        raise ValueError(f"Expected {ihmm.dimensions} dimensions, got {trajectory.shape[1]}")
    
    # Create design matrix from new data
    X_new = ihmm.make_design_matrix(trajectory)
    trajectory = trajectory[ihmm.order:, ...]
    
    # Get parameters
    invSigma = ihmm.convergence['invSigma'][iter]
    A = ihmm.convergence['A'][iter]

    if len(ihmm.convergence['mu']) > 0:
        mu = ihmm.convergence['mu'][iter]
    else:
        mu = ihmm.theta['mu']
    
    dimu = trajectory.shape[1]
    T = X_new.shape[1]
    
    log_likelihood = np.zeros([ihmm.max_states, ihmm.Ks, T])
    
    for kz in range(ihmm.max_states):
        for ks in range(ihmm.Ks):
            cholinvSigma = np.linalg.cholesky(invSigma[:, :, kz, ks]).T
            dcholinvSigma = np.diag(cholinvSigma)
            
            # Compute residuals: y - A @ X - mu
            v = trajectory.T - A[:, :, kz, ks] @ X_new - mu[:, kz * np.ones([T], dtype=int), ks]
            
            # Transform by Cholesky factor
            u = cholinvSigma @ v
            
            # Compute log likelihood
            log_likelihood[kz, ks, :] = -0.5 * np.square(u).sum(axis=0) + np.log(dcholinvSigma).sum()
    
    # Normalize for numerical stability
    normalizer = log_likelihood.max(axis=0).max(axis=0)
    log_likelihood_normalized = log_likelihood - normalizer
    likelihood = np.exp(log_likelihood_normalized)
    
    return likelihood, log_likelihood_normalized


def forward_backward(ihmm, iter, trajectory):
    """Run forward-backward algorithm on new data to get state likelihoods.
    
    Parameters
    ----------
    ihmm : InfiniteHMM
        Fitted HDP-HMM model
    iter : int
        Iteration number of the fitted model to use
    trajectory : np.ndarray
        Trajectory data, shape (T, dimensions)
        
    Returns
    -------
    log_likelihood : float
        Total log likelihood of the trajectory given the model
    forward_messages : np.ndarray
        Forward messages (alpha), shape (max_states, T)
    backward_messages : np.ndarray
        Backward messages (beta), shape (max_states, T)
    posteriors : np.ndarray
        Posterior state probabilities P(z_t | x_{1:T}), shape (max_states, T)
    """
    
    # Compute likelihoods for new data
    likelihood, _ = compute_likelihood(ihmm, iter, trajectory)
    block_like = likelihood.sum(axis=1)  # sum over substates: (max_states, T)
    
    T = block_like.shape[1]
    max_states = block_like.shape[0]

    # get parameters for iteration
    pi_init = ihmm.convergence['pi_init'][iter]
    pi_z = ihmm.convergence['T'][iter]
    
    # Forward pass
    forward_messages = np.zeros([max_states, T])
    forward_messages[:, 0] = pi_init * block_like[:, 0]
    forward_messages[:, 0] /= forward_messages[:, 0].sum()
    
    log_likelihood_forward = 0
    
    for t in range(1, T):
        # Compute transition weighted by likelihood
        forward_messages[:, t] = (pi_z.T @ forward_messages[:, t-1]) * block_like[:, t]
        
        # Normalize and accumulate log likelihood
        norm = forward_messages[:, t].sum()
        log_likelihood_forward += np.log(norm) if norm > 0 else -np.inf
        forward_messages[:, t] /= norm
    
    # Backward pass
    backward_messages = np.ones([max_states, T])
    
    for t in range(T - 2, -1, -1):
        backward_messages[:, t] = (pi_z @ (block_like[:, t+1] * backward_messages[:, t+1]))
        # Normalize to avoid numerical issues
        norm = backward_messages[:, t].sum()
        if norm > 0:
            backward_messages[:, t] /= norm
    
    # Compute posterior marginals
    posteriors = forward_messages * backward_messages
    posteriors /= posteriors.sum(axis=0)  # normalize across states
    
    # Total log likelihood
    log_likelihood_total = log_likelihood_forward + np.log(block_like[:, 0].sum())
    
    return log_likelihood_total, forward_messages, backward_messages, posteriors