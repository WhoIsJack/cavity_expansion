# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:20:21 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  Simulation engine for simple_cell_sim.
"""


# Imports
import warnings
import numpy as np

from simple_cell_sim import force_funcs


def get_dists(pos):
    """Get pairwise Euclidean and vector distances between cells.

    Parameters
    ----------
    pos : numpy array of shape (n_cells, 2[yx])
        2D (y, x) coordinates of cell locations.
    
    Returns
    -------
    x_dist : numpy array of shape (n_cells, n_cells)
        Pairwise vector distances along x axis.
    y_dist : numpy array of shape (n_cells, n_cells)
        Pairwise vector distances along y axis.
    dist : numpy array of shape (n_cells, n_cells)
        Pairwise Euclidean distances.
    """
    
    # Get vector distance
    x_dist = pos[:,1] - pos[:,1][:,None]
    y_dist = pos[:,0] - pos[:,0][:,None] 
    
    # Get Euclidian distance
    dist = np.sqrt(x_dist**2.0 + y_dist**2.0)
    
    # Return calculated results
    return x_dist, y_dist, dist


def timestep(pos, force_terms, delta_t):
    """Main simulation function that executes a time step and returns
    the new positions.
    
    This function handles the following steps:
    1. Get pairwise distances between cell positions
    2. Compute forces based on one or several force terms (See below)
    3. Update cell positions based on summed force terms and delta_t

    A force term is a list containing the following 7 components:

    force_func : callable(dist, *force_params)
        Function that computes the pairwise forces between cells based
        on their pairwise distances and other parameters.
    force_params : list
        List of force_params to be passed to force_func after dist.
    min_range : float
        Force is set to zero for distances smaller than this value.
    max_range : float
        Force is et to zero for distances larger than this value.
    state_mask : None, or numpy array, shape (n_cells, n_cells), type bool
        Interaction matrix for cells; forces between cells that are False
        in this mask are set to zero. Useful when specifying different force
        terms, each of which should affect only a particular pairing of
        different cell states/types. Ignored if None.
    rnd_stdev : None, or float
        Standard deviation for Gaussian random forces to add as noise
        to the forces computed with this force term. No noise is added
        if this is None.
    rnd_bound : None, or float
        The Gaussian random forces generated based on rnd_stdev will be
        bounded within (-rnd_bound, rnd_bound). The Gaussian distribution
        is unbounded if this is None.
    
    Parameters
    ----------
    pos : numpy array of shape (n_cells, 2[yx])
        2D (y, x) input coordinates of cell locations.
    force_terms : list
        List containing the different force terms to be computed and
        summed in order to arrive at the total force. See above.
    delta_t : float
        Factor by which forces are reduced for numerical updating.
    
    Returns
    -------
    pos_new : numpy array of shape (n_cells, 2[yx])
        Updated coordinates of cell locations.
    force : numpy array of shape (n_cells, 2[yx])
        Force vectors (y, x) affecting each cell.
    """
    
    # Get distance information
    x_dist, y_dist, dist = get_dists(pos)  
    
    # Generate self-mask (to avoid div0 later)
    self_ref_mask = ~np.eye(x_dist.shape[0], dtype=bool)
    
    # Initialize force outputs
    force = np.zeros(pos.shape)
    
    # For each force term...
    for i,force_term in enumerate(force_terms):
        force_func, force_params, min_range, max_range, state_mask, rnd_stdev, rnd_bound = force_term
        
        # Calculate forces
        forces = force_func(dist, *force_params)
        
        # Add random noise
        if rnd_stdev is not None:
            random_forces = np.random.normal(0.0, rnd_stdev, forces.shape)
            if rnd_bound is not None:
                random_forces[random_forces >  rnd_bound] =  rnd_bound
                random_forces[random_forces < -rnd_bound] = -rnd_bound
            forces += random_forces
        
        # Apply range constraints
        forces[dist < min_range] = 0.0
        forces[dist > max_range] = 0.0
        
        # Apply state mask
        if state_mask is not None:
            forces[~state_mask] = 0.0
        
        # Decompose into x and y components (avoiding div0 in center)
        x_forces, y_forces = np.zeros_like(x_dist), np.zeros_like(y_dist)
        np.divide(forces * x_dist, dist, out=x_forces, where=self_ref_mask)
        np.divide(forces * y_dist, dist, out=y_forces, where=self_ref_mask)
        
        # Sum up over neighbors & add to full function
        force[:, 0] += y_forces.sum(axis=1)
        force[:, 1] += x_forces.sum(axis=1)
    
    # Update positions based on force
    pos_new = pos + (delta_t * force)
    
    # Done    
    return pos_new, force