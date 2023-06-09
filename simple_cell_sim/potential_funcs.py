# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:28:48 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  Potential functions E_pot=f(dist, ...) for use in understanding
            and specifying simple_cell_sim models.
"""


# Imports
import numpy as np


def pot_Hooke(dist, dist0, k):
    """Hooke's law spring potential function.

    Parameters
    ----------
    dist : numpy array of shape (n_cells, n_cells)
        Pairwise Euclidean distances between cells.
    dist0 : float
        The spring's resting distance between cells.
    k : float
        The spring constant.
    
    Returns
    -------
    potential : numpy array of shape (n_cells, n_cells)
        Potential energy between each cell pair.
    """
    potential = 1.0/2.0 * k * (dist-dist0)**2.0
    return potential


def pot_expdecay(dist, dist0, pot0, e):
    """Exponential decay potential function. The resulting potential
    decays from a value pot0 at dist0 to a value of 0 asymptotically.

    Parameters
    ----------
    dist : numpy array of shape (n_cells, n_cells)
        Pairwise Euclidean distances between cells.
    dist0 : float
        The distance at which E_pot=pot0.
    e : float
        The exponent of the potential.
    
    Returns
    -------
    potential : numpy array of shape (n_cells, n_cells)
        Potential energy between each cell pair.
    """
    potential = pot0 * np.exp(-e * (dist-dist0))
    return potential



def pot_expneg(dist, dist0, pot0, e):
    """Negative exponential potential function. The resulting potential
    increases from a value -pot0 at dist0 to a value of 0 asymptotically.

    Parameters
    ----------
    dist : numpy array of shape (n_cells, n_cells)
        Pairwise Euclidean distances between cells.
    dist0 : float
        The distance at which E_pot=-pot0.
    e : float
        The exponent of the potential.
    
    Returns
    -------
    potential : numpy array of shape (n_cells, n_cells)
        Potential energy between each cell pair.
    """
    potential = pot0 - pot0 * np.exp(-e * (dist-dist0))
    return potential


def pot_anharmonic(dist, dist0, pot0, m, e1, e2):
    """Anharmonic oscillator potential function.

    $E_{pot} = -pot_0 \ \left[\left(\frac{dist_0}{dist}\right)^{e_1} - m \ \left(\frac{dist_0}{dist}\right)^{e_2}\right]$
        where dist>0.0,
        otherwise 0.0

    See also: https://en.wikipedia.org/wiki/Anharmonicity

    Parameters
    ----------
    dist : numpy array of shape (n_cells, n_cells)
        Pairwise Euclidean distances between cells.
    dist0 : float
        The distance at which the potential energy is
        minimal, at least given m=2 and e1/e2=2; for
        other parameter values, the minimum may shift.
    m : float
        Parameter, see equation above.
    e1 : float
        Parameter, see equation above.
    e2 : float
        Parameter, see equation above.
        
    Returns
    -------
    potential : numpy array of shape (n_cells, n_cells)
        Potential energy between each cell pair.
    """
    potential = np.zeros_like(dist)
    potential[dist>0.0] = -pot0 * ((dist0 / dist[dist>0.0])**e1 - m * (dist0 / dist[dist>0.0])**e2)
    return potential