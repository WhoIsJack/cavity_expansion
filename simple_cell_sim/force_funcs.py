# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:29:45 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  Force functions F=f(dist, ...) for use in specifying
            simple_cell_sim models.
"""


# Imports
import numpy as np


def f_Hooke(dist, dist0, k):
    """Hooke's law spring force function.

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
    force : numpy array of shape (n_cells, n_cells)
        Force between each cell pair.
    """
    force = k * (dist-dist0)
    return force


def f_expdecay(dist, dist0, pot0, e):
    """Exponential decay force function matching the potential
    energy landscape in `potential_funcs.pot_expdecay`.

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
    force : numpy array of shape (n_cells, n_cells)
        Force between each cell pair.
    """
    force = -e * pot0 * np.exp(-e * (dist - dist0))
    return force


def f_expneg(dist, dist0, pot0, e):
    """Negative exponential force function matching the potential
    energy landscape in `potential_funcs.pot_expneg`.

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
    force : numpy array of shape (n_cells, n_cells)
        Force between each cell pair.
    """
    force =  e * pot0 * np.exp(-e * (dist - dist0))
    return force


def f_anharmonic(dist, dist0, pot0, m, e1, e2):
    """Anharmonic oscillator force function matching the potential
    energy landscape in `potential_funcs.pot_anharmonic`.

    Note: Returns F=0 at dist=0.

    Parameters
    ----------
    dist : numpy array of shape (n_cells, n_cells)
        Pairwise Euclidean distances between cells.
    dist0 : float
        The distance at which the potential energy is
        minimal, at least given m=2 and e1/e2=2; for
        other parameter values, the minimum may shift.
    m : float
        Parameter, see equation in pot_anharmonic.
    e1 : float
        Parameter, see equation in pot_anharmonic.
    e2 : float
        Parameter, see equation in pot_anharmonic.
        
    Returns
    -------
    force : numpy array of shape (n_cells, n_cells)
        Force between each cell pair.
    """
    force = np.zeros_like(dist)
    force[dist>0.0] = (pot0 * (e1 * (dist0/dist[dist>0.0])**e1 - m * e2 * (dist0/dist[dist>0.0])**e2)) / dist[dist>0.0]
    return force