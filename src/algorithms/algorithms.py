"""
Archived Metropolis-Hastings algorithm implementations.

This module previously defined ``AbstractMetropolisHastings`` and
``GaussianMetropolisHastings``. These have been archived to
``_mcmc_archive.py`` and should not be used in new code. For MCMC
sampling, prefer ``emcee`` or ``scipy.stats.sampling`` directly.
"""

from ._mcmc_archive import (  # noqa: F401
    AbstractMetropolisHastings,
    GaussianMetropolisHastings,
)
