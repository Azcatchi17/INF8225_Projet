"""Reserved namespace for shared library code.

For now, the experimental iterations live under ``experiments/msd/`` and
import their helpers from ``experiments.msd._shared``. As the project
matures, components that prove reusable across iterations (e.g. data
loaders, evaluation metrics, plotting utilities) should migrate here.
"""

__version__ = "1.0.0"
