import poetry_version
from .data import DataLoader
from .visualise import plot_random_sample
from .model import SegNetModel

__version__ = poetry_version.extract(__file__)
