"""R-MGM objective with selectable MLP or local GMM generator."""

from .model import MGM, Critic, create_method
from .generators import LocalGMMGenerator, make_generator

__all__ = ["MGM", "Critic", "LocalGMMGenerator", "create_method", "make_generator"]
