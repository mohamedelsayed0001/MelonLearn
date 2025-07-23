from .sgd import SGD
from .adam import Adam
from .momentum import Momentum
from .RMSprop import RMSProp

__all__ = ["SGD", "Adam", "Momentum", "RMSProp"]

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
    "adam": Adam,
    "momentum": Momentum,
    "rmsprop": RMSProp,
}
