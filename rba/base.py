from abc import abstractmethod

import numpy as np


class AttackModel():

    def __init__(self):
        super().__init__()

    @abstractmethod
    def perturb(self, X, y):
        pass