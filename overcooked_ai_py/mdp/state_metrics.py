import numpy as np


class StateMetrics(object):

    def __init__(self, mdp):
        self.mdp = mdp

    def l1_distance(self, matrix0, matrix1):
        assert matrix0.shape == matrix1.shape
        return np.sum(np.abs(matrix0 - matrix1))

    def l2_distance(self, matrix0, matrix1):
        assert matrix0.shape == matrix1.shape
        return np.sum(np.power(matrix0 - matrix1, 2))
