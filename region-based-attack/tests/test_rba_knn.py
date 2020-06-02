"""
Unittest for Nearest Neighbor attack
"""  
import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier

from ..knn import KNNExactRBA, KNNApproxRBA


class TestNNAttack(unittest.TestCase):
    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'datasets/diabetes_scale')
        self.X, self.y = load_svmlight_file(dataset_filepath)
        self.X = self.X.toarray().astype(np.float64)
        self.y = ((self.y + 1) // 2).astype(np.int)

    def test_small(self):
        trnX = np.array([[0, 1], [1, 0], [1, 1], [2, 2], [-1, -1], [2, -1], [-1, 2]], dtype=np.float)
        trny = np.array([0, 0, 0, 1, 1, 1, 1])
        tstX = np.array([[0, 0]], dtype=np.float)
        tsty = np.array([0])

        attack = KNNExactRBA(trnX=trnX, trny=trny, n_neighbors=3, norm=2, n_jobs=1)
        perturb = attack.perturb(tstX, tsty)
        tstX = tstX + perturb

        self.assertTrue(np.logical_or(
            np.allclose(np.array([[-0.5, 0.5]]), perturb, atol=1e-3),
            np.allclose(np.array([[0.5, -0.5]]), perturb, atol=1e-3)
        ))

    def test_realdata(self):
        trnX, trny = self.X[:100], self.y[:100]
        tstX, tsty = self.X[100:110], self.y[100:110]
        clf = KNeighborsClassifier(n_neighbors=1).fit(trnX, trny)

        for norm in [1, 2, np.inf]:
            attack = KNNExactRBA(trnX=trnX, trny=trny, n_neighbors=1, norm=norm, n_jobs=1)
            perturb = attack.perturb(tstX, tsty)

            self.assertTrue(clf.score(tstX + perturb, tsty) == 0.)

            pert_dist = np.linalg.norm(perturb, ord=norm, axis=1)
            if norm == 1:
                assert_almost_equal(
                    [0., 0.661, 0.823, 0.47, 0., 0.203, 0.661, 0.172, 0., 0.],
                    pert_dist, decimal=3
                )
            elif norm == 2:
                assert_almost_equal(
                    [0., 0.478, 0.467, 0.265, 0., 0.131, 0.363, 0.132, 0., 0.],
                    pert_dist, decimal=3
                )
            elif norm == np.inf:
                assert_almost_equal(
                    [0., 0.207, 0.195, 0.113, 0., 0.056, 0.17 , 0.065, 0., 0.],
                    pert_dist, decimal=3
                )

    def test_realdata_approx(self):
        trnX, trny = self.X[:100], self.y[:100]
        tstX, tsty = self.X[100:110], self.y[100:110]
        clf = KNeighborsClassifier(n_neighbors=1).fit(trnX, trny)

        for norm in [1, 2, np.inf]:
            attack = KNNApproxRBA(trnX=trnX, trny=trny, n_neighbors=1, norm=norm, n_jobs=1)
            perturb = attack.perturb(tstX, tsty)

            self.assertTrue(clf.score(tstX + perturb, tsty) == 0.)

            pert_dist = np.linalg.norm(perturb, ord=norm, axis=1)
            if norm == 1:
                assert_almost_equal(
                    [0., 0.661, 0.823, 0.47, 0., 0.203, 0.661, 0.172, 0., 0.],
                    pert_dist, decimal=3
                )
            elif norm == 2:
                assert_almost_equal(
                    [0., 0.478, 0.467, 0.265, 0., 0.131, 0.363, 0.132, 0., 0.],
                    pert_dist, decimal=3
                )
            elif norm == np.inf:
                assert_almost_equal(
                    [0., 0.207, 0.195, 0.113, 0., 0.056, 0.17 , 0.065, 0., 0.],
                    pert_dist, decimal=3
                )

if __name__ == '__main__':
    unittest.main()
