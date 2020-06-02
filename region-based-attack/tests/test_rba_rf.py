"""
Unittest for Nearest Neighbor attack
"""  
import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier

from ..tree.rf_attack import RFAttack


class TestRFAttack(unittest.TestCase):
    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'datasets/diabetes_scale')
        self.X, self.y = load_svmlight_file(dataset_filepath)
        self.X = self.X.toarray().astype(np.float64)
        self.y = ((self.y + 1) // 2).astype(np.int)

        self.trnX, self.trny = self.X[:100], self.y[:100]
        self.tstX, self.tsty = self.X[100:110], self.y[100:110]
        self.clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(self.trnX, self.trny)

    def test_realdata_rf(self):
        trnX, trny, tstX, tsty, clf = self.trnX, self.trny, self.tstX, self.tsty, self.clf

        for norm in [2, np.inf]:
            attack = RFAttack(trnX=trnX, trny=trny, clf=clf, norm=norm, n_jobs=1)
            perturb = attack.perturb(tstX, tsty)

            self.assertTrue(clf.score(tstX + perturb, tsty) == 0.)

            pert_dist = np.linalg.norm(perturb, ord=norm, axis=1)
            if norm == 2:
                assert_almost_equal(
                    [0., 0.322, 0., 0.232, 0.393, 0.127, 0.059, 0.065, 0.01, 0.],
                    pert_dist, decimal=3
                )
            elif norm == np.inf:
                assert_almost_equal(
                    [0., 0.317, 0., 0.212, 0.317, 0.117, 0.059, 0.061, 0.01, 0.],
                    pert_dist, decimal=3
                )

    def test_realdata_rf_approx(self):
        trnX, trny, tstX, tsty, clf = self.trnX, self.trny, self.tstX, self.tsty, self.clf

        for norm in [2, np.inf]:
            attack = RFAttack(trnX=trnX, trny=trny, clf=clf, method='rev', norm=norm, n_jobs=1)
            perturb = attack.perturb(tstX, tsty)

            self.assertTrue(clf.score(tstX + perturb, tsty) == 0.)

            pert_dist = np.linalg.norm(perturb, ord=norm, axis=1)
            if norm == 2:
                assert_almost_equal(
                    [0., 0.517, 0., 0.282, 0.35, 0.127, 0.35, 0.063, 0.01, 0.],
                    pert_dist, decimal=3
                )
            elif norm == np.inf:
                assert_almost_equal(
                    [0., 0.517, 0., 0.273, 0.35, 0.117, 0.35, 0.061, 0.01, 0.],
                    pert_dist, decimal=3
                )

if __name__ == '__main__':
    unittest.main()
