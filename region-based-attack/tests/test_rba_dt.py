"""
Unittest for Nearest Neighbor attack
"""  
import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeClassifier

from ..tree.dt_attack import DTExactRBA


class TestRFAttack(unittest.TestCase):
    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'datasets/diabetes_scale')
        self.X, self.y = load_svmlight_file(dataset_filepath)
        self.X = self.X.toarray().astype(np.float64)
        self.y = ((self.y + 1) // 2).astype(np.int)

        self.trnX, self.trny = self.X[:100], self.y[:100]
        self.tstX, self.tsty = self.X[100:110], self.y[100:110]
        self.clf = DecisionTreeClassifier(random_state=1126).fit(self.trnX, self.trny)

    def test_realdata_dt(self):
        tstX, tsty, clf = self.tstX, self.tsty, self.clf

        for norm in [2, np.inf]:
            attack = DTExactRBA(clf=clf, norm=norm, n_jobs=1, random_state=1126)
            perturb = attack.perturb(tstX, tsty)

            self.assertTrue(clf.score(tstX + perturb, tsty) == 0.)

            pert_dist = np.linalg.norm(perturb, ord=norm, axis=1)
            if norm == 2:
                assert_almost_equal(
                    [0., 0.116, 0., 0.073, 0., 0.156, 0., 0.01, 0., 0.009],
                    pert_dist, decimal=3
                )
            elif norm == np.inf:
                assert_almost_equal(
                    [0., 0.116, 0., 0.07, 0., 0.156, 0., 0.01, 0., 0.009],
                    pert_dist, decimal=3
                )

if __name__ == '__main__':
    unittest.main()
