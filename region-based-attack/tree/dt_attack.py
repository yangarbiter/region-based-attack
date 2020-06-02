from copy import deepcopy

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from cvxopt import matrix, solvers
import cvxopt.glpk
import cvxopt

from ..base import AttackModel
from ..utils import seed_random_state
from .tree_utils import get_tree_constraints

solvers.options['maxiters'] = 30
solvers.options['show_progress'] = False
solvers.options['refinement'] = 0
solvers.options['feastol'] = 1e-7
solvers.options['abstol'] = 1e-7
solvers.options['reltol'] = 1e-7
cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

def get_sol_l2(target_x, target_y, paths, tree, constraints):
    value = tree.value
    fet_dim = tree.n_features
    temp = (target_x, np.inf)

    for i, path in enumerate(paths):
        if np.argmax(value[path[-1]]) != target_y:
            G, h = constraints[i]
            G, h = matrix(G, tc='d'), matrix(h, tc='d')
            temph = h - 1e-4

            c = matrix(np.concatenate((np.zeros(fet_dim), np.ones(1))), tc='d')

            Q = 2 * matrix(np.eye(fet_dim), tc='d')

            q = matrix(-2*target_x, tc='d')

            try:
                sol = solvers.qp(P=Q, q=q, G=G, h=temph)
            except ValueError as e:
                print(e)
                print("Rare error from solver")
                # usually not able to sat constraint (domain error)

            if sol['status'] == 'optimal':
                ret = np.array(sol['x'], np.float64).reshape(-1)
                eps = np.linalg.norm(ret - target_x, ord=2)
                if eps < temp[1]:
                    temp = (ret, eps)

    if temp [1] != np.inf:
        return temp[0] - target_x
    else:
        return np.zeros_like(target_x)

def get_sol_linf(target_x, target_y, paths, tree, constraints):
    value = tree.value
    fet_dim = tree.n_features
    temp = (target_x, np.inf)

    for i, path in enumerate(paths):
        if np.argmax(value[path[-1]]) != target_y:
            G, h = constraints[i]

            c = matrix(np.concatenate((np.zeros(fet_dim), np.ones(1))), tc='d')

            G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
            G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
            G = np.hstack((G, np.zeros((G.shape[0], 1))))
            G = np.vstack((G, G2, G3))
            h = np.concatenate((h, target_x, -target_x))

            G, h = matrix(G, tc='d'), matrix(h, tc='d')
            temph = h - 1e-4

            lp_sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')
            if lp_sol['status'] == 'optimal':
                sol = np.array(lp_sol['x']).reshape(-1)[:-1]
                eps = np.linalg.norm(sol - target_x, ord=np.inf)
                if eps < temp[1]:
                    temp = (sol, eps)

    if temp [1] != np.inf:
        return temp[0] - target_x
    else:
        return np.zeros_like(target_x)

def _get_path_constraints(clf, path, direction):
    direction = np.asarray(direction)
    path = np.asarray(path)[:-1]

    tree = clf.tree_
    threshold = tree.threshold
    feature = tree.feature

    h = threshold[path]
    G = np.zeros((len(path), tree.n_features), np.float64)
    G[np.arange(len(path)), feature[path]] = 1

    h = h * direction
    G = G * direction.reshape((-1, 1))

    return G, h


class DTExactRBA(AttackModel):
    def __init__(self, clf: DecisionTreeClassifier, norm, n_jobs: int, random_state=None):
        super().__init__()
        self.clf = clf
        self.norm = norm
        self.random_state = seed_random_state(random_state)

        self.paths, self.constraints = get_tree_constraints(clf)

    def perturb(self, X, y):
        pert_X = np.zeros_like(X)
        if len(self.clf.tree_.feature) == 1 and self.clf.tree_.feature[0] == -2:
            # only root and root don't split
            return pert_X

        pred_y = self.clf.predict(X)

        if self.norm == 2:
            get_sol_fn = get_sol_l2
        elif self.norm == np.inf:
            get_sol_fn = get_sol_linf
        else:
            raise ValueError("norm %s not supported", self.norm)

        for sample_id in range(len(X)):
            if pred_y[sample_id] != y[sample_id]:
               continue
            target_x = X[sample_id]
            pert_x = get_sol_fn(target_x, y[sample_id], self.paths,
                                self.clf.tree_, self.constraints)
            if np.linalg.norm(pert_x) != 0:
                assert self.clf.predict([X[sample_id] + pert_x])[0] != y[sample_id]
                pert_X[sample_id, :] = pert_x
            else:
                raise ValueError("shouldn't happend")

        return pert_X
