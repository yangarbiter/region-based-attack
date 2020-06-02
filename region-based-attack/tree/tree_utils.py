from copy import deepcopy

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed


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

def get_tree_constraints(clf: DecisionTreeClassifier, n_jobs=1):
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    paths: list = []
    directions: list = []

    path = []
    direction = []

    def _dfs(node_id):
        path.append(node_id)

        if children_left[node_id] != children_right[node_id]:
            direction.append(1)
            _dfs(children_left[node_id])
            direction[-1] = -1
            _dfs(children_right[node_id])
            direction.pop()
        else:
            paths.append(deepcopy(path))
            directions.append(deepcopy(direction))

        path.pop()
    _dfs(0)

    constraints = Parallel(n_jobs=n_jobs)(
        delayed(_get_path_constraints)(clf, p, d) for p, d in zip(paths, directions))
    return paths, constraints
