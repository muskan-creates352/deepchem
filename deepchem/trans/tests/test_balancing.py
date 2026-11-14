import itertools
import tempfile

import numpy as np

import deepchem as dc


def test_binary_1d():
    """Test balancing transformer on single-task dataset without explicit task dimension."""
    n_samples = 20
    n_features = 3
    n_classes = 2
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples,))
    w = np.ones((n_samples,))
    dataset = dc.data.NumpyDataset(X, y, w)

    balancing_transformer = dc.trans.BalancingTransformer(dataset=dataset)
    dataset = balancing_transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    y_task = y_t
    w_task = w_t
    w_orig_task = w
    # Assert that entries with zero weight retain zero weight
    np.testing.assert_allclose(w_task[w_orig_task == 0],
                               np.zeros_like(w_task[w_orig_task == 0]))
    # Check that sum of 0s equals sum of 1s in transformed for each task
    assert np.isclose(np.sum(w_task[y_task == 0]), np.sum(w_task[y_task == 1]))


def test_binary_singletask():
    """Test balancing transformer on single-task dataset."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_classes = 2
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w)

    balancing_transformer = dc.trans.BalancingTransformer(dataset=dataset)
    dataset = balancing_transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(dataset.get_task_names()):
        y_task = y_t[:, ind]
        w_task = w_t[:, ind]
        w_orig_task = w[:, ind]
        # Assert that entries with zero weight retain zero weight
        np.testing.assert_allclose(w_task[w_orig_task == 0],
                                   np.zeros_like(w_task[w_orig_task == 0]))
        # Check that sum of 0s equals sum of 1s in transformed for each task
        assert np.isclose(np.sum(w_task[y_task == 0]),
                          np.sum(w_task[y_task == 1]))


def test_binary_multitask():
    """Test balancing transformer on multitask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 5
    n_classes = 2
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    multitask_dataset = dc.data.NumpyDataset(X, y, w)
    balancing_transformer = dc.trans.BalancingTransformer(
        dataset=multitask_dataset)
    multitask_dataset = balancing_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = (multitask_dataset.X, multitask_dataset.y,
                            multitask_dataset.w, multitask_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(multitask_dataset.get_task_names()):
        y_task = y_t[:, ind]
        w_task = w_t[:, ind]
        w_orig_task = w[:, ind]
        # Assert that entries with zero weight retain zero weight
        np.testing.assert_allclose(w_task[w_orig_task == 0],
                                   np.zeros_like(w_task[w_orig_task == 0]))
        # Check that sum of 0s equals sum of 1s in transformed for each task
        assert np.isclose(np.sum(w_task[y_task == 0]),
                          np.sum(w_task[y_task == 1]))


def test_multiclass_singletask():
    """Test balancing transformer on single-task dataset."""
    n_samples = 50
    n_features = 3
    n_tasks = 1
    n_classes = 5
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w)

    balancing_transformer = dc.trans.BalancingTransformer(dataset=dataset)
    dataset = balancing_transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(dataset.get_task_names()):
        y_task = y_t[:, ind]
        w_task = w_t[:, ind]
        # Check that sum of 0s equals sum of 1s in transformed for each task
        for i, j in itertools.product(range(n_classes), range(n_classes)):
            if i == j:
                continue
            assert np.isclose(np.sum(w_task[y_task == i]),
                              np.sum(w_task[y_task == j]))


def test_transform_to_directory():
    """Test that output can be written to a directory."""
    n_samples = 20
    n_features = 3
    n_classes = 2
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples,))
    w = np.ones((n_samples,))
    dataset = dc.data.NumpyDataset(X, y, w)

    balancing_transformer = dc.trans.BalancingTransformer(dataset=dataset)
    with tempfile.TemporaryDirectory() as tmpdirname:
        dataset = balancing_transformer.transform(dataset, out_dir=tmpdirname)
        balanced_dataset = dc.data.DiskDataset(tmpdirname)
        X_t, y_t, w_t, ids_t = (balanced_dataset.X, balanced_dataset.y,
                                balanced_dataset.w, balanced_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    y_task = y_t
    w_task = w_t
    w_orig_task = w
    # Assert that entries with zero weight retain zero weight
    np.testing.assert_allclose(w_task[w_orig_task == 0],
                               np.zeros_like(w_task[w_orig_task == 0]))
    # Check that sum of 0s equals sum of 1s in transformed for each task
    assert np.isclose(np.sum(w_task[y_task == 0]), np.sum(w_task[y_task == 1]))


def test_array_shapes():
    """Test BalancingTransformer when y and w have different shapes."""
    n_samples = 20
    X = np.random.rand(n_samples, 5)
    y = np.random.randint(2, size=n_samples)
    w = np.ones((n_samples, 1))
    dataset = dc.data.NumpyDataset(X, y, w)
    transformer = dc.trans.BalancingTransformer(dataset)
    Xt, yt, wt, ids = transformer.transform_array(X, y, w, dataset.ids)
    sum0 = np.sum(wt[np.where(y == 0)])
    sum1 = np.sum(wt[np.where(y == 1)])
    assert np.isclose(sum0, sum1)


def test_balancing_transformer_2d_y_1d_w():
    """Test BalancingTransformer with 2D y (N, 1) and 1D w (N,)."""
    n_samples = 20
    n_features = 3
    n_classes = 2
    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples, 1))
    w = np.ones(n_samples)
    dataset = dc.data.NumpyDataset(X, y, w)

    transformer = dc.trans.BalancingTransformer(dataset=dataset)
    transformed_dataset = transformer.transform(dataset)

    assert transformed_dataset.y.shape == (n_samples, 1)
    assert transformed_dataset.w.shape == (n_samples,)
    w_t = transformed_dataset.w
    y_t = transformed_dataset.y.flatten()
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 1]))


def test_array_shapes_2d_y_1d_w():
    """Test BalancingTransformer.transform_array when y is (N, 1) and w is (N,)."""
    n_samples = 20
    X = np.random.rand(n_samples, 5)
    y = np.random.randint(2, size=(n_samples, 1))
    w = np.ones(n_samples)
    dataset = dc.data.NumpyDataset(X, y, w)
    transformer = dc.trans.BalancingTransformer(dataset)
    Xt, yt, wt, ids = transformer.transform_array(X, y, w, dataset.ids)
    sum0 = np.sum(wt[np.where(y.flatten() == 0)])
    sum1 = np.sum(wt[np.where(y.flatten() == 1)])
    assert wt.shape == (n_samples,)
    assert np.isclose(sum0, sum1)
