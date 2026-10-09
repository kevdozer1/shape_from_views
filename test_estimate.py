"""Tests of the estimator's geometry (estimate.py) on synthetic cameras: no images, no GPU needed.

Run with:  python -m pytest -q test_estimate.py   (skipped unless the estimator's packages are installed:
requirements-estimate.txt)
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("numba")
pytest.importorskip("cv2")
pytest.importorskip("lightglue")
import estimate as est  # noqa: E402


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angle_deg(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))


def turn(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return est.expmap(axis * np.radians(deg))


def test_hub_model_explains_two_views_and_is_recovered():
    """Two cameras looking at the hub: the hub-model essential matrix explains every correspondence, (R, rho) can be
    read back out of it, and pair_depths gives each point's depths (in units of camera i's hub distance)."""
    rng = np.random.default_rng(1)
    Ri = random_rotation(rng)
    Rj = turn(rng.normal(size=3), 25) @ Ri                       # a neighbour 25 degrees away
    di, dj = 10.0, 13.0
    X = rng.uniform(-1.5, 1.5, (200, 3))                         # points around the hub (the origin)
    ci, cj = X @ Ri.T + [0, 0, di], X @ Rj.T + [0, 0, dj]
    xi, xj = ci / ci[:, 2:], cj / cj[:, 2:]                      # normalised homogeneous image points
    R, rho = Rj @ Ri.T, dj / di
    E = est.hub_essential(R[None], np.array([rho]))
    assert np.abs(est.sampson(np.repeat(E, len(X), 0), xi, xj)).max() < 1e-12
    z0, z1 = est.pair_depths(np.repeat(R[None], len(X), 0), np.full(len(X), rho), xi, xj)
    assert np.allclose(z0, ci[:, 2] / di) and np.allclose(z1, cj[:, 2] / di)
    R_back, rho_back, _ = est.hub_from_essential(E[0] / np.linalg.norm(E[0]))
    assert np.allclose(R_back, R, atol=1e-9) and rho_back == pytest.approx(rho, rel=1e-9)


def swarm(rng, n):
    """n cameras around the hub, looking at it: world-to-camera rotations."""
    return np.stack([random_rotation(rng) for _ in range(n)])


def test_rotation_averaging_ignores_doppelganger_links():
    """The verified pairs give relative attitudes R_j R_i^T with small noise, plus a few wrong links (a symmetric
    target's opposite sides matched: the relative turn is off by ~170 degrees). The averaging must recover every
    attitude (up to the common turn of the gauge) and drop the wrong links."""
    rng = np.random.default_rng(2)
    n = 14
    R_true = swarm(rng, n)
    fwd = R_true[:, 2]
    edges, wrong = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if fwd[i] @ fwd[j] < np.cos(np.radians(75)):          # only overlapping views are linked
                continue
            Rij = turn(rng.normal(size=3), abs(rng.normal(0, 0.05))) @ R_true[j] @ R_true[i].T
            edges.append(dict(i=i, j=j, R=Rij, rho=1.0, info=np.eye(3) / np.radians(0.1) ** 2, m=np.zeros((120, 2))))
    for k in rng.choice(len(edges), 3, replace=False):            # doppelgangers: plausible match counts, wrong turn
        e = edges[k]
        e["R"] = turn([0, 0, 1], 170) @ e["R"]
        e["m"] = np.zeros((60, 2))
        wrong.append(k)
    R, member, keep, anchor, _, _ = est._global_attitudes(n, edges, np.random.default_rng(0))
    assert member.all()
    assert not keep[wrong].any()
    errors = [angle_deg(R[j] @ R[i].T, R_true[j] @ R_true[i].T) for i in range(n) for j in range(n)]
    assert max(errors) < 0.2


def test_distances_from_ratios_ignore_an_outlier():
    rng = np.random.default_rng(3)
    d = rng.uniform(8, 20, 10)
    I, J = np.triu_indices(10, 1)
    rho = d[J] / d[I] * np.exp(rng.normal(0, 0.002, len(I)))
    rho[5] *= 1.5                                                 # one wrong ratio
    got = est._global_distances(10, I, J, rho, np.ones(len(I)), anchor=0)
    assert np.allclose(got, d / d[0], rtol=0.003)
