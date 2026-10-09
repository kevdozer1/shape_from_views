"""Tests: conventions, file formats, poses and an end-to-end visual hull of a synthetic object.

Run with:  python -m pytest -q
The end-to-end test ray-traces a "toy plane" (ellipsoid fuselage + thin wings + tail fin) from
three rings of cameras, writes a real dataset folder (PNG images and masks), runs the command line
tools on it and compares the carved hull with the true shape.
"""
import math
import shutil
import struct

import numpy as np
import pytest
from PIL import Image as PILImage

import colmap
import shape_from_views as cli
import visual_hull as vh
from colmap import Camera, Image, Point3D, project, read_model, write_model
from poses import (look_at, orbit_states, pose_from_state, qvec_to_rotmat, rotmat_to_qvec,
                   state_from_pose, to_differential)

# ----------------------------------------------------------------------------- the synthetic object

ELLIPSOIDS = [((0.0, 0.0, 0.0), (0.6, 3.0, 0.6), (150, 150, 150))]              # (centre, semi-axes, colour)
BOXES = [((-3.0, -0.5, -0.08), (3.0, 0.7, 0.08), (40, 80, 220)),                 # wings  (lo, hi, colour)
         ((-0.06, -2.9, 0.3), (0.06, -2.2, 1.5), (220, 50, 50)),                # tail fin
         ((-1.2, -2.9, -0.05), (1.2, -2.4, 0.05), (40, 80, 220))]               # horizontal stabiliser
RINGS = "10,-4;12,0;10,4"
WIDTH, HEIGHT = 160, 120


def inside_object(P):
    """True for points inside the toy plane."""
    inside = np.zeros(len(P), bool)
    for c, s, _ in ELLIPSOIDS:
        inside |= (((P - c) / s) ** 2).sum(axis=1) <= 1
    for lo, hi, _ in BOXES:
        inside |= np.all((P >= lo) & (P <= hi), axis=1)
    return inside


def render(camera, R, t):
    """Exact ray casting through every pixel centre -> (RGB image, mask)."""
    f, cx, cy = camera.params
    u, v = np.meshgrid(np.arange(camera.width) + 0.5, np.arange(camera.height) + 0.5)
    rays_cam = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], axis=-1).reshape(-1, 3)
    d = rays_cam @ R                       # world directions (R^T applied to each row)
    o = -R.T @ t                            # camera centre
    depth = np.full(len(d), np.inf)
    color = np.zeros((len(d), 3))
    for c, s, rgb in ELLIPSOIDS:            # |(o + s d - c) / axes| = 1
        oo, dd = (o - c) / s, d / s
        a, b, cc = (dd * dd).sum(1), 2 * (oo * dd).sum(1), (oo * oo).sum() - 1
        disc = b * b - 4 * a * cc
        hit = disc >= 0
        near = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        closer = hit & (near > 0) & (near < depth)
        depth[closer], color[closer] = near[closer], rgb
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi, rgb in BOXES:           # slab test
            t0, t1 = (np.array(lo) - o) / d, (np.array(hi) - o) / d
            near, far = np.minimum(t0, t1).max(1), np.maximum(t0, t1).min(1)
            closer = (near <= far) & (near > 0) & (near < depth)
            depth[closer], color[closer] = near[closer], rgb
    mask = np.isfinite(depth).reshape(camera.height, camera.width)
    return color.reshape(camera.height, camera.width, 3).astype(np.uint8), mask


def make_dataset(folder, rolled=False, cropped=()):
    """Write images/, masks/ and the orbit's sparse/0 into folder. rolled=True stores the
    ORIGINAL repo's camera convention (every camera turned 180 degrees about its axis).
    Views listed in `cropped` use a camera whose principal point is shifted 70 px down, so the
    lower part of the plane falls outside those images."""
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    camera = cli.render_camera(WIDTH, HEIGHT, 40.0, 36.0)
    shifted = Camera(2, "SIMPLE_PINHOLE", WIDTH, HEIGHT, [camera.params[0], camera.params[1], camera.params[2] + 70])
    names = [f"{i:04d}.png" for i in range(36)]
    rings = [tuple(map(float, r.split(","))) for r in RINGS.split(";")]
    images = {}
    for i, (name, state) in enumerate(zip(names, orbit_states(len(names), rings))):
        R, t = pose_from_state(*state)
        cam = shifted if i in cropped else camera
        rgb, mask = render(cam, R, t)
        PILImage.fromarray(rgb).save(folder / "images" / name)
        PILImage.fromarray((mask * 255).astype(np.uint8)).save(folder / "masks" / name)
        if rolled:
            R, t = np.diag([-1.0, -1.0, 1.0]) @ R, np.diag([-1.0, -1.0, 1.0]) @ t
        images[i + 1] = Image(i + 1, rotmat_to_qvec(R), tuple(t), cam.id, name)
    write_model(folder / "sparse" / "0", {1: camera, 2: shifted} if cropped else {1: camera}, images, {})
    return folder


# ----------------------------------------------------------------------------- unit tests

def test_look_at_uses_opencv_axes():
    C = (0.0, -10.0, 0.0)
    R = look_at(C)
    t = -R @ np.array(C)
    camera = Camera(1, "SIMPLE_PINHOLE", 100, 100, [100.0, 50.0, 50.0])
    for world, check in [((1.0, 0.0, 0.0), lambda u, v: u > 50),     # +x is to the right
                         ((0.0, 0.0, 1.0), lambda u, v: v < 50)]:    # +z (up) is higher in the image
        x, y, z = R @ np.array(world) + t
        assert check(*project(camera, x, y, z))
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)
        q2 = rotmat_to_qvec(qvec_to_rotmat(q))
        assert q2[0] >= 0 and np.allclose(q2, q, atol=1e-12)


def test_opencv_projection_matches_hand_computation():
    camera = Camera(1, "OPENCV", 640, 480, [500.0, 510.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002])
    a, b = 0.2, -0.1
    r2 = a * a + b * b
    radial = 0.1 * r2 + 0.01 * r2 * r2
    ad = a + a * radial + 2 * 0.001 * a * b + 0.002 * (r2 + 2 * a * a)
    bd = b + b * radial + 2 * 0.002 * a * b + 0.001 * (r2 + 2 * b * b)
    u, v = project(camera, 0.4, -0.2, 2.0)
    assert (u, v) == (500.0 * ad + 320.0, 510.0 * bd + 240.0)


@pytest.mark.parametrize("text", [False, True])
def test_colmap_model_round_trip(tmp_path, text):
    cameras = {1: Camera(1, "PINHOLE", 640, 480, [500.0, 501.0, 320.5, 239.5]),
               2: Camera(2, "OPENCV", 800, 600, [600.0, 600.0, 400.0, 300.0, -0.1, 0.01, 0.0, 0.001])}
    images = {1: Image(1, (1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3), 1, "a.jpg", [(1.5, 2.5, 7), (3.0, 4.0, -1)]),
              2: Image(2, (0.5, 0.5, 0.5, 0.5), (1.0, -2.0, 3.0), 2, "b.jpg", [])}
    points = {7: Point3D(7, (1.0, 2.0, 3.0), (10, 20, 30), 0.5, [(1, 0)])}
    write_model(tmp_path, cameras, images, points, text=text)
    assert read_model(tmp_path) == (cameras, images, points)


def test_orbit_states_replay_through_poses(tmp_path):
    rings = [(10.0, -4.0), (12.0, 0.0), (10.0, 4.0)]
    states = orbit_states(360, rings)
    baseline, differences = to_differential(states)
    assert baseline[1] == pytest.approx(-21.80140948635181)
    assert differences[120][2] == pytest.approx(21.80140948635181)    # middle ring: +21.8 deg
    for (distance, yaw, pitch, roll), (d, dy, dp, dr) in zip(states, differences):
        R1, t1 = pose_from_state(distance, yaw, pitch, roll)
        R2, t2 = pose_from_state(d, baseline[0] + dy, baseline[1] + dp, baseline[2] + dr)
        assert np.allclose(R1, R2, atol=1e-9) and np.allclose(t1, t2, atol=1e-9)
    # the camera positions are exactly the rings
    R, t = pose_from_state(*states[150])               # middle ring, 30th frame: yaw 90
    assert np.allclose(-R.T @ t, [12.0, 0.0, 0.0])


def test_state_from_pose_inverts_pose_from_state():
    rng = np.random.default_rng(3)
    for _ in range(200):
        state = (rng.uniform(1, 20), rng.uniform(-179, 179), rng.uniform(-85, 85), rng.uniform(-179, 179))
        R, t = pose_from_state(*state)
        assert np.allclose(state_from_pose(R, t[2]), state, atol=1e-9)


def test_relative_estimates_replay_through_poses(plane, tmp_path):
    """What `estimate` writes (attitudes relative to observer 1, distances as multiples of its distance) goes
    through `poses --photos` and gives back the true cameras, up to the origin's turn and the scale."""
    folder, _, _ = plane
    _, images, _ = read_model(folder / "sparse" / "0")
    by_name = {im.name: im for im in images.values()}
    photos = sorted(by_name)[::3][::-1]                 # a subset, observer 1 not first by name
    R = np.stack([qvec_to_rotmat(by_name[n].qvec) for n in photos])
    t = np.stack([np.array(by_name[n].tvec) for n in photos])
    d = t[:, 2]                                         # the cameras look at the origin: t = (0, 0, distance)
    cli.write_state_lists(tmp_path / "states", photos, cli.relative_to_states(R @ R[0].T, d / d[0], 2.5))
    cli.main(["poses", str(folder), str(tmp_path / "states" / "distances.txt"), str(tmp_path / "states" / "attitudes.txt"),
              "--baseline", "0", "0", "0", "--photos", str(tmp_path / "states" / "photos.txt"), "--out", str(tmp_path)])
    _, replayed, _ = read_model(tmp_path / "sparse" / "0")
    got = {im.name: im for im in replayed.values()}
    assert sorted(got) == sorted(photos)
    Rg = np.stack([qvec_to_rotmat(got[n].qvec) for n in photos])
    tg = np.stack([np.array(got[n].tvec) for n in photos])
    assert np.allclose(Rg @ Rg[0].T, R @ R[0].T, atol=1e-9)          # same relative attitudes
    assert np.allclose(tg[:, 2] / 2.5, d / d[0], atol=1e-9)          # same relative distances, scaled by 2.5
    assert np.allclose(Rg[0], pose_from_state(2.5, 0, 0, 0)[0])      # observer 1 at yaw/pitch/roll 0


def test_bad_arguments_are_errors(plane):
    folder = str(plane[0])
    for argv in (["hull", "--data", "x", "--passb-res", "100"],     # typo of --passB-res
                 ["hull", "--data", folder, "--passB-res", "-5", "--no-export"],
                 ["hull", "--data", folder, "--max-misses", "-1", "--no-export"],
                 ["subset", folder, "--every", "0"]):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_subset_selection_matches_the_cpp_version():
    assert next(cli.splitmix64(0)) == 0xE220A8397B1DCDAF          # reference value of splitmix64
    assert sorted(cli.shuffled(range(36), 42)[:5]) == [0, 2, 4, 31, 32]   # subset_dataset --random 5 --seed 42


def test_masks_are_thresholded_like_the_cpp_version(tmp_path):
    grey16 = np.array([[32767, 32768, 65535, 0]], dtype=np.uint16)   # 16-bit PNG: the high byte counts
    PILImage.fromarray(grey16).save(tmp_path / "m16.png")
    assert vh.load_mask(tmp_path / "m16.png").tolist() == [[False, True, True, False]]
    rgb = np.array([[[0, 218, 0], [0, 219, 0]]], dtype=np.uint8)      # colour: PIL's rounded luma
    PILImage.fromarray(rgb).save(tmp_path / "rgb.png")
    assert vh.load_mask(tmp_path / "rgb.png").tolist() == [[True, True]]


# ----------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def plane(tmp_path_factory):
    folder = make_dataset(tmp_path_factory.mktemp("plane"))
    views = vh.load_views(folder / "images", folder / "masks", folder / "sparse" / "0")
    return folder, views, vh.visual_hull(views, voxel_num=48, passB_res=120)


def test_hull_contains_the_object(plane):
    _, _, hull = plane
    samples = np.random.default_rng(1).uniform([-3.1, -3.1, -0.7], [3.1, 3.1, 1.6], size=(200000, 3))
    samples = samples[inside_object(samples)]
    ijk = np.floor((samples - hull.lo) / hull.voxel_size).astype(int)
    assert np.all((ijk >= 0) & (ijk < hull.occupied.shape))
    # nothing of the object is carved, except slivers thinner than a pixel in some mask (the masks
    # are sampled at pixel centres, so such slivers are simply not in the data)
    assert hull.occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]].mean() >= 0.999


def test_hull_is_tight(plane):
    _, _, hull = plane
    centers = np.argwhere(hull.occupied) * hull.voxel_size + hull.lo + hull.voxel_size / 2
    truth = inside_object(centers).sum()
    assert len(centers) <= 2.5 * truth                    # silhouettes cannot carve concavities; 2.5x is generous


def test_views_agree_with_the_hull(plane):
    _, _, hull = plane
    assert hull.coverage.mean() >= 0.97 and hull.coverage.min() >= 0.9


def test_colours_come_from_the_right_parts(plane):
    _, _, hull = plane
    on_wing = (np.abs(hull.points[:, 0]) > 1.5) & (np.abs(hull.points[:, 2]) < 0.2)
    blue = hull.colors[on_wing, 2] > hull.colors[on_wing, 0]
    assert on_wing.sum() > 50 and blue.mean() > 0.8


def test_original_camera_convention_is_detected(tmp_path):
    """Regression test for the original bug: 180-degree-rolled cameras make the views disagree."""
    folder = make_dataset(tmp_path / "rolled", rolled=True)
    views = vh.load_views(folder / "images", folder / "masks", folder / "sparse" / "0")
    try:
        hull = vh.visual_hull(views, voxel_num=48, passB_res=120, color=False)
    except RuntimeError:     # nothing survived at all: also a loud failure
        return
    assert hull.coverage.mean() < 0.7


def test_parts_outside_some_frames_survive(tmp_path):
    """Views that cannot see a part (it is outside their image) must not carve it away."""
    folder = make_dataset(tmp_path / "cropped", cropped=(0, 12, 24))   # one view per ring
    views = vh.load_views(folder / "images", folder / "masks", folder / "sparse" / "0")
    assert min(int(v.mask[-1].sum()) for v in views if v.camera.id == 2) > 0   # really cut by the border
    samples = np.random.default_rng(2).uniform([-3.1, -3.1, -0.7], [3.1, 3.1, 1.6], size=(200000, 3))
    samples = samples[inside_object(samples)]

    def kept_fraction(**options):
        hull = vh.visual_hull(views, voxel_num=48, passB_res=120, color=False, **options)
        ijk = np.floor((samples - hull.lo) / hull.voxel_size).astype(int)
        inside = np.all((ijk >= 0) & (ijk < hull.occupied.shape), axis=1)
        return np.mean(inside & hull.occupied[tuple(np.clip(ijk, 0, np.array(hull.occupied.shape) - 1).T)])

    assert kept_fraction() >= 0.999                    # default: views that cannot see a part leave it alone
    assert kept_fraction(min_votes=len(views)) < 0.8   # the old "every view must see it" rule loses it


def test_command_line_end_to_end(plane, tmp_path):
    folder, _, _ = plane
    out = tmp_path / "hull.ply"
    cli.main(["orbit", str(folder), "--rings", RINGS, "--out", str(tmp_path / "orbit"), "--write-states", str(tmp_path)])
    cli.main(["poses", str(folder), str(tmp_path / "distances.txt"), str(tmp_path / "attitudes.txt"),
              "--baseline", "0", "-21.80140948635181", "0", "--out", str(tmp_path / "poses")])
    for generated in ("orbit", "poses"):   # both generators reproduce the poses the images were rendered with
        _, rendered, _ = read_model(folder / "sparse" / "0")
        _, images, _ = read_model(tmp_path / generated / "sparse" / "0")
        for a, b in zip(rendered.values(), images.values()):
            assert a.name == b.name and np.allclose(a.qvec, b.qvec, atol=1e-9) and np.allclose(a.tvec, b.tvec, atol=1e-9)
    cli.main(["hull", "--data", str(folder), "--sparse", str(tmp_path / "poses" / "sparse" / "0"),
              "--voxel-num", "32", "--passB-res", "64", "--out-ply", str(out)])
    header = out.read_bytes().split(b"end_header\n")[0].decode()
    assert "property float nx" in header and "format binary_little_endian" in header
    n_points = int(header.split("element vertex ")[1].split()[0])
    _, _, points = read_model(tmp_path / "poses" / "sparse" / "0")
    assert len(points) == n_points > 0
    cli.main(["poses", str(folder), str(tmp_path / "distances.txt"), str(tmp_path / "attitudes.txt"),
              "--baseline", "0", "-21.80140948635181", "0", "--out", str(tmp_path / "poses2")])
    shutil.copy(tmp_path / "poses" / "sparse" / "0" / "points3D.ply", tmp_path / "poses2" / "sparse" / "0")
    cli.main(["poses", str(folder), str(tmp_path / "distances.txt"), str(tmp_path / "attitudes.txt"),
              "--baseline", "0", "-21.80140948635181", "0", "--out", str(tmp_path / "poses2")])
    assert (tmp_path / "poses2" / "sparse" / "0" / "points3D_previous.ply").exists()   # stale hull moved aside
    assert not (tmp_path / "poses2" / "sparse" / "0" / "points3D.ply").exists()
    cli.main(["to-txt", str(tmp_path / "poses" / "sparse" / "0")])
    sparse = tmp_path / "poses" / "sparse" / "0"
    assert colmap.read_images_txt(sparse / "images.txt") == colmap.read_images_bin(sparse / "images.bin")
    assert colmap.read_points3d_txt(sparse / "points3D.txt") == colmap.read_points3d_bin(sparse / "points3D.bin")
    cli.main(["subset", str(folder), "--every", "6", "--out", str(tmp_path / "every6")])
    _, subset, _ = read_model(tmp_path / "every6" / "sparse" / "0")
    assert [im.name for im in subset.values()] == [f"{i:04d}.png" for i in range(0, 36, 6)]
    with pytest.raises(SystemExit):   # never mix a new selection into an old folder
        cli.main(["subset", str(folder), "--every", "6", "--out", str(tmp_path / "every6")])
    cli.main(["transplant", str(tmp_path / "every6" / "sparse" / "0"), str(tmp_path / "poses" / "sparse" / "0")])
    assert (tmp_path / "every6" / "sparse" / "0" / "images_original.bin").exists()
