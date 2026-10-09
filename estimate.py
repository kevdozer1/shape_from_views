"""Order-free hub-and-spoke state estimation for a swarm of observers: SuperPoint + LightGlue on the GPU, two-view
geometry under the hub model, anisotropic rotation averaging, hub-and-spoke bundle adjustment.

The state model: every observer looks straight at the target's centre (the hub), so camera i is
        X_cam = R_i X + (0, 0, d_i)          (hub at the origin; OpenCV axes: x right, y down, z forward)
with R_0 = I, d_0 = 1 for the origin observer (the first input). Nothing here uses the order of the inputs except
that the first one is the origin: pairs are chosen from the images, every link between observers comes from matches.

Stages (each reported to `progress(stage, seconds)` when it ends, with the seconds of that occurrence; matching and
mapping repeat for a large swarm):
  load      images and masks (threads); also the model warm-up if the caller skipped warm_up()
  features  SuperPoint on the aircraft only (background blanked), the 1024 strongest keypoints inside the eroded mask,
            sub-pixel positions; silhouette outlines (convex hulls) for the epipolar tangency tests.
            ROLL: SuperPoint and LightGlue are not rotation-invariant and a swarm has random roll, so small swarms also
            describe every crop turned in-plane by +-ROLL_DEG (keypoints mapped back to the original image exactly);
            a pair is matched once, in the version whose descriptors agree best (relative roll left: <= ~20 deg)
  matching  which pairs: all of them for up to 64 observers, else a mutual-nearest-neighbour shortlist (plus the origin
            with everyone); LightGlue batched on the GPU. Large swarms match more pairs while mapping: those the
            estimate says overlap, and each unplaced observer with its most similar placed ones
  geometry  two-view geometry under the hub model: essential-matrix RANSAC for the inliers, then the 4 unknowns (relative
            attitude + distance ratio) of all pairs at once, with the silhouettes' epipolar tangency pinning the orbit
            angle that two views of a small object leave loose; checks against doppelgangers (the aircraft is left/right
            symmetric with the same decals on both sides); robust rotation averaging that knows each pair's weak
            direction (RANSAC over spanning trees, anisotropic IRLS); distances from the ratios. Every group of observers
            the verified pairs link is kept: the largest one is the start, the others are UNITS (rigid groups with known
            internal states) that are later attached as a whole
  mapping   tracks (union-find), hub-and-spoke bundle adjustment (4 unknowns per observer) with silhouette tangency for
            camera pairs; then rounds that place the observers left out: resection (P3P with the hub as the third
            point), or, with no feature link, the silhouette alone (pose search against the visual hull of the placed
            observers; a unit is searched and refined as a whole). A placed observer brings its unit along. A
            silhouette-only placement is compared with its MIRROR TWINS (the aircraft is left/right symmetric: under a
            narrow lens a pose and its twin Rz(180) R Rot180(axis) cast the same silhouette) on the visual hull first
            (the placed observers and the whole unit at each candidate's pose: a twin carves away what the others see),
            then on the unit members that agree and, for near ties, on the feature matches. Each round re-tracks and
            adjusts
  bundle    final adjustment at 1 px; every observer must agree with the others' silhouettes (epipolar tangency) and
            pass the hull test (it may not carve away what the others see; a weakly held one must also lie inside the
            hull the others carve); the worst offender is dropped, repeatedly. Dropped observers are grown back once
            (as units: a group placed wrongly through a doppelganger link keeps its internal states)

Result: R_rel, d_rel, registered (origin first), timings, seconds, status ("ok": every observer placed; "partial": the
origin and possibly others, see dropped; "origin_unplaced", "origin_unseen", "no_links"), dropped {index: reason} for
every observer not returned, warnings {index: text}; if the origin could not be placed, "unanchored" holds the placed
group in its own gauge; with viz=True also "viz" (illustrative matches, the view graph, the states after each stage,
coloured points, the bundle adjustments' RMS history).
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import cv2  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as tF  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

NAME = "hub-and-spoke"
THREADS = 8               # CPU threads for OpenCV, torch and numba
GPU_GB = None             # cap this process's share of the GPU (GB); None: what is free at warm-up less GPU_MARGIN_GB
GPU_MARGIN_GB = 0.75      # (past the free memory Windows pages GPU memory to system RAM and matching gets 3-10x slower;
                          # at the cap torch raises out-of-memory instead, and the batches are split)
DETECT_SCALE = 1.5        # the aircraft crop is upsampled before detection: finer keypoints, more of them
NUM_KPTS = 1024           # keypoints per image (the strongest inside the eroded mask)
ERODE_PX = 6              # keypoints keep this far inside the silhouette (points on the outline slide with the view)
ALL_PAIRS_MAX = 64        # up to this many observers every pair is matched; above, a shortlist ...
SHORTLIST = 12            # ... of each image's most similar images (mutual nearest neighbours of descriptors)
GUIDED_DEG = 50           # large swarms: more pairs where the estimate says the views overlap (this close) ...
GUIDED_K = 6              # ... up to this many new partners per observer and round
BATCH = 48                # LightGlue pairs per batch
MIN_MATCHES = 20          # LightGlue matches needed to try a pair
MIN_INLIERS = 25          # hub-model inliers needed for a view-graph edge
HUB_RATIO = 0.95          # ... and at least this fraction of the free essential matrix's inliers (see _two_view)
INLIER_PX = 1.5           # epipolar (Sampson) tolerance of an inlier
TANGENCY_PX = 8.0         # a pair's silhouettes must agree (outer epipolar tangents) within this many pixels
USE_SILHOUETTES = True    # tangency residuals of camera pairs in the bundle adjustment
MAX_TANGENCY_DEG = 140.0  # ... for pairs whose viewing directions are closer than this
GROW_ROUNDS = 10          # rounds of resection (observers outside the first reconstruction) + re-tracking + BA
                          # (growth stops as soon as a round places nobody)
MIN_RESECT = 12           # 2-D/3-D inliers needed to place an observer by resection
SIL_MEDIAN_PX = 2.5       # a placed camera's tangency residuals against the others: median ...
SIL_P80_PX = 6.0          # ... and 80th percentile (px)
HULL_DS = 8               # silhouette registration (no feature link): masks downsampled 8x for the search ...
HULL_MEDIAN_PX = 1.0      # ... and the stricter agreement such a camera must reach (median, 80th percentile px)
HULL_P80_PX = 3.0
HULL_PER_ROUND = 6        # silhouette registrations tried per growth round ...
HULL_TRIES = 3            # ... and per observer in one growth call (the origin's unit: every round)
BA_SIGMAS = (6.0, 2.0)    # robust scale (px) of the first bundle-adjustment rounds (then 8 -> 2 px per growth round)
FINAL_SIGMAS = (1.0,)     # ... and of the last adjustment
TAN_PX = 2.0              # robust scale (px) of the silhouette tangency residuals in the bundle adjustment
MAX_TRACK = 16            # observations kept per point (long tracks: the dense orbit)
MIN_CAM_OBS = 15          # a camera is placed if this many of its observations agree with the model ...
STRONG_OBS = 60           # ... this many make its feature evidence strong (a clipped silhouette: see the verification)
MIN_MASK_PX = 400         # an observer whose silhouette is smaller than this (pixels) does not see the target
ROLL_DEG = 40.0           # small swarms: every crop is also described turned in-plane by +-ROLL_DEG ...
ROLL_MAX_N = 40           # ... up to this many observers (larger swarms have enough close neighbours),
ROLL_SCALE = 1.0          # ... upsampled by this instead of DETECT_SCALE (they only serve the pairs with large roll)
LOO_DROP = 0.12           # hull test (_coverage): an observer's silhouette inside the others' hull may fall this far
HARM_MAX = 0.15           # below the swarm's median share, and it may carve away at most this share of another's
HULL_TEST_MIN = 6         # ... applied with at least this many placed observers (fewer carve too loose a hull)
TWIN_MARGIN = 0.02        # a silhouette-only placement must beat its mirror twins by this much hull coverage ...
TWIN_FEAT = 8             # ... or by this many more feature matches that agree with it
TWIN_MIN_DEG = 25.0       # a twin closer than this to the pose it mirrors is not a rival (just a nearby pose)
REGROW = True             # observers the verification dropped are grown back once (as units)
E3 = np.array([0.0, 0.0, 1.0])

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NET = {}
DEBUG = {}                # intermediate results of the last call, CPU arrays only (for diagnostics)
_VIZ = None               # the viz collector of the call in progress (None: viz=False, nothing is recorded)


# ----------------------------------------------------------------------------- small helpers

def skew(v):
    v = np.asarray(v, float)
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def expmap(w):
    w = np.asarray(w, float)
    return Rotation.from_rotvec(w.reshape(-1, 3)).as_matrix().reshape(w.shape[:-1] + (3, 3))


def inv3(A):
    """Inverses of a stack of 3x3 matrices (closed form: much faster than np.linalg.inv on many small blocks)."""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    co = np.stack([np.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
                   np.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
                   np.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1)], -2)
    det = a * co[..., 0, 0] + b * co[..., 1, 0] + c * co[..., 2, 0]
    return co / det[..., None, None]


def _angle_deg(A, B):
    """The angle (degrees) of the rotation between attitudes A and B."""
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def logmap(R):
    R = np.asarray(R, float)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


class _Clock:
    """Wall time per stage; tells `progress` when a stage ends."""

    def __init__(self, progress):
        self.progress, self.t0, self.last, self.timings = progress, time.perf_counter(), time.perf_counter(), {}

    def stage(self, name):
        """The time since the last call was spent on `name`. Stages can recur: `timings` adds them up, `progress` gets
        the seconds of this occurrence (so a caller that sums the callbacks gets the total)."""
        now = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0.0) + now - self.last
        if self.progress is not None and now - self.last > 1e-3:
            self.progress(name, now - self.last)
        self.last = now


# ----------------------------------------------------------------------------- models

def warm_up():
    """Load SuperPoint and LightGlue onto the GPU and run them once (kernel selection, cuDNN autotune)."""
    from lightglue import LightGlue, SuperPoint
    cv2.setNumThreads(THREADS)
    torch.set_num_threads(THREADS)
    numba.set_num_threads(min(THREADS, numba.config.NUMBA_NUM_THREADS))
    if _DEVICE.type == "cuda":
        free, total = torch.cuda.mem_get_info()                     # (+ this process's own cache, which the driver
        cap = GPU_GB * 2**30 if GPU_GB else free + torch.cuda.memory_reserved() - GPU_MARGIN_GB * 2**30   # counts as used)
        torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.05, cap / total)))
    if not _NET:
        _NET["sp"] = SuperPoint(max_num_keypoints=None).eval().to(_DEVICE)
        _NET["lg"] = LightGlue(features="superpoint", depth_confidence=0.95, width_confidence=-1).eval().to(_DEVICE)
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920), np.uint8)
    mask = np.zeros((1080, 1920), bool)
    mask[200:900, 300:1600] = True
    angles = _roll_angles(2)
    f = _group_versions(_features([_prepare(img, mask, a) for a in angles] * 2), 2, len(angles))
    _, best_v = _similarity(f)
    K = np.array([[2000.0, 0, 959.5], [0, 2000.0, 539.5], [0, 0, 1]])
    small = torch.ones(2, 1080 // HULL_DS, 1920 // HULL_DS, dtype=torch.bool, device=_DEVICE)
    X = _hull_points(np.tile(np.eye(3), (2, 1, 1)), np.ones(2), [0, 1], small, K, HULL_DS, np.full(2, 0.3), np.zeros(2, bool))
    _hull_iou(X[:100], np.tile(np.eye(3), (4, 1, 1)), 1.0, small[0], K, HULL_DS)
    i64 = np.int64                                                    # compile the BA and track kernels
    _union_tracks(np.zeros(1, i64), np.ones(1, i64), np.arange(2, dtype=i64), 2, 2)
    _schur_subtract(np.zeros((2, 4, 3)), np.zeros((2, 4, 3)), np.zeros(2, i64), np.arange(2, dtype=i64), np.array([0, 2], i64), np.zeros((4, 4)))
    _accumulate(np.zeros((2, 2, 4)), np.zeros((2, 2, 3)), np.zeros((2, 2)), np.ones(2), np.zeros(2, i64), np.zeros(2, i64), 1, 1)
    _reproject(np.tile(np.eye(3), (1, 1, 1)), np.ones(1), np.zeros((1, 3)), np.zeros(2, i64), np.zeros(2, i64), np.zeros((2, 2)), 1.0, 1.0, 0.0, 0.0)
    _sampson_rows(np.zeros((1, 3, 3)), np.zeros(2, i64), np.ones((2, 3)), np.ones((2, 3)))
    outl = _outline(mask, K)                                        # compile the tangency kernel
    _Silhouettes([outl, outl], K[0, 0]).relative(expmap(np.array([[0.0, 0.3, 0.0]])), np.ones(1), np.array([[0, 1]]))
    _match_batches([(0, 1)] * 4, f, best_v)
    _coverage(np.tile(np.eye(3), (2, 1, 1)), np.ones(2), [0, 1], small, K, HULL_DS, np.full(2, 0.3), np.zeros(2, bool))
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- 1. load

def _load(paths):
    image_path, mask_path = paths
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None:
        raise IOError(f"cannot read {image_path} or {mask_path}")
    return gray, mask > 127, cv2.resize(mask, (mask.shape[1] // HULL_DS, mask.shape[0] // HULL_DS), interpolation=cv2.INTER_AREA) > 127


# ----------------------------------------------------------------------------- 2. features on the aircraft only

def _superpoint_dense(net, x):
    """SuperPoint's dense keypoint scores (B, H, W) and descriptor map (B, 256, H/8, W/8)."""
    r = net.relu
    x = net.pool(r(net.conv1b(r(net.conv1a(x)))))
    x = net.pool(r(net.conv2b(r(net.conv2a(x)))))
    x = net.pool(r(net.conv3b(r(net.conv3a(x)))))
    x = r(net.conv4b(r(net.conv4a(x))))
    s = torch.softmax(net.convPb(r(net.convPa(x))), 1)[:, :-1]
    b, _, h, w = s.shape
    s = s.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
    return s, tF.normalize(net.convDb(r(net.convDa(x))), dim=1)


def _prepare(gray, mask, deg=0.0):
    """The crop SuperPoint sees (CPU, run in the thread pool): the aircraft's bounding box with the background blanked,
    upsampled by DETECT_SCALE, and where keypoints may go (the mask eroded by ERODE_PX).
    deg != 0: the image is first turned in-plane by deg about its centre c (p_turned = A (p - c) + c, pixel axes x right,
    y down); one affine warp goes straight from the original image to the crop (scale ROLL_SCALE) of the turned silhouette, with
    the same pixel-centre convention as the resize (crop + 0.5 = scale * (p_turned - offset + 0.5)). The keypoints are
    mapped back exactly: p = A^T (p_turned - c) + c (_features)."""
    ys, xs = np.nonzero(mask)
    x0, y0 = max(xs.min() - 16, 0), max(ys.min() - 16, 0)
    x1, y1 = min(xs.max() + 17, gray.shape[1]), min(ys.max() + 17, gray.shape[0])
    inner = cv2.erode(mask[y0:y1, x0:x1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1,) * 2))
    c = np.array([(gray.shape[1] - 1) / 2.0, (gray.shape[0] - 1) / 2.0])
    if deg == 0.0:
        h, w = int(round((y1 - y0) * DETECT_SCALE)), int(round((x1 - x0) * DETECT_SCALE))
        img = cv2.resize(gray[y0:y1, x0:x1] * mask[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
        keep = cv2.resize(inner, (w, h), interpolation=cv2.INTER_NEAREST)
        return dict(img=img, keep=keep, offset=(float(x0), float(y0)), scale=(w / (x1 - x0), h / (y1 - y0)),
                    size=gray.shape[::-1], A=np.eye(2), c=c)
    a = np.radians(deg)
    A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cs, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    P = (np.concatenate([q.reshape(-1, 2) for q in cs]) + (x0, y0) - c) @ A.T + c   # the outline, turned
    o, e = np.floor(P.min(0)) - 16, np.ceil(P.max(0)) + 17
    w, h = int(round((e[0] - o[0]) * ROLL_SCALE)), int(round((e[1] - o[1]) * ROLL_SCALE))
    sc = np.array([w / (e[0] - o[0]), h / (e[1] - o[1])])
    M = np.c_[sc[:, None] * A, sc * (c - o + 0.5) - 0.5 - (sc[:, None] * A) @ c]   # original pixel -> crop pixel
    M[:, 2] += M[:, :2] @ (x0, y0)                                           # ... from the bounding-box crop
    img = cv2.warpAffine(gray[y0:y1, x0:x1] * mask[y0:y1, x0:x1], M, (w, h), flags=cv2.INTER_LINEAR)
    keep = cv2.warpAffine(inner, M, (w, h), flags=cv2.INTER_NEAREST)
    return dict(img=img, keep=keep, offset=(float(o[0]), float(o[1])), scale=(float(sc[0]), float(sc[1])),
                size=gray.shape[::-1], A=A, c=c)


def _roll_angles(n):
    """The in-plane turns every crop is described in: (0, +ROLL_DEG, -ROLL_DEG) for small swarms, else (0,).
    With a pair matched in the best of the three, a relative roll up to ~60 deg is left at ~20 deg or less, which
    SuperPoint + LightGlue tolerate (51 % of the pool's pairs less than 40 deg apart differ by more than 20 deg of roll)."""
    return (0.0, ROLL_DEG, -ROLL_DEG) if n <= ROLL_MAX_N else (0.0,)


def _group_versions(flat, n, nv):
    """Per observer: its versions (turned crops) [dict(kp (turned frame, for LightGlue), desc, start)], and kp_np: the
    keypoints of all versions in original-image pixels, concatenated (matches index into it; `start` is the offset)."""
    feats = []
    for i in range(n):
        vers, kps, start = [], [], 0
        for v in range(nv):
            f = flat[i * nv + v]
            vers.append(dict(kp=f["kp"], desc=f["desc"], start=start))
            kps.append(f["kp_orig"].cpu().numpy().astype(float))
            start += len(f["kp"])
        feats.append(dict(size=flat[i * nv]["size"], vers=vers, kp_np=np.concatenate(kps)))
    return feats


def _features(preps, max_pixels=3.5e6):
    """NUM_KPTS SuperPoint keypoints per image on the aircraft: the strongest local maxima of the score map inside the
    eroded mask, with sub-pixel positions (a parabola through the neighbouring scores). Images go through the network
    in batches (crops padded to a common size, half precision). Returns per image the keypoints (full-image pixels,
    strongest first) and descriptors, on the GPU."""
    from lightglue.superpoint import sample_descriptors, simple_nms
    feats = [None] * len(preps)
    order = list(np.argsort([p["img"].size for p in preps]))
    batches = []                                          # similar sizes together, at most max_pixels per batch
    for k in order:
        b = batches[-1] if batches else None
        if b is not None:
            hh = max(max(preps[q]["img"].shape[0] for q in b), preps[k]["img"].shape[0])
            ww = max(max(preps[q]["img"].shape[1] for q in b), preps[k]["img"].shape[1])
        if b is None or (len(b) + 1) * hh * ww > max_pixels:
            batches.append([k])
        else:
            b.append(k)
    while batches:
        idx = batches.pop(0)
        H = -(-max(preps[k]["img"].shape[0] for k in idx) // 8) * 8          # SuperPoint needs multiples of 8
        W = -(-max(preps[k]["img"].shape[1] for k in idx) // 8) * 8
        img = np.zeros((len(idx), H, W), np.uint8)
        keep = np.zeros((len(idx), H, W), np.uint8)
        for a, k in enumerate(idx):
            h, w = preps[k]["img"].shape
            img[a, :h, :w], keep[a, :h, :w] = preps[k]["img"], preps[k]["keep"]
        with torch.no_grad():
            t = torch.from_numpy(img).to(_DEVICE)[:, None].float().div_(255)
            try:
                with torch.autocast(_DEVICE.type, dtype=torch.float16, enabled=_DEVICE.type == "cuda"):
                    scores, desc = _superpoint_dense(_NET["sp"], t)
                scores, desc = scores.float(), desc.float()
                nms = simple_nms(scores * torch.from_numpy(keep).to(_DEVICE), 4)
                nms[:, :4], nms[:, -4:], nms[:, :, :4], nms[:, :, -4:] = 0, 0, 0, 0
                val, top = torch.topk(nms.flatten(1), min(NUM_KPTS, H * W), dim=1)
            except torch.OutOfMemoryError:                 # the GPU is shared: one image at a time
                if len(idx) == 1:
                    raise
                batches = [[k] for k in idx] + batches
                continue
            for a, k in enumerate(idx):
                t_ = top[a][val[a] > 0]
                yy, xx = t_ // W, t_ % W
                sc = scores[a]
                c, l, r_, u, d = sc[yy, xx], sc[yy, xx - 1], sc[yy, xx + 1], sc[yy - 1, xx], sc[yy + 1, xx]
                dx = ((r_ - l) / (2 * (2 * c - l - r_)).clamp(min=1e-12)).clamp(-0.5, 0.5)
                dy = ((d - u) / (2 * (2 * c - u - d)).clamp(min=1e-12)).clamp(-0.5, 0.5)
                kp = torch.stack([xx + dx, yy + dy], -1).float()
                des = sample_descriptors(kp[None], desc[a:a + 1], 8)[0].t().contiguous()
                scale = torch.tensor(preps[k]["scale"], device=_DEVICE, dtype=torch.float32)
                kp_full = (kp + 0.5) / scale - 0.5 + torch.tensor(preps[k]["offset"], device=_DEVICE)
                A = torch.tensor(preps[k]["A"], device=_DEVICE, dtype=torch.float64)       # turned -> original frame
                cc = torch.tensor(preps[k]["c"], device=_DEVICE, dtype=torch.float64)
                kp_orig = (kp_full.double() - cc) @ A + cc
                feats[k] = dict(kp=kp_full, kp_orig=kp_orig, desc=des, size=preps[k]["size"])
    return feats


# ----------------------------------------------------------------------------- 3. pairs and LightGlue

def _similarity(feats, top=512):
    """How alike two images look, and in which version (in-plane turn) of the second one: the number of mutual nearest
    neighbours among their `top` strongest descriptors (cosine > 0.5), image i upright against every version of every
    image, one batched product per image. Returns the symmetric score (n, n) of the best versions (diagonal -1) and
    best_v (n, n): for the pair (i, j), the version of j to match with i's upright one (ties: upright)."""
    n, nv = len(feats), len(feats[0]["vers"])
    D = torch.zeros(n * nv, top, feats[0]["vers"][0]["desc"].shape[1], device=_DEVICE, dtype=torch.float16)
    for i, f in enumerate(feats):
        for v, ver in enumerate(f["vers"]):
            D[i * nv + v, :min(top, len(ver["desc"]))] = ver["desc"][:top].half()
    score = torch.zeros(n, n * nv, device=_DEVICE)
    ar = torch.arange(top, device=_DEVICE)
    with torch.no_grad():
        for i in range(n):
            sim = torch.einsum("kd,nld->nkl", D[i * nv], D)
            best, nn_ij = sim.max(2)
            nn_ji = sim.argmax(1)
            score[i] = ((nn_ji.gather(1, nn_ij) == ar) & (best > 0.5)).sum(1).float()
    score = score.view(n, n, nv).cpu().numpy()
    best_v = np.argmax(score, 2)
    s = score.max(2)
    s = np.minimum(s, s.T)
    np.fill_diagonal(s, -1)
    return s, best_v


def _shortlist(score, k=SHORTLIST):
    """Each image paired with its k most similar images."""
    pairs = set()
    for i in range(len(score)):
        for j in np.argsort(-score[i])[:k]:
            if score[i, j] > 0:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def _rematch_weak(ctx, pairs, nv, min_pairs=3):
    """The version chosen by the descriptor vote (_similarity) is occasionally wrong (a pair that gives 5 matches upright
    and 111 turned). For the origin, and every observer with fewer than min_pairs matched pairs, every pair of it that
    failed (fewer than MIN_MATCHES matches) is matched again in the other versions. Another version wins only on strong
    evidence (at least 2 MIN_MATCHES matches and 3x as many): weak gains are mostly doppelgangers of the symmetric
    aircraft, which can tie a group to the rest wrongly. Updates ctx.matches and ctx.best_v."""
    n, matches = ctx.n, ctx.matches
    good = [k for k, m in matches.items() if len(m) >= MIN_MATCHES]
    deg = np.bincount(np.array(good, int).ravel(), minlength=n) if good else np.zeros(n, int)
    weak = {0} | {int(i) for i in np.nonzero(deg < min_pairs)[0]}
    retry = [k for k in pairs if (k[0] in weak or k[1] in weak) and len(matches.get(k, ())) < MIN_MATCHES]
    for v in range(nv):
        sub = [k for k in retry if ctx.best_v[k] != v]
        if not sub:
            continue
        for k, m in _match_batches(sub, ctx.feats, np.full((n, n), v)).items():
            if len(m) >= 2 * MIN_MATCHES and len(m) >= 3 * len(matches.get(k, ())):
                matches[k] = m
                ctx.best_v[k] = v
    DEBUG["rematched"] = len(retry)


def _reach_pairs(score, member, matches, k=SHORTLIST // 2):
    """Every observer not placed yet with its k most similar placed observers it has not been matched with: the
    matches resection needs to place it."""
    pairs = set()
    placed = np.nonzero(member)[0]
    if len(placed) == 0:
        return []
    for c in np.nonzero(~member)[0]:
        ranked = placed[np.argsort(-score[c, placed])]
        new = [(min(c, int(j)), max(c, int(j))) for j in ranked if score[c, j] > 0 and (min(c, int(j)), max(c, int(j))) not in matches]
        pairs.update(new[:k])
    return sorted(pairs)


@torch.no_grad()
def _lightglue(lg, kp0, d0, kp1, d1, size, depth_confidence=0.95, threshold=0.1):
    """LightGlue on a batch of pairs with equal keypoint counts (no padding, so the attention runs on the fast path).
    A pair leaves the batch as soon as its tokens are confident (LightGlue's early exit, per pair).
    Returns a list of (k, 2) index tensors."""
    from lightglue.lightglue import filter_matches, normalize_keypoints
    B = kp0.shape[0]
    x0, x1 = lg.input_proj(d0), lg.input_proj(d1)
    e0, e1 = lg.posenc(normalize_keypoints(kp0, size)), lg.posenc(normalize_keypoints(kp1, size))
    out, active, L = [None] * B, torch.arange(B), lg.conf.n_layers
    n_tok = kp0.shape[1] + kp1.shape[1]
    for i in range(L):
        x0, x1 = lg.transformers[i](x0, x1, e0, e1)
        if i == L - 1:
            done = torch.ones(len(active), dtype=torch.bool, device=x0.device)
        else:
            c0, c1 = lg.token_confidence[i](x0, x1)
            thr = lg.confidence_thresholds[i]
            done = 1.0 - ((c0 < thr).sum(1) + (c1 < thr).sum(1)).float() / n_tok > depth_confidence
        if done.any():
            k = torch.nonzero(done)[:, 0]
            scores, _ = lg.log_assignment[i](x0[k], x1[k])
            m0, _, _, _ = filter_matches(scores.float(), threshold)
            for a, b in enumerate(k.tolist()):
                idx = torch.nonzero(m0[a] > -1)[:, 0]
                out[active[b]] = torch.stack([idx, m0[a][idx]], -1)
            keep = torch.nonzero(~done)[:, 0]
            if len(keep) == 0:
                break
            active, x0, x1, e0, e1 = active[keep.cpu()], x0[keep], x1[keep], e0[:, keep], e1[:, keep]
    return out


def _match_batches(pairs, feats, best_v=None, skipped=None):
    """LightGlue on every pair (i, j): i upright against version best_v[i, j] of j (in that version's turned frame, so
    the pair looks upright to LightGlue). A pair uses the k strongest keypoints of both, k = the smaller count rounded
    down to a multiple of 64, so pairs with equal k share a batch; a pair with fewer than 64 keypoints is not matched
    (noted in `skipped`). Returns {(i, j): (m, 2) int array} of indices into the observers' kp_np."""
    lg = _NET["lg"]
    groups = {}
    for i, j in pairs:
        v = int(best_v[i, j]) if best_v is not None else 0
        k = min(len(feats[i]["vers"][0]["kp"]), len(feats[j]["vers"][v]["kp"])) // 64 * 64
        if k >= 64:
            groups.setdefault(k, []).append((i, j, v))
        elif skipped is not None:
            skipped[(i, j)] = "fewer than 64 keypoints"
    out, fits = {}, {}                                      # fits[k]: the batch size that fits (after an out-of-memory)

    def run(chunk, k):
        kp0 = torch.stack([feats[i]["vers"][0]["kp"][:k] for i, _, _ in chunk])
        kp1 = torch.stack([feats[j]["vers"][v]["kp"][:k] for _, j, v in chunk])
        d0 = torch.stack([feats[i]["vers"][0]["desc"][:k] for i, _, _ in chunk])
        d1 = torch.stack([feats[j]["vers"][v]["desc"][:k] for _, j, v in chunk])
        size = torch.tensor(feats[chunk[0][0]]["size"], device=_DEVICE, dtype=torch.float32)
        res = None
        try:
            with torch.autocast(_DEVICE.type, dtype=torch.float16, enabled=_DEVICE.type == "cuda"):
                res = _lightglue(lg, kp0, d0, kp1, d1, size)
        except torch.OutOfMemoryError:                     # the GPU is shared: retry in halves (outside this block,
            if len(chunk) == 1:                            # so the failed attempt's tensors are released first)
                raise
        if res is None:
            del kp0, kp1, d0, d1
            torch.cuda.empty_cache()
            fits[k] = min(fits.get(k, BATCH), len(chunk) // 2)   # the next batches start at a size that fits
            run(chunk[:len(chunk) // 2], k)
            run(chunk[len(chunk) // 2:], k)
            return
        sizes = [len(m) for m in res]
        allm = torch.cat(res).cpu().numpy() if res else np.zeros((0, 2), int)  # one transfer for the batch
        for (i, j, v), m in zip(chunk, np.split(allm, np.cumsum(sizes)[:-1])):
            out[(i, j)] = m + np.array([0, feats[j]["vers"][v]["start"]])       # into kp_np (all versions)

    for k, prs in groups.items():
        s = 0
        while s < len(prs):
            step = fits.get(k, BATCH)
            run(prs[s:s + step], k)
            s += step
    return out


# ----------------------------------------------------------------------------- 4. two-view geometry, hub model

def hub_essential(R, rho):
    """E = [t]x R with t = rho e3 - R e3: camera j = R camera i, both looking at the hub, distance ratio rho = d_j/d_i."""
    return skew(rho[..., None] * E3 - R[..., :, 2]) @ R


def sampson(E, x0, x1):
    """Signed Sampson distance (normalised image units) of x0 <-> x1 (homogeneous rows) for per-row E."""
    Ex0 = np.einsum("nij,nj->ni", E, x0)
    Etx1 = np.einsum("nji,nj->ni", E, x1)
    return np.einsum("ni,ni->n", x1, Ex0) / np.sqrt(Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2 + 1e-30)


def pair_depths(R, rho, x0, x1):
    """Depth of each correspondence in camera i (unit = the hub's depth) and in camera j, per row."""
    t = rho[:, None] * E3 - R[:, :, 2]
    Rx = np.einsum("nij,nj->ni", R, x0)
    c1, c2 = np.cross(x1, Rx), np.cross(x1, t)
    z0 = -np.sum(c1 * c2, 1) / np.maximum(np.sum(c1 * c1, 1), 1e-30)
    return z0, Rx[:, 2] * z0 + t[:, 2]


def hub_from_essential(E):
    """The (R, rho) inside an essential matrix that keeps the hub in front of both cameras (t ~ rho e3 - R e3)."""
    best = None
    R1, R2, t = cv2.decomposeEssentialMat(E)
    for R in (R1, R2):
        A = np.c_[E3, -R @ E3]
        (lj, li), *_ = np.linalg.lstsq(A, t.ravel(), rcond=None)
        resid = np.linalg.norm(A @ [lj, li] - t.ravel())
        if li * lj > 0 and (best is None or resid < best[2]):
            best = (R, float(np.clip(lj / li, 0.25, 4.0)), resid)
    return best


def _outline(mask, K):
    """The silhouette's convex hull (ordered vertices, normalised homogeneous coordinates) in two variants, and its
    centroid. Thin spikes (the pitot needle: a pixel or two wide) are masked in some views and broken or missing in
    others, so a hull vertex at a needle tip is unreliable; the second variant leaves out the outline points farther
    than 4 px from the mask opened with a 7x7 square (which removes such spikes). The tangency residual of a pair
    takes whichever variant fits better. Where the silhouette touches the image border it is cut off by the frame, not
    by the object: the touched stretch of each border is returned (normalised units; see _tangent_pair)."""
    m8 = mask.astype(np.uint8)
    ys, xs = np.nonzero(m8)
    x0, y0 = max(xs.min() - 8, 0), max(ys.min() - 8, 0)
    crop = m8[y0:ys.max() + 9, x0:xs.max() + 9]
    cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c.reshape(-1, 2) for c in cs])
    opened = cv2.morphologyEx(crop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    near = cv2.distanceTransform((opened == 0).astype(np.uint8), cv2.DIST_L2, 5)[pts[:, 1], pts[:, 0]] <= 4.0
    Kinv = np.linalg.inv(K)
    hulls = []
    for p in (pts, pts[near] if near.sum() >= 3 else pts):
        h = cv2.convexHull(p.astype(np.float32)).reshape(-1, 2).astype(float) + (x0, y0)
        hulls.append(np.c_[h, np.ones(len(h))] @ Kinv.T)
    Hh, Ww = mask.shape
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    segs = np.zeros((4, 3))                                 # per border: its line coordinate, touched from .. to
    segs[:, 1], segs[:, 2] = 1.0, -1.0                      # (empty unless the mask touches it)
    for b, (line, along) in enumerate(((mask[0, :], ((0 - cy) / f, cx)), (mask[-1, :], ((Hh - 1 - cy) / f, cx)),
                                       (mask[:, 0], ((0 - cx) / f, cy)), (mask[:, -1], ((Ww - 1 - cx) / f, cy)))):
        segs[b, 0] = along[0]
        idx = np.nonzero(line)[0]
        if len(idx):
            segs[b, 1], segs[b, 2] = (idx.min() - along[1]) / f, (idx.max() - along[1]) / f
    return hulls, np.r_[hulls[0][:, :2].mean(0), 1.0], segs


def _pad_outlines(outlines, idx, size, variant):
    """Hulls (one variant) of the images `idx` as one (P, size, 3) array (short hulls repeat their last vertex)."""
    H = np.empty((len(idx), size, 3))
    for p, k in enumerate(idx):
        h = outlines[k][0][variant]
        H[p, :len(h)], H[p, len(h):] = h, h[-1]
    return H


@numba.njit(cache=True)
def _crosses_cut(l, segs, tol):
    """Does line l cross a stretch of image border that the silhouette touches (segs: top, bottom, left, right)?"""
    for b in range(4):
        if segs[b, 1] > segs[b, 2]:
            continue
        if b < 2:                                           # horizontal border y = Y: where does the line cross it?
            if abs(l[0]) < 1e-15:
                continue
            x = -(l[1] * segs[b, 0] + l[2]) / l[0]
        else:                                               # vertical border x = X
            if abs(l[1]) < 1e-15:
                continue
            x = -(l[0] * segs[b, 0] + l[2]) / l[1]
        if segs[b, 1] - tol <= x <= segs[b, 2] + tol:
            return True
    return False


@numba.njit(cache=True)
def _tangent_pair(Em, e, es, hs, ns, ss, hd, nd, sd, cd):
    """One direction of one pair: the two outer epipolar tangents of the src outline hs (ns vertices, normalised
    homogeneous), carried into the dst image by Em, against the dst outline hd's own outer tangents through the epipole
    e. Returns the two signed misses (normalised units: distance of the dst tangent point from the carried line),
    whether the test applies at all (no outer tangents when the epipole lies inside the dst hull) and, per tangent,
    whether neither image's tangent line crosses a stretch of border its silhouette touches (ss, sd; es is the epipole
    in the src image): there the frame, not the object, may bound the silhouette.
    A line through the epipole is placed by its key: its angle to the direction from the dst centroid towards the
    epipole, or, for a far epipole (nearly parallel lines), where it crosses the line through the centroid across that
    direction."""
    dx, dy = e[0] - e[2] * cd[0], e[1] - e[2] * cd[1]
    nrm = np.sqrt(dx * dx + dy * dy)
    if nrm == 0.0:
        return 0.0, 0.0, False, False, False
    dnx, dny = dx / nrm, dy / nrm
    far = nrm > 3.0 * abs(e[2])
    out = np.empty(4)                    # key and index of the smallest / largest key, src then dst
    best = np.empty((2, 2))
    idx = np.zeros((2, 2), np.int64)
    for side in range(2):
        best[side, 0], best[side, 1] = np.inf, -np.inf
        n = ns if side == 0 else nd
        for k in range(n):
            if side == 0:                # epipolar line of a src vertex: l = Em x
                x = hs[k]
                l0 = Em[0, 0] * x[0] + Em[0, 1] * x[1] + Em[0, 2] * x[2]
                l1 = Em[1, 0] * x[0] + Em[1, 1] * x[1] + Em[1, 2] * x[2]
                l2 = Em[2, 0] * x[0] + Em[2, 1] * x[1] + Em[2, 2] * x[2]
            else:                        # line from the epipole through a dst vertex: l = e x x
                x = hd[k]
                l0 = e[1] * x[2] - e[2] * x[1]
                l1 = e[2] * x[0] - e[0] * x[2]
                l2 = e[0] * x[1] - e[1] * x[0]
            if far:
                den = l0 * -dny + l1 * dnx
                if den == 0.0:
                    continue
                key = -(l0 * cd[0] + l1 * cd[1] + l2 * cd[2]) / den
            else:
                key = np.arctan2(-dnx * l0 - dny * l1, dnx * l1 - dny * l0)
                if key > np.pi / 2:
                    key -= np.pi
                elif key <= -np.pi / 2:
                    key += np.pi
            if key < best[side, 0]:
                best[side, 0], idx[side, 0] = key, k
            if key > best[side, 1]:
                best[side, 1], idx[side, 1] = key, k
        if best[side, 0] == np.inf:
            return 0.0, 0.0, False, False, False
    for m in range(2):                   # residual: dst tangent point against the carried src tangent line
        x = hs[idx[0, m]]
        l0 = Em[0, 0] * x[0] + Em[0, 1] * x[1] + Em[0, 2] * x[2]
        l1 = Em[1, 0] * x[0] + Em[1, 1] * x[1] + Em[1, 2] * x[2]
        l2 = Em[2, 0] * x[0] + Em[2, 1] * x[1] + Em[2, 2] * x[2]
        c = hd[idx[1, m]]
        out[m] = (l0 * c[0] + l1 * c[1] + l2 * c[2]) / max(np.sqrt(l0 * l0 + l1 * l1), 1e-300)
    if not far:                          # epipole inside the dst hull: no outer tangents
        ex, ey = e[0] / e[2], e[1] / e[2]
        pos, neg = True, True
        for k in range(nd):
            a, b = hd[k], hd[(k + 1) % nd]
            cr = (b[0] - a[0]) * (ey - a[1]) - (b[1] - a[1]) * (ex - a[0])
            pos &= cr >= -1e-12
            neg &= cr <= 1e-12
        if pos or neg:
            return out[0], out[1], False, False, False
    valid = np.ones(2, np.bool_)
    tol = 3e-3
    for m in range(2):
        x, c = hs[idx[0, m]], hd[idx[1, m]]
        ls = np.array([es[1] * x[2] - es[2] * x[1], es[2] * x[0] - es[0] * x[2], es[0] * x[1] - es[1] * x[0]])
        ld = np.array([e[1] * c[2] - e[2] * c[1], e[2] * c[0] - e[0] * c[2], e[0] * c[1] - e[1] * c[0]])
        lc = Em @ x                                         # the src tangent carried into dst, and back
        lcs = Em.T @ c
        valid[m] = not (_crosses_cut(ls, ss, tol) or _crosses_cut(ld, sd, tol) or _crosses_cut(lc, sd, tol) or _crosses_cut(lcs, ss, tol))
    return out[0], out[1], True, valid[0], valid[1]


@numba.njit(parallel=True, cache=True)
def _tangency_kernel(R, rho, I, J, hulls, nh, segs, cens, res, ok):
    """Tangency residuals of camera pairs (i, j) with relative poses R (j <- i) and distance ratios rho: res (P, 4)
    (both directions) and their validity ok (P, 4). Per residual the outline variant (axis 0 of hulls) with the smaller
    miss."""
    for p in numba.prange(len(R)):
        t = np.array([-R[p, 0, 2], -R[p, 1, 2], rho[p] - R[p, 2, 2]])          # baseline: t = rho e3 - R e3
        tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        E = tx @ R[p]
        Et = E.T.copy()
        ei = -(R[p].T @ t)                                                    # camera j's centre seen from i
        i, j = I[p], J[p]
        for k in range(4):
            res[p, k] = np.inf
            ok[p, k] = False
        for v in range(hulls.shape[0]):
            a0, a1, o1, va0, va1 = _tangent_pair(E, t, ei, hulls[v, i], nh[v, i], segs[i], hulls[v, j], nh[v, j], segs[j], cens[j])
            b0, b1, o2, vb0, vb1 = _tangent_pair(Et, ei, t, hulls[v, j], nh[v, j], segs[j], hulls[v, i], nh[v, i], segs[i], cens[i])
            if o1 and o2:
                cand = (a0, a1, b0, b1)
                valid = (va0, va1, vb0, vb1)
                for k in range(4):
                    if valid[k] and abs(cand[k]) < abs(res[p, k]):
                        res[p, k] = cand[k]
                        ok[p, k] = True
        for k in range(4):
            if not ok[p, k]:
                res[p, k] = 0.0


class _Silhouettes:
    """The silhouette outlines of all observers (convex hulls in two variants), for the epipolar tangency residuals of
    camera pairs (a compiled kernel: the BA evaluates thousands of pairs many times)."""

    def __init__(self, outlines, f):
        size = max(len(h) for o in outlines for h in o[0])
        n = len(outlines)
        self.hulls = np.stack([_pad_outlines(outlines, range(n), size, v) for v in (0, 1)])
        self.segs = np.stack([o[2] for o in outlines])
        self.nh = np.array([[len(o[0][v]) for o in outlines] for v in (0, 1)], np.int64)
        self.cens = np.stack([o[1] for o in outlines])
        self.f = f

    def relative(self, Rrel, rho, pairs):
        """(P, 4) tangency residuals in pixels of camera pairs (i, j) with relative poses R_j R_i^T and d_j / d_i, and
        validity. Per residual, the outline variant that fits better (a needle tip broken off in one mask)."""
        P = len(pairs)
        res, ok = np.empty((P, 4)), np.empty((P, 4), np.bool_)
        if P:
            _tangency_kernel(np.ascontiguousarray(Rrel, dtype=np.float64), np.ascontiguousarray(rho, dtype=np.float64),
                             np.ascontiguousarray(pairs[:, 0], dtype=np.int64), np.ascontiguousarray(pairs[:, 1], dtype=np.int64),
                             self.hulls, self.nh, self.segs, self.cens, res, ok)
        return self.f * res, ok

    def residuals(self, R, d, pairs):
        """The same under absolute states (R, d)."""
        i, j = pairs[:, 0], pairs[:, 1]
        return self.relative(R[j] @ np.transpose(R[i], (0, 2, 1)), d[j] / d[i], pairs)

    def pairs(self, R, d, cams, rng, per_cam=32, max_deg=MAX_TANGENCY_DEG):
        """Camera pairs whose tangency can be tested: viewing directions less than max_deg apart (for nearly opposite
        cameras the epipole sits close to the silhouette and the tangents swing wildly) and no epipole inside a
        silhouette; for many cameras a random subset of per_cam partners per camera."""
        cams = np.asarray(cams)
        P = np.array([(a, b) for k, a in enumerate(cams) for b in cams[k + 1:]], int).reshape(-1, 2)
        if len(P) == 0:
            return P
        if len(cams) > per_cam + 1:
            P = P[rng.random(len(P)) < 2.0 * per_cam / (len(cams) - 1)]
        P = P[np.einsum("pi,pi->p", R[P[:, 0], 2], R[P[:, 1], 2]) > np.cos(np.radians(max_deg))]
        _, ok = self.residuals(R, d, P)
        return P[ok.any(1)]


def _refine_pairs(x0, x1, pid, R, rho, f, sil, pairs, scale_px=1.0, tan_px=2.0, iters=12):
    """Levenberg-Marquardt on the 4 hub-model unknowns of every pair at once. Residuals: the Sampson error r (px) of the
    matches under a Cauchy loss (s^2/2) log(1 + r^2/s^2), s = scale_px, plus the 4 silhouette tangency residuals (px)
    under a pseudo-Huber loss t^2 (sqrt(1 + r^2/t^2) - 1), t = tan_px; they pin the orbit angle the matches alone
    leave loose (pairs: the (i, j) of every pair, for the silhouettes). The normal equations are the IRLS form of
    exactly this cost (weights rho'(r)/r = 1/(1 + r^2/s^2) and 1/sqrt(1 + r^2/t^2)), so every LM step is a descent
    step of the cost its acceptance test compares.
    Returns R, rho, the tangency residuals (px) and each pair's 3x3 attitude information matrix (distance ratio
    marginalised out), for perturbations R <- exp(w) R."""
    P, h = len(R), 1e-6
    R, s, lam = R.copy(), np.log(rho), np.full(len(R), 1e-3)
    pid_t = np.repeat(np.arange(P), 4)

    pid64 = pid.astype(np.int64)

    def resid(Rp, sp):
        rs = f * _sampson_rows(np.ascontiguousarray(hub_essential(Rp, np.exp(sp))), pid64, x0, x1)
        rt, ok = sil.relative(Rp, np.exp(sp), pairs)
        return rs, np.where(ok, rt, 0.0).ravel()

    def cost(rs, rt):                                   # per pair; its IRLS weights are the ws, wt of normal()
        return np.bincount(pid, 0.5 * scale_px ** 2 * np.log1p((rs / scale_px) ** 2), P) + \
            np.bincount(pid_t, tan_px ** 2 * (np.sqrt(1 + (rt / tan_px) ** 2) - 1), P)

    def normal(Rp, sp, rs, rt):
        Js, Jt = np.empty((len(rs), 4)), np.empty((len(rt), 4))
        for k in range(4):
            a, b = resid(expmap(np.eye(3)[k] * h) @ Rp, sp) if k < 3 else resid(Rp, sp + h)
            Js[:, k], Jt[:, k] = (a - rs) / h, (b - rt) / h
        ws, wt = 1.0 / (1.0 + (rs / scale_px) ** 2), 1.0 / np.sqrt(1.0 + (rt / tan_px) ** 2)
        A = np.stack([np.bincount(pid, ws * Js[:, a] * Js[:, b], P) + np.bincount(pid_t, wt * Jt[:, a] * Jt[:, b], P)
                      for a in range(4) for b in range(4)], 1).reshape(P, 4, 4)
        g = np.stack([np.bincount(pid, ws * rs * Js[:, a], P) + np.bincount(pid_t, wt * rt * Jt[:, a], P)
                      for a in range(4)], 1)
        return A, g

    rs, rt = resid(R, s)
    c = cost(rs, rt)
    for _ in range(iters):
        A, g = normal(R, s, rs, rt)
        for _ in range(5):
            D = A + lam[:, None, None] * (np.eye(4) * np.diagonal(A, axis1=1, axis2=2)[:, None, :] + 1e-9 * np.eye(4))
            step = -np.linalg.solve(D, g[:, :, None])[:, :, 0]
            Rn, sn = expmap(step[:, :3]) @ R, np.clip(s + step[:, 3], -1.5, 1.5)
            cn = cost(*resid(Rn, sn))
            better = cn <= c
            R[better], s[better], c[better] = Rn[better], sn[better], cn[better]
            lam = np.where(better, lam * 0.3, lam * 10)
            if better.all():
                break
        rs, rt = resid(R, s)
    A, _ = normal(R, s, rs, rt)
    info = A[:, :3, :3] - A[:, :3, 3:] @ A[:, 3:, :3] / np.maximum(A[:, 3:, 3:], 1e-12)
    rt, ok = sil.relative(R, np.exp(s), pairs)
    return R, np.exp(s), np.where(ok, np.abs(rt), np.nan), info


def _two_view(feats, matches, K, pool, sil):
    """Every matched pair -> (R_ij, rho_ij, information, inlier matches) under the hub model, or nothing.
    The free essential matrix (MAGSAC, 5 unknowns) only selects the inliers: for a small object seen through a narrow
    lens its decomposition is ambiguous (several (R, t) fit almost equally). The hub model (4 unknowns) is then fitted
    to all pairs at once, with the silhouettes' epipolar tangency fixing the orbit angle; a pair whose silhouettes
    still disagree is not trusted (this also rejects most doppelgangers: views of opposite sides that match)."""
    Kinv, f = np.linalg.inv(K), K[0, 0]
    kps = [x["kp_np"] for x in feats]

    def job(key):
        m = matches[key]
        if len(m) < MIN_MATCHES:
            return None
        p0, p1 = kps[key[0]][m[:, 0]], kps[key[1]][m[:, 1]]
        E, inl = cv2.findEssentialMat(p0, p1, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0, maxIters=1000)
        if E is None or E.shape != (3, 3) or inl.sum() < MIN_INLIERS:
            return None
        hub = hub_from_essential(E)
        return None if hub is None else (hub[0], hub[1], inl.ravel() > 0)

    keys = list(matches)
    res = [(k, r) for k, r in zip(keys, pool.map(job, keys)) if r is not None]
    if not res:
        return []
    X0, X1, PID = [], [], []
    for p, ((i, j), (_, _, inl)) in enumerate(res):
        m = matches[(i, j)][inl]
        X0.append(np.c_[kps[i][m[:, 0]], np.ones(len(m))] @ Kinv.T)
        X1.append(np.c_[kps[j][m[:, 1]], np.ones(len(m))] @ Kinv.T)
        PID.append(np.full(len(m), p))
    pairs = np.array([k for k, _ in res])
    R, rho, tan, info = _refine_pairs(np.concatenate(X0), np.concatenate(X1), np.concatenate(PID),
                                      np.stack([r[0] for _, r in res]), np.array([r[1] for _, r in res]), f, sil, pairs)
    edges = []
    for p, ((i, j), (_, _, inl_free)) in enumerate(res):
        m = matches[(i, j)]
        x0 = np.c_[kps[i][m[:, 0]], np.ones(len(m))] @ Kinv.T
        x1 = np.c_[kps[j][m[:, 1]], np.ones(len(m))] @ Kinv.T
        Rp, rp = np.broadcast_to(R[p], (len(m), 3, 3)), np.full(len(m), rho[p])
        err = f * np.abs(sampson(hub_essential(Rp, rp), x0, x1))
        z0, z1 = pair_depths(Rp, rp, x0, x1)
        inl = (err < INLIER_PX) & (z0 > 0.5) & (z0 < 2.0) & (z1 > 0.5 * rho[p]) & (z1 < 2.0 * rho[p])
        tan_err = np.nanmax(tan[p]) if np.isfinite(tan[p]).any() and R[p][2, 2] > np.cos(np.radians(MAX_TANGENCY_DEG)) else np.nan
        DEBUG.setdefault("all_pairs", []).append(dict(i=i, j=j, R=R[p], rho=rho[p], n_hub=int(inl.sum()), n_free=int(inl_free.sum()), tan=tan_err, info=info[p]))
        if inl.sum() >= MIN_INLIERS and inl.sum() >= HUB_RATIO * inl_free.sum() and tan_err <= TANGENCY_PX:
            edges.append(dict(i=i, j=j, R=R[p], rho=rho[p], info=info[p], m=m[inl], n_free=int(inl_free.sum()), tan=tan_err))
    return edges


# ----------------------------------------------------------------------------- 5. global attitudes and distances

def _edge_weights(info, floor_deg=0.2):
    """Per-edge 3x3 weight for rotation averaging: the two-view information matrix (1 px noise; matches and silhouette
    tangency), every direction capped at a noise floor of floor_deg (model and keypoint bias), so no edge dominates."""
    lam, V = np.linalg.eigh(info)
    lam = np.clip(lam, 1e-3, np.radians(floor_deg) ** -2)
    return (V * lam[:, None, :]) @ np.transpose(V, (0, 2, 1))


def _edge_residuals(R, I, J, Rij, W):
    """Per edge: e = log(R_j R_i^T R_ij^T) (camera j's axes) and its Mahalanobis norm under W."""
    e = logmap(R[J] @ np.transpose(R[I], (0, 2, 1)) @ np.transpose(Rij, (0, 2, 1)))
    return e, np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", e, W, e), 0))


def _chain(n, I, J, Rij, weight, root=0):
    """Attitudes chained along the maximum spanning tree (by weight) from `root` (R_j = R_ij R_i)."""
    tree = minimum_spanning_tree(sp.coo_matrix((-weight, (I, J)), shape=(n, n)).tocsr())
    tree = tree + tree.T
    lookup = {(i, j): k for k, (i, j) in enumerate(zip(I, J))}
    order, parent = breadth_first_order(tree, root, directed=False)
    R = np.tile(np.eye(3), (n, 1, 1))
    for v in order[1:]:
        p = parent[v]
        R[v] = Rij[lookup[(p, v)]] @ R[p] if (p, v) in lookup else Rij[lookup[(v, p)]].T @ R[p]
    return R


def _average_rotations(R, I, J, Rij, W, anchor, c=4.0, iters=20):
    """Anisotropic robust rotation averaging (Gauss-Newton + IRLS, the anchor fixed): with R_k <- exp(w_k) R_k every
    edge asks e_ij + w_j - (R_j R_i^T) w_i = 0, weighted by its 3x3 W and a Cauchy weight on its Mahalanobis residual."""
    n, m = len(R), len(I)
    R = R.copy()
    free = np.arange(n) != anchor
    for _ in range(iters):
        e, r = _edge_residuals(R, I, J, Rij, W)
        Wr = W * (1.0 / (1.0 + (r / c) ** 2))[:, None, None]
        Rji = R[J] @ np.transpose(R[I], (0, 2, 1))
        Ai, Aj = -Rji, np.tile(np.eye(3), (m, 1, 1))
        H = np.zeros((3 * n, 3 * n))
        g = np.zeros(3 * n)
        for (a, Aa) in ((I, Ai), (J, Aj)):
            for (b, Ab) in ((I, Ai), (J, Aj)):
                blk = np.transpose(Aa, (0, 2, 1)) @ Wr @ Ab
                rows = (3 * a)[:, None, None] + np.arange(3)[None, :, None]
                cols = (3 * b)[:, None, None] + np.arange(3)[None, None, :]
                np.add.at(H, (np.broadcast_to(rows, blk.shape), np.broadcast_to(cols, blk.shape)), blk)
            np.add.at(g, (3 * a)[:, None] + np.arange(3), (np.transpose(Aa, (0, 2, 1)) @ Wr @ e[:, :, None])[:, :, 0])
        f3 = np.repeat(free, 3)
        H, g = H[np.ix_(f3, f3)] + 1e-6 * np.eye(3 * n - 3), g[f3]
        w = -cho_solve(cho_factor(H), g)
        R[free] = expmap(w.reshape(-1, 3)) @ R[free]
        if np.abs(w).max() < 1e-7:
            break
    return R


def _global_attitudes(n, edges, rng, trees=24, c=4.0):
    """Attitudes of the largest group of observers the verified pairs link together, in the gauge of its best-linked
    member (the anchor). A spanning tree gives a first guess; since one wrong link in the tree (a doppelganger pair)
    would carry everything behind it, several randomly perturbed maximum spanning trees are tried and the one most
    edges agree with wins (RANSAC over trees). Then robust anisotropic averaging; edges it cannot explain are dropped
    and the averaging is repeated. Returns R, the members, the edges kept, the anchor."""
    I = np.array([e["i"] for e in edges])
    J = np.array([e["j"] for e in edges])
    Rij = np.stack([e["R"] for e in edges])
    ninl = np.array([len(e["m"]) for e in edges], float)
    W = _edge_weights(np.stack([e["info"] for e in edges]))

    def largest(keep):
        comp = connected_components(sp.coo_matrix((np.ones(keep.sum()), (I[keep], J[keep])), shape=(n, n)), directed=False)[1]
        size = np.bincount(comp, minlength=n).astype(float)
        size[comp[0]] += 0.5                                   # ties go to the origin's group
        member = comp == np.argmax(size)
        deg = np.bincount(np.r_[I[keep], J[keep]], minlength=n) * member
        return member, int(np.argmax(deg))

    member, anchor = largest(np.ones(len(I), bool))
    best = None
    for t in range(trees):
        wt = ninl * (1.0 if t == 0 else rng.uniform(0.3, 1.0, len(ninl)))
        R = _chain(n, I, J, Rij, wt, anchor)
        _, r = _edge_residuals(R, I, J, Rij, W)
        support = np.sum(np.minimum(r, 3 * c) ** 2 * member[I])
        if best is None or support < best[0]:
            best = (support, R)
    R = best[1]
    R_chain = R.copy()
    keep = member[I] & member[J]
    for _ in range(2):
        R = _average_rotations(R, I[keep], J[keep], Rij[keep], W[keep], anchor, c)
        _, r = _edge_residuals(R, I, J, Rij, W)
        keep = member[I] & member[J] & (r < 3 * c)
    member, anchor2 = largest(keep)
    if not member[anchor]:                                     # the averaging split the group: re-anchor
        R = R @ R[anchor2].T
        R_chain = R_chain @ R_chain[anchor2].T
        anchor = anchor2
    keep &= member[I] & member[J]
    R = _average_rotations(R, I[keep], J[keep], Rij[keep], W[keep], anchor, c)   # once more with the final edge set
    return R, member, keep, anchor, R_chain


def _view_groups(n, edges, rng):
    """Every group of observers the verified pairs link, largest first: dict(R, d, member, kept (edges), anchor, R_chain),
    each in the gauge of its best-linked member. The first one is where the reconstruction starts; the others are UNITS:
    their internal states are known, so once one member is placed the rest follow (_bring_units), and a silhouette-only
    placement is judged on all of them at once (a mirror twin is not a rigid move of a unit: _register_by_silhouette)."""
    groups, avail = [], np.ones(len(edges), bool)
    I = np.array([e["i"] for e in edges])
    J = np.array([e["j"] for e in edges])
    while avail.any():
        sub = [e for e, a in zip(edges, avail) if a]
        R, member, keep, anchor, R_chain = _global_attitudes(n, sub, rng)
        kept = [e for e, k in zip(sub, keep) if k]
        if member.sum() < 2 or not kept:
            break
        d = _global_distances(n, np.array([e["i"] for e in kept]), np.array([e["j"] for e in kept]),
                              np.array([e["rho"] for e in kept]), np.array([len(e["m"]) for e in kept], float), anchor)
        groups.append(dict(R=R, d=d, member=member, kept=kept, anchor=anchor, R_chain=R_chain))
        avail &= ~(member[I] | member[J])
    return groups


def _global_distances(n, I, J, rho, weight, anchor, iters=10, sigma=0.01):
    """log d_j - log d_i = log rho_ij, Cauchy-weighted least squares, log d_anchor = 0."""
    m = len(I)
    free = np.arange(n) != anchor
    A = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[J, I])), shape=(m, n))[:, free]
    logd = np.zeros(n)
    for _ in range(iters):
        wt = weight / (1.0 + ((np.log(rho) - logd[J] + logd[I]) / sigma) ** 2)
        Aw = A.multiply(wt[:, None]).tocsr()
        logd[free] = sp.linalg.spsolve((Aw.T @ A + 1e-9 * sp.eye(n - 1)).tocsc(), Aw.T @ np.log(rho))
    return np.exp(logd)


# ----------------------------------------------------------------------------- 6. tracks, bundle adjustment

@numba.njit(cache=True)
def _union_tracks(a, b, img, n_nodes, n_imgs):
    """Union-find over keypoints (global indices) joined by the matches a[k] <-> b[k], taken in order: a merge that
    would put two keypoints of one image into one track is refused (one of the matches must be wrong), so a single bad
    match cannot glue unrelated points or wipe out a whole track. Returns each keypoint's track root."""
    parent = np.arange(n_nodes)
    size = np.ones(n_nodes, np.int64)
    nxt = np.full(n_nodes, -1, np.int64)                  # each set's members as a linked list from its root
    last = np.arange(n_nodes)
    stamp = np.zeros(n_imgs, np.int64)
    for k in range(len(a)):
        ra = a[k]
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        rb = b[k]
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        if ra == rb:
            continue
        small, large = (ra, rb) if size[ra] <= size[rb] else (rb, ra)
        x = small
        while x != -1:                                    # images of the smaller set ...
            stamp[img[x]] = k + 1
            x = nxt[x]
        x, clash = large, False
        while x != -1:                                    # ... must not appear in the larger one
            if stamp[img[x]] == k + 1:
                clash = True
                break
            x = nxt[x]
        if clash:
            continue
        parent[small] = large
        nxt[last[large]] = small
        last[large] = last[small]
        size[large] += size[small]
    for x in range(n_nodes):
        r = x
        while parent[r] != r:
            r = parent[r]
        parent[x] = r
    return parent


def _tracks(feats, pair_matches, n_cams):
    """Matches (i, j, (m, 2) keypoint indices) joined into tracks, the pairs with the most matches first (their matches
    win the conflicts). Returns per observation: track, camera, keypoint index, pixel."""
    offset = np.r_[0, np.cumsum([len(f["kp_np"]) for f in feats])]
    pair_matches = sorted(pair_matches, key=lambda x: -len(x[2]))
    a = np.concatenate([offset[i] + m[:, 0] for i, j, m in pair_matches]).astype(np.int64)
    b = np.concatenate([offset[j] + m[:, 1] for i, j, m in pair_matches]).astype(np.int64)
    N = int(offset[-1])
    img = np.repeat(np.arange(len(feats)), np.diff(offset)).astype(np.int64)
    root = _union_tracks(a, b, img, N, len(feats))                 # (a, b, img are int64: one compiled signature)
    used = np.zeros(N, bool)
    used[a] = used[b] = True
    node = np.nonzero(used)[0]
    _, trk, length = np.unique(root[node], return_inverse=True, return_counts=True)
    ok = length[trk] >= 2
    if length.max() > MAX_TRACK:                           # a point seen by very many observers: keep MAX_TRACK of
        rank = np.empty(len(node), np.int64)               # its views (the BA's cost grows with the square of a
        order = np.lexsort((np.random.default_rng(0).random(len(node)), img[node] != 0, trk))   # track's length; they
        # add little); the origin's view is always kept: its own error turns every relative attitude
        rank[order] = np.arange(len(node)) - np.r_[0, np.cumsum(np.bincount(trk))][trk[order]]
        ok &= rank < MAX_TRACK
    node = node[ok]
    cam = img[node]
    kp = np.concatenate([f["kp_np"] for f in feats])[node]
    return np.unique(trk[ok], return_inverse=True)[1], cam, node - offset[cam], kp


def _track_matches(feats, matches, edges, R, d, member, K, px=2.0, edge_deg=5.0):
    """Matches for the tracks of the placed observers: the inliers of every verified two-view edge whose relative pose
    agrees with the current states (within edge_deg and 3 % in distance; verified pair by pair, so they also hold an
    observer whose state is still rough), plus every LightGlue match that agrees with the current states (epipolar
    error below px, in front of both cameras within the object's depth range)."""
    out = {}
    for e in edges:
        i, j = e["i"], e["j"]
        if member[i] and member[j]:
            dR = R[j] @ R[i].T @ e["R"].T
            if np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < edge_deg and abs(np.log(d[j] / d[i] / e["rho"])) < 0.03:
                out[(i, j)] = e["m"]
    for i, j, m in _consistent_matches(feats, matches, R, d, member, K, px):
        out[(i, j)] = np.unique(np.concatenate([out[(i, j)], m]), axis=0) if (i, j) in out else m
    return [(i, j, m) for (i, j), m in out.items()]


def _consistent_matches(feats, matches, R, d, member, K, px=2.0):
    """The LightGlue matches of every pair of placed observers that agree with the current states: epipolar (Sampson)
    error below px and the point in front of both cameras, within the object's depth range."""
    keys = [k for k, m in matches.items() if member[k[0]] and member[k[1]] and len(m)]
    if not keys:
        return []
    Kinv = np.linalg.inv(K)
    m_all = [matches[k] for k in keys]
    pid = np.concatenate([np.full(len(m), p) for p, m in enumerate(m_all)])
    x0 = np.c_[np.concatenate([feats[i]["kp_np"][m[:, 0]] for (i, _), m in zip(keys, m_all)]), np.ones(len(pid))] @ Kinv.T
    x1 = np.c_[np.concatenate([feats[j]["kp_np"][m[:, 1]] for (_, j), m in zip(keys, m_all)]), np.ones(len(pid))] @ Kinv.T
    I, J = np.array([k[0] for k in keys]), np.array([k[1] for k in keys])
    Rrel, rho = (R[J] @ np.transpose(R[I], (0, 2, 1)))[pid], (d[J] / d[I])[pid]
    err = K[0, 0] * np.abs(sampson(hub_essential(Rrel, rho), x0, x1))
    z0, z1 = pair_depths(Rrel, rho, x0, x1)
    ok = (err < px) & (z0 > 0.5) & (z0 < 2.0) & (z1 > 0.5 * rho) & (z1 < 2.0 * rho)
    starts = np.r_[0, np.cumsum([len(m) for m in m_all])]
    out = []
    for p, ((i, j), m) in enumerate(zip(keys, m_all)):
        k = ok[starts[p]:starts[p + 1]]
        if k.sum() >= 3:
            out.append((i, j, m[k]))
    return out


def _project(R, d, X, cam, trk, K):
    P = np.einsum("nij,nj->ni", R[cam], X[trk]) + d[cam, None] * E3
    return P[:, :2] / P[:, 2:] * K[[0, 1], [0, 1]] + K[:2, 2], P


def _triangulate(R, d, cam, trk, kp, K, T):
    """Per track, the point closest (least squares) to its viewing rays."""
    v = np.einsum("nji,nj->ni", R[cam], np.c_[(kp - K[:2, 2]) / K[[0, 1], [0, 1]], np.ones(len(kp))])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    C = -d[cam, None] * R[cam, 2, :]
    M = np.eye(3)[None] - v[:, :, None] * v[:, None, :]
    A = np.stack([np.bincount(trk, M.reshape(-1, 9)[:, k], T) for k in range(9)], 1).reshape(T, 3, 3)
    b = np.stack([np.bincount(trk, np.einsum("nij,nj->ni", M, C)[:, k], T) for k in range(3)], 1)
    return np.linalg.solve(A + 1e-9 * np.eye(3), b[:, :, None])[:, :, 0]


@numba.njit(parallel=True, cache=True)
def _reproject(R, d, X, cam, trk, kp, fx, fy, cx, cy):
    """Per observation: the reprojection residual r (2), its derivatives with respect to the camera's 4 unknowns
    (rotation w with R <- exp(w) R, and log d) and to the point (3), and the squared error (1e6 behind the camera)."""
    n = len(cam)
    r, Jc, Jx, e2 = np.empty((n, 2)), np.empty((n, 2, 4)), np.empty((n, 2, 3)), np.empty(n)
    for o in numba.prange(n):
        c, t = cam[o], trk[o]
        rx = R[c, 0, 0] * X[t, 0] + R[c, 0, 1] * X[t, 1] + R[c, 0, 2] * X[t, 2]
        ry = R[c, 1, 0] * X[t, 0] + R[c, 1, 1] * X[t, 1] + R[c, 1, 2] * X[t, 2]
        rz = R[c, 2, 0] * X[t, 0] + R[c, 2, 1] * X[t, 1] + R[c, 2, 2] * X[t, 2]
        z = rz + d[c]
        if z <= 1e-6:
            z = 1e-6
        u, v = rx / z, ry / z
        r[o, 0], r[o, 1] = u * fx + cx - kp[o, 0], v * fy + cy - kp[o, 1]
        e2[o] = r[o, 0] ** 2 + r[o, 1] ** 2 if rz + d[c] > 1e-6 else 1e6
        # d(pixel)/d(X_cam): [[fx/z, 0, -fx u/z], [0, fy/z, -fy v/z]]
        a0, a2 = fx / z, -fx * u / z
        b1, b2 = fy / z, -fy * v / z
        # rotation: d X_cam / dw = -[RX]x  ->  row k of J = P_k . (-[RX]x)
        Jc[o, 0, 0] = a2 * ry                   # P0 = (a0, 0, a2):  -P0 x-row products
        Jc[o, 0, 1] = a0 * rz - a2 * rx
        Jc[o, 0, 2] = -a0 * ry
        Jc[o, 1, 0] = -b1 * rz + b2 * ry
        Jc[o, 1, 1] = -b2 * rx
        Jc[o, 1, 2] = b1 * rx
        Jc[o, 0, 3], Jc[o, 1, 3] = a2 * d[c], b2 * d[c]   # log d: d X_cam = d e3
        for j in range(3):                       # point: P R
            Jx[o, 0, j] = a0 * R[c, 0, j] + a2 * R[c, 2, j]
            Jx[o, 1, j] = b1 * R[c, 1, j] + b2 * R[c, 2, j]
    return r, Jc, Jx, e2


@numba.njit(parallel=True, cache=True)
def _sampson_rows(E, pid, x0, x1):
    """Signed Sampson distances (normalised units) of the correspondences x0 <-> x1 (homogeneous rows), each under its
    pair's essential matrix E[pid]."""
    n = len(pid)
    out = np.empty(n)
    for k in numba.prange(n):
        e = E[pid[k]]
        a0 = e[0, 0] * x0[k, 0] + e[0, 1] * x0[k, 1] + e[0, 2] * x0[k, 2]
        a1 = e[1, 0] * x0[k, 0] + e[1, 1] * x0[k, 1] + e[1, 2] * x0[k, 2]
        a2 = e[2, 0] * x0[k, 0] + e[2, 1] * x0[k, 1] + e[2, 2] * x0[k, 2]
        b0 = e[0, 0] * x1[k, 0] + e[1, 0] * x1[k, 1] + e[2, 0] * x1[k, 2]
        b1 = e[0, 1] * x1[k, 0] + e[1, 1] * x1[k, 1] + e[2, 1] * x1[k, 2]
        out[k] = (x1[k, 0] * a0 + x1[k, 1] * a1 + x1[k, 2] * a2) / np.sqrt(a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1 + 1e-30)
    return out


@numba.njit(cache=True)
def _accumulate(Jc, Jx, r, w, ccol, trk, nc, T):
    """The BA's normal equations, observation by observation: camera blocks U (nc, 4, 4) and gc (nc, 4), point blocks
    V (T, 3, 3) and gp (T, 3), and the camera-point coupling Y = Jc^T W Jx (n, 4, 3)."""
    n = len(trk)
    U, gc = np.zeros((nc, 4, 4)), np.zeros((nc, 4))
    V, gp = np.zeros((T, 3, 3)), np.zeros((T, 3))
    Y = np.zeros((n, 4, 3))
    for o in range(n):
        t, c, wo = trk[o], ccol[o], w[o]
        for i in range(3):
            gp[t, i] += wo * (Jx[o, 0, i] * r[o, 0] + Jx[o, 1, i] * r[o, 1])
            for j in range(3):
                V[t, i, j] += wo * (Jx[o, 0, i] * Jx[o, 0, j] + Jx[o, 1, i] * Jx[o, 1, j])
        if c < 0:
            continue
        for i in range(4):
            gc[c, i] += wo * (Jc[o, 0, i] * r[o, 0] + Jc[o, 1, i] * r[o, 1])
            for j in range(4):
                U[c, i, j] += wo * (Jc[o, 0, i] * Jc[o, 0, j] + Jc[o, 1, i] * Jc[o, 1, j])
            for j in range(3):
                Y[o, i, j] = wo * (Jc[o, 0, i] * Jx[o, 0, j] + Jc[o, 1, i] * Jx[o, 1, j])
    return U, gc, V, gp, Y


@numba.njit(cache=True)
def _schur_subtract(Z, Y, ccol, order, starts, S):
    """S -= sum over points of W_t V_t^-1 W_t^T: for every pair of observations (a, b) of one point, the 4x4 block
    (camera of a, camera of b) loses Z_a Y_b^T (Z = Y V^-1); fixed cameras (ccol < 0) have no block."""
    for t in range(len(starts) - 1):
        for p in range(starts[t], starts[t + 1]):
            a = order[p]
            ca = ccol[a]
            if ca < 0:
                continue
            for q in range(starts[t], starts[t + 1]):
                b = order[q]
                cb = ccol[b]
                if cb < 0:
                    continue
                for i in range(4):
                    for j in range(4):
                        S[4 * ca + i, 4 * cb + j] -= Z[a, i, 0] * Y[b, j, 0] + Z[a, i, 1] * Y[b, j, 1] + Z[a, i, 2] * Y[b, j, 2]


def _bundle(R, d, X, cam, trk, kp, K, scale_px, free, sil=None, sil_pairs=None, tan_px=2.0, iters=15, record=True):
    """Levenberg-Marquardt over the state model: per camera a small rotation w (3) and log d (1), per point X (3);
    cameras with free = False stay put (the anchor). Cauchy-robust reprojection error (scale_px), plus the epipolar
    tangency residuals of the silhouettes of camera pairs sil_pairs (also Cauchy, scale tan_px): they hold the orbit
    angles that two-view tracks leave loose. The points are eliminated (Schur complement: their 3x3 blocks are
    independent), the 4N x 4N camera system is solved densely. With viz on, the RMS reprojection error (px) of every
    accepted step is recorded."""
    N, T, n = len(R), len(X), len(cam)
    fxy = K[[0, 1], [0, 1]]
    ci, nc = np.cumsum(free) - 1, int(free.sum())
    use_sil = sil is not None and sil_pairs is not None and len(sil_pairs) > 0
    if use_sil:                                     # the 8 unknowns of each pair (-1: fixed camera)
        col8 = np.concatenate([np.where(free[sil_pairs[:, k], None], 4 * ci[sil_pairs[:, k], None] + np.arange(4), -1) for k in (0, 1)], 1)

    cam64, trk64, kpc = cam.astype(np.int64), trk.astype(np.int64), np.ascontiguousarray(kp, dtype=np.float64)

    def evaluate(R, d, X):
        r, Jc, Jx, e2 = _reproject(np.ascontiguousarray(R), np.ascontiguousarray(d), np.ascontiguousarray(X), cam64,
                                   trk64, kpc, fxy[0], fxy[1], K[0, 2], K[1, 2])
        cost = 0.5 * scale_px ** 2 * float(np.sum(np.log1p(e2 / scale_px ** 2)))
        rt = None
        if use_sil:
            rt, _ = sil.residuals(R, d, sil_pairs)
            cost += 0.5 * tan_px ** 2 * float(np.sum(np.log1p((rt / tan_px) ** 2)))
        return r, Jc, Jx, e2, cost, rt

    def silhouette_normal(R, d, rt):
        """J^T W J (P, 8, 8) and J^T W r (P, 8) of the tangency residuals (numeric derivatives through the relative
        pose: camera i turned by exp(w) gives R_j R_i^T exp(-w), camera j gives exp(w) R_j R_i^T)."""
        h = 1e-6
        i, j = sil_pairs[:, 0], sil_pairs[:, 1]
        Rrel, rho = R[j] @ np.transpose(R[i], (0, 2, 1)), d[j] / d[i]
        dR = expmap(np.eye(3) * h)                                   # the 8 perturbations, evaluated in one batch
        Rs = np.concatenate([Rrel @ dR[k].T for k in range(3)] + [Rrel] + [dR[k] @ Rrel for k in range(3)] + [Rrel])
        rhos = np.concatenate([rho] * 3 + [rho * np.exp(-h)] + [rho] * 3 + [rho * np.exp(h)])
        P = len(sil_pairs)
        rr = sil.relative(Rs, rhos, np.tile(sil_pairs, (8, 1)))[0].reshape(8, P, 4)
        J = np.transpose((rr - rt[None]) / h, (1, 2, 0))
        w = 1.0 / (1.0 + (rt / tan_px) ** 2)
        return np.einsum("pra,pr,prb->pab", J, w, J), np.einsum("pra,pr,pr->pa", J, w, rt)

    r, Jc, Jx, e2, cost, rt = evaluate(R, d, X)
    ccol = np.where(free[cam], ci[cam], -1).astype(np.int64)    # camera block of each observation (-1: fixed)
    order = np.argsort(trk, kind="stable").astype(np.int64)     # observations grouped by point, for the Schur kernel
    starts = np.r_[0, np.cumsum(np.bincount(trk, minlength=T))].astype(np.int64)
    lam = 1e-4
    for _ in range(iters):
        w = 1.0 / (1.0 + e2 / scale_px ** 2)
        Ub, gc, V, gp, Y = _accumulate(Jc, Jx, r, w, ccol, trk64, nc, T)
        k = ccol >= 0
        U = np.zeros((4 * nc, 4 * nc))
        for c in range(nc):
            U[4 * c:4 * c + 4, 4 * c:4 * c + 4] = Ub[c]
        gc = gc.ravel()
        if use_sil:
            Ht, gt = silhouette_normal(R, d, rt)
            ok = col8 >= 0
            pa, ra, rb = np.nonzero(ok[:, :, None] & ok[:, None, :])
            np.add.at(U, (col8[pa, ra], col8[pa, rb]), Ht[pa, ra, rb])
            pg, rg = np.nonzero(ok)
            np.add.at(gc, col8[pg, rg], gt[pg, rg])
        while True:
            Vinv = inv3(V + lam * V * np.eye(3) + 1e-9 * np.eye(3))
            Z = np.einsum("nik,nkj->nij", Y, Vinv[trk])                # Jc^T W Jx V^-1 per observation
            S = U + lam * np.diag(np.diag(U)) + 1e-9 * np.eye(4 * nc)
            _schur_subtract(np.ascontiguousarray(Z), Y, ccol, order, starts, S)   # S -= W V^-1 W^T, point by point
            zg = np.einsum("nij,nj->ni", Z, gp[trk])
            rhs = -gc + np.stack([np.bincount(ccol[k], zg[k, q], nc) for q in range(4)], 1).ravel()
            try:
                dc = cho_solve(cho_factor(S), rhs).reshape(nc, 4)
            except np.linalg.LinAlgError:
                lam *= 10
                if lam > 1e8:
                    return R, d, X
                continue
            ydc = np.zeros((n, 3))
            ydc[k] = np.einsum("nij,ni->nj", Y[k], dc[ccol[k]])           # W^T dc per observation
            back = np.stack([np.bincount(trk, ydc[:, q], T) for q in range(3)], 1)
            dp = np.einsum("tij,tj->ti", Vinv, -gp - back)
            Rn, dn = R.copy(), d.copy()
            Rn[free] = expmap(dc[:, :3]) @ R[free]
            dn[free] = d[free] * np.exp(dc[:, 3])
            new = evaluate(Rn, dn, X + dp)
            if new[4] < cost:
                gain = (cost - new[4]) / cost
                R, d, X, (r, Jc, Jx, e2, cost, rt) = Rn, dn, X + dp, new
                lam = max(lam * 0.3, 1e-9)
                if _VIZ is not None and record:                       # RMS of the observations within 3 sigma
                    inl = e2 < 9 * scale_px ** 2
                    _VIZ["history"].append(float(np.sqrt(e2[inl].mean())) if inl.any() else float("nan"))
                step = max(float(np.abs(dc[:, :3]).max()), float(np.abs(dc[:, 3]).max())) if nc else 0.0
                break
            lam *= 10
            if lam > 1e8:
                return R, d, X
        if gain < 1e-4 or step < 2e-5:                     # cameras settled (2e-5 rad = 0.001 deg)
            break
    return R, d, X


def _adjust(R, d, cam, trk, kp, K, member, anchor, sil, rng, sigmas=BA_SIGMAS):
    """Robust schedule: per round triangulate, trust the observations within 3 sigma, adjust with Cauchy scale sigma
    (together with the silhouette tangency of the placed cameras). Returns R, d, the points, which observations are
    trusted and every observation's reprojection error."""
    T = trk.max() + 1
    use = member[cam]
    free = member.copy()
    free[anchor] = False
    sil_pairs = None
    if sil is not None:      # (giving the origin every partner was tried: it over-weights its outline, orbit 0.056 -> 0.080 deg)
        sil_pairs = sil.pairs(R, d, np.nonzero(member)[0], rng, per_cam=32 if len(R) <= ALL_PAIRS_MAX else 12)
    for sigma in sigmas:
        X = _triangulate(R, d, cam[use], trk[use], kp[use], K, T)
        uv, P = _project(R, d, X, cam, trk, K)
        err = np.where(P[:, 2] > 0, np.linalg.norm(uv - kp, axis=1), np.inf)
        ok = member[cam] & (err < 3 * sigma)
        ok &= np.bincount(trk[ok], minlength=T)[trk] >= 2
        pts, tr = np.unique(trk[ok], return_inverse=True)
        R, d, Xp = _bundle(R, d, X[pts], cam[ok], tr, kp[ok], K, sigma, free, sil, sil_pairs, tan_px=TAN_PX)
        X[pts] = Xp
        use = ok
        DEBUG.setdefault("ba_rounds", []).append(dict(sigma=sigma, R=R.copy(), d=d.copy(), n_obs=int(ok.sum()), n_pts=len(pts)))
    uv, P = _project(R, d, X, cam, trk, K)
    err = np.where(P[:, 2] > 0, np.linalg.norm(uv - kp, axis=1), np.inf)
    ok = member[cam] & (err < 3 * sigmas[-1])
    ok &= np.bincount(trk[ok], minlength=T)[trk] >= 2
    return R, d, X, ok, err


# ----------------------------------------------------------------------------- 7. growth: resection, silhouette registration

def _resect(uv, X, K, rng, thr=4.0, iters=250):
    """A camera's state from 2-D/3-D correspondences. Each RANSAC hypothesis is P3P on two correspondences plus the hub
    (world origin <-> principal point), so it already looks at the hub (t = (0, 0, d)); the best one is refined on its
    inliers (4 unknowns). Returns (R, d, inlier mask) or None."""
    from scipy.optimize import least_squares
    n = len(X)
    if n < 6:
        return None
    fxy, c = K[[0, 1], [0, 1]], K[:2, 2]

    def inliers(Rm, dm):
        P = X @ Rm.T + dm * E3
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.linalg.norm(P[:, :2] / P[:, 2:] * fxy + c - uv, axis=1)
        return (P[:, 2] > 0) & (e < thr)

    best, it, need = (0, None), 0, iters
    while it < min(need, iters):
        it += 1
        if it == 80 and best[0] < 6:                        # nothing consistent after 80 hypotheses: give up
            break
        s = rng.choice(n, 2, replace=False)
        try:
            _, rvecs, tvecs = cv2.solveP3P(np.vstack([np.zeros(3), X[s]]), np.vstack([c, uv[s]]), K, None, flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        for rv, tv in zip(rvecs, tvecs):
            if tv[2, 0] <= 0:
                continue
            Rm = cv2.Rodrigues(rv)[0]
            k = int(inliers(Rm, tv[2, 0]).sum())
            if k > best[0]:
                best = (k, (Rm, float(tv[2, 0])))
                need = np.log(1e-4) / np.log(max(1 - (k / n) ** 2, 1e-12))
    if best[1] is None or best[0] < 6:
        return None
    Rm, dm = best[1]
    for _ in range(2):                                      # least squares on the inliers (4 unknowns), re-select
        inl = inliers(Rm, dm)

        def resid(p):
            P = X[inl] @ (expmap(p[:3]) @ Rm).T + dm * np.exp(p[3]) * E3
            return (P[:, :2] / P[:, 2:] * fxy + c - uv[inl]).ravel()
        p = least_squares(resid, np.zeros(4), loss="soft_l1", f_scale=thr / 3).x
        Rm, dm = expmap(p[:3]) @ Rm, dm * np.exp(p[3])
    return Rm, dm, inliers(Rm, dm)


def _silhouette_agreement(sil, R, d, cams, others):
    """Per camera in `cams`: median and 80th percentile of the absolute tangency residuals (px) against `others`, and
    the number of pairs that could be tested."""
    out = []
    for c in cams:
        o = np.array([b for b in others if b != c], int)
        P = np.c_[np.full(len(o), c), o].astype(int).reshape(-1, 2)
        P = P[np.einsum("pi,pi->p", R[P[:, 0], 2], R[P[:, 1], 2]) > np.cos(np.radians(MAX_TANGENCY_DEG))]
        if len(P) == 0:
            out.append((np.nan, np.nan, 0))
            continue
        r, ok = sil.residuals(R, d, P)
        a = np.abs(r[ok])
        out.append((float(np.median(a)), float(np.percentile(a, 80)), int(ok.any(1).sum())) if a.size else (np.nan, np.nan, 0))
    return out


def _register(feats, matches, member, R, d, X, cam, trk, kidx, ok, K, sil, clipped, rng):
    """Place observers outside the reconstruction by resection: their LightGlue matches with placed observers whose
    keypoints belong to triangulated points give 2-D/3-D correspondences. A new camera must also agree with the placed
    cameras' silhouettes (it would otherwise carve the hull wrongly). Returns the newly placed observers."""
    n = len(member)
    track_of = np.full((n, max(len(f["kp_np"]) for f in feats)), -1)
    track_of[cam[ok], kidx[ok]] = trk[ok]
    cands = {}
    for (i, j), m in matches.items():
        if len(m) == 0:
            continue
        for a, b, ma, mb in ((i, j, m[:, 0], m[:, 1]), (j, i, m[:, 1], m[:, 0])):
            if not member[a] and member[b]:
                t = track_of[b, mb]
                cands.setdefault(a, []).append(np.c_[ma[t >= 0], t[t >= 0]])
    added = []
    for c, lst in sorted(cands.items(), key=lambda kv: -sum(len(x) for x in kv[1])):
        corr = np.unique(np.concatenate(lst), axis=0)
        if len(corr) < MIN_RESECT:
            continue
        res = _resect(feats[c]["kp_np"][corr[:, 0]], X[corr[:, 1]], K, rng)
        if res is None or res[2].sum() < MIN_RESECT:
            continue
        Rc, dc = R.copy(), d.copy()
        Rc[c], dc[c] = res[0], res[1]
        med, p80, cnt = _silhouette_agreement(sil, Rc, dc, [c], np.nonzero(member)[0])[0]
        DEBUG.setdefault("resect", []).append((c, len(corr), int(res[2].sum()), med, p80, cnt))
        if cnt >= 2 and med < SIL_MEDIAN_PX and (p80 < 2 * SIL_P80_PX or (clipped[c] and res[2].sum() >= STRONG_OBS)):
            R[c], d[c] = res[0], res[1]
            added.append(c)
    return added


def _hull_points(R, d, cams, small, K, ds, reach, clipped, G=56):
    """Voxel centres of the visual hull carved from the observers `cams` (downsampled masks `small`, on the GPU), in
    the hub frame. The grid spans a cube that holds the object: half its side is the farthest any silhouette reaches
    from the image centre, times that observer's distance. An observer whose silhouette is cut off by its frame
    (clipped) does not carve outside its frame: it cannot tell what is there."""
    f, c = float(K[0, 0]), K[:2, 2]
    rad = 1.1 * float(np.max(reach[cams] * d[cams]))
    ax = torch.linspace(-rad, rad, G, device=_DEVICE)
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    keep = torch.ones(len(X), dtype=torch.bool, device=_DEVICE)
    H, W = small.shape[1:]
    for k in cams:
        P = X @ torch.tensor(R[k], dtype=torch.float32, device=_DEVICE).T
        P[:, 2] += float(d[k])
        u, v = (P[:, 0] / P[:, 2] * f + c[0]) / ds, (P[:, 1] / P[:, 2] * f + c[1]) / ds
        seen = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[:, 2] > 0)
        inside = small[k][v.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]
        keep &= (seen & inside) | (~seen & bool(clipped[k]))
    return X[keep]


def _hull_iou(X, Rc, dc, mask, K, ds):
    """IoU between a silhouette (downsampled) and the hull points projected (splatted) by each candidate attitude."""
    f, c = float(K[0, 0]), K[:2, 2]
    H, W = mask.shape
    out = []
    for s in range(0, len(Rc), 256):
        Rt = torch.tensor(Rc[s:s + 256], dtype=torch.float32, device=_DEVICE)
        P = torch.einsum("nij,pj->npi", Rt, X)
        z = P[..., 2] + dc
        u, v = (P[..., 0] / z * f + c[0]) / ds, (P[..., 1] / z * f + c[1]) / ds
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        flat = torch.where(inside, v.long().clamp(0, H - 1) * W + u.long().clamp(0, W - 1), H * W)   # off-frame: bin H*W
        img = torch.zeros(len(Rt), H * W + 1, dtype=torch.bool, device=_DEVICE)
        img.scatter_(1, flat, True)
        img = tF.max_pool2d(img[:, :H * W].view(len(Rt), 1, H, W).float(), 3, 1, 1)[:, 0] > 0     # close the splat's gaps
        out.append(((img & mask).sum((1, 2)).float() / (img | mask).sum((1, 2)).float()).cpu().numpy())
    return np.concatenate(out)


def _hull_register(c, member, R, d, small, size, reach, clipped, sil, K, ds, rng, unit=None, samples=4000):
    """Place observer c from its silhouette alone: carve the visual hull of the placed observers, find the attitude
    whose projection of the hull best overlaps c's silhouette (IoU; random attitudes, then a local search around the
    best, at the distance the silhouette sizes suggest), then refine attitude and distance on the epipolar tangency
    with every placed camera. With a unit, every candidate moves the whole unit rigidly and the IoU is averaged over
    its unplaced members: several silhouettes pin the pose far better than one. Returns the hull points (numpy) and the
    refined starts [(tangency median, 80th percentile, pairs, R_c, d_c)], best first, or None if the hull is too small."""
    others = np.nonzero(member)[0]
    X = _hull_points(R, d, others, small, K, ds, reach, clipped)
    if len(X) < 50:
        return None
    X = X[torch.from_numpy(rng.permutation(len(X))[:6000]).to(_DEVICE)]   # the seeded rng: reproducible calls
    mem = [c] + ([int(k) for k in unit["members"] if k != c and not member[k]] if unit is not None else [])
    mem = mem[:1] + sorted(mem[1:], key=lambda k: -size[k])[:3]         # the unit's 4 largest silhouettes suffice
    rel = [unit["R"][k] @ unit["R"][c].T if unit is not None else np.eye(3) for k in mem]   # R_k = rel_k R_c
    dsc = [unit["d"][k] / unit["d"][c] if unit is not None else 1.0 for k in mem]          # d_k = dsc_k d_c
    ref = float(np.median(size[others] * d[others]))
    dc0 = float(np.median([ref / size[k] / s for k, s in zip(mem, dsc)]))

    def iou(cand, dc):
        return sum(_hull_iou(X, np.matmul(rk, cand), dc * s, small[k], K, ds) for k, rk, s in zip(mem, rel, dsc)) / len(mem)

    cand = Rotation.random(samples if len(mem) == 1 else samples // 2, random_state=int(rng.integers(1 << 30))).as_matrix()
    best = []
    for scale in (0.9, 1.0, 1.1):
        sc = iou(cand, dc0 * scale)
        best += [(sc[k], cand[k], dc0 * scale) for k in np.argsort(-sc)[:6]]
    starts = []
    for _, Rc, dc in sorted(best, key=lambda b: -b[0])[:5]:
        loc = expmap(rng.normal(0, np.radians(6), (400, 3))) @ Rc
        loc[0] = Rc
        sc = iou(loc, dc)
        starts.append((sc.max(), loc[int(np.argmax(sc))], dc))
    out = []
    for _, Rc, dc in sorted(starts, key=lambda b: -b[0])[:3]:
        r = _refine_on_silhouettes(c, Rc, dc, R, d, others, sil, K)
        if r is not None:
            out.append(r)
    return X.cpu().numpy(), sorted(out, key=lambda o: o[0])


def _refine_on_silhouettes(c, Rc, dc, R, d, others, sil, K):
    """Attitude and distance of observer c refined on the epipolar tangency with the placed cameras `others` (they stay
    put). Returns (median, 80th percentile, tested pairs, R_c, d_c), or None if fewer than 3 pairs can be tested."""
    Rx, dx = R.copy(), d.copy()
    Rx[c], dx[c] = Rc, dc
    pairs = np.c_[np.full(len(others), c), others]
    pairs = pairs[np.einsum("i,oi->o", Rx[c, 2], Rx[others, 2]) > np.cos(np.radians(MAX_TANGENCY_DEG))]
    if len(pairs) < 3:
        return None
    free = np.zeros(len(R), bool)
    free[c] = True
    hub = (np.zeros((1, 3)), np.zeros(2, int), np.zeros(2, int), np.tile(K[:2, 2], (2, 1)))   # the hub, seen at the
    Rb, db, _ = _bundle(Rx, dx, *hub, K, 1.0, free, sil, pairs, iters=30, record=False)       # principal point
    med, p80, cnt = _silhouette_agreement(sil, Rb, db, [c], others)[0]
    return (med, p80, cnt, Rb[c], db[c]) if cnt >= 3 else None


# ----------------------------------------------------------------------------- 8. the hull test, mirror twins, units

def _coverage(R, d, cams, small, K, ds, reach, clipped, G=64):
    """Do the observers `cams` carve a visual hull that explains each one's silhouette? One carving pass on the GPU
    (G^3 voxels) counts, per voxel, the observers that reject it, and which one if only one does. Then per observer q,
    on its downsampled mask, the share of its silhouette covered by the projection of
      full: the hull all of them carve,
      loo:  the hull the OTHERS carve (leave-one-out: a wrong pose is not explained by what the others see),
    and per observer c its harm: the most it uncovers of another observer's silhouette (how much that one's coverage
    rises if c is left out: a wrong pose carves away what the others see). A mirror twin can pass the tangency test at
    sub-pixel level and still fail here (the real tools' coverage check flags such a hull). Returns (loo, full, harm)."""
    cams = [int(k) for k in cams]
    nc = len(cams)
    f, c0 = float(K[0, 0]), K[:2, 2]
    rad = 1.1 * float(np.max(reach[cams] * d[cams]))
    ax = torch.linspace(-rad, rad, G, device=_DEVICE)
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    H, W = small.shape[1:]

    def project(k):
        P = X @ torch.tensor(R[k], dtype=torch.float32, device=_DEVICE).T
        z = P[:, 2] + float(d[k])
        u, v = (P[:, 0] / z * f + c0[0]) / ds, (P[:, 1] / z * f + c0[1]) / ds
        seen = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        return seen, v.long().clamp(0, H - 1) * W + u.long().clamp(0, W - 1)

    count = torch.zeros(len(X), dtype=torch.int32, device=_DEVICE)
    who = torch.zeros(len(X), dtype=torch.int64, device=_DEVICE)
    for q, k in enumerate(cams):
        seen, flat = project(k)
        rej = ~((seen & small[k].view(-1)[flat]) | (~seen & bool(clipped[k])))   # a clipped view cannot carve off-frame
        count += rej.int()
        who += rej.long() * q
    base, single = count == 0, count == 1
    loo, full, harm = np.ones(nc), np.ones(nc), np.zeros(nc)
    for q, k in enumerate(cams):
        seen, flat = project(k)
        step = f * (2 * rad / (G - 1)) / float(d[k]) / ds             # voxel spacing in the downsampled image
        ksz = 2 * int(np.ceil(step / 2)) + 1                        # a splat that closes the gaps between voxels
        m = small[k]
        tot = m.sum().clamp(min=1).float()
        b = torch.zeros(H * W, device=_DEVICE)
        b[flat[base & seen]] = 1.0
        b = tF.max_pool2d(b.view(1, 1, H, W), ksz, 1, ksz // 2)[0, 0] > 0
        s = single & seen                                           # voxels only one observer rejects, by rejecter
        e = torch.zeros(nc * H * W, device=_DEVICE)
        e[who[s] * (H * W) + flat[s]] = 1.0
        e = tF.max_pool2d(e.view(nc, 1, H, W), ksz, 1, ksz // 2)[:, 0] > 0
        gain = ((e & ~b & m).sum((1, 2)).float() / tot).cpu().numpy()   # q's coverage gained by leaving out each one
        full[q] = float((b & m).sum() / tot)
        loo[q] = full[q] + gain[q]
        gain[q] = 0.0
        harm = np.maximum(harm, gain)
    return loo, full, harm


def _twins(Rc, X):
    """The mirror twins of attitude Rc. A body symmetric about a plane through the hub (normal a) casts, under a narrow
    lens (nearly orthographic), the same silhouette from R and from Rz(180) R Rot180(a): Rot180(a) = -reflection(a) maps
    the body onto its point reflection, and Rz(180) undoes the point reflection in the image. The two are
    2 * angle(viewing direction, a) apart: up to 180 deg for views near the symmetry plane (from above, below, ahead,
    behind). The plane is not known: a = each principal axis of the placed observers' hull points X."""
    X = np.asarray(X, np.float64)                          # (the hull points come from the GPU in float32)
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Rz = np.diag([-1.0, -1.0, 1.0])
    twins = []
    for a in Vt:
        T = Rz @ Rc @ (2.0 * np.outer(a, a) - np.eye(3))
        U, _, Wt = np.linalg.svd(T)                         # exactly orthonormal again
        twins.append(U @ Wt)
    return twins


def _feature_support(cams, Rn, dn, member, feats, matches, K, px=2.0):
    """How many LightGlue matches of the observers `cams` (at states Rn, dn) with the placed observers agree with those
    states (epipolar error below px, in front of both cameras within the object's depth range)."""
    mem = member.copy()
    mem[list(cams)] = True
    sub = {k: m for k, m in matches.items() if (k[0] in cams and member[k[1]]) or (k[1] in cams and member[k[0]])}
    return sum(len(m) for _, _, m in _consistent_matches(feats, sub, Rn, dn, mem, K, px))


def _move_unit(unit, c, Rc, dc, R, d, placed=None):
    """States if observer c goes to (Rc, dc) and its unit (if any) comes along: the unit turns rigidly about the hub and
    scales (camera k: R_k <- R_k R_c^T Rc, d_k <- d_k dc / d_c, with the unit's internal states). Members that are
    already placed (`placed`, a boolean mask) keep their own states: moving them would judge a candidate against
    cameras that are not where the reconstruction has them."""
    Rn, dn = R.copy(), d.copy()
    Rn[c], dn[c] = Rc, dc
    if unit is not None:
        k = np.asarray(unit["members"])
        if placed is not None:
            k = k[~placed[k]]
        Rn[k] = unit["R"][k] @ unit["R"][c].T @ Rc
        dn[k] = unit["d"][k] * dc / unit["d"][c]
    return Rn, dn


def _refine_gauge(mem, Rn, dn, others, sil, iters=15, tan_px=TAN_PX):
    """A unit `mem` moved rigidly about the hub to fit the epipolar tangency with the placed cameras `others`: one turn
    G and one scale s for all its members (R_k <- R_k G, d_k <- s d_k: the unit's world turned and scaled), so its
    internal states stay as the features measured them. Levenberg-Marquardt on the 4 unknowns, Cauchy weights, numeric
    derivatives. Returns the refined states."""
    pairs = np.array([(k, o) for k in mem for o in others], int).reshape(-1, 2)
    pairs = pairs[np.einsum("pi,pi->p", Rn[pairs[:, 0], 2], Rn[pairs[:, 1], 2]) > np.cos(np.radians(MAX_TANGENCY_DEG))]
    if len(pairs) < 3:
        return Rn, dn

    def states(p):
        Rx, dx = Rn.copy(), dn.copy()
        Rx[mem] = Rn[mem] @ expmap(p[:3])
        dx[mem] = dn[mem] * np.exp(p[3])
        return Rx, dx

    def resid(p):
        r, ok = sil.residuals(*states(p), pairs)
        return np.where(ok, r, 0.0).ravel()

    def cost(r):
        return float(np.sum(np.log1p((r / tan_px) ** 2)))

    p, lam, h = np.zeros(4), 1e-3, 1e-6
    r = resid(p)
    c = cost(r)
    for _ in range(iters):
        J = np.stack([(resid(p + h * np.eye(4)[k]) - r) / h for k in range(4)], 1)
        w = 1.0 / (1.0 + (r / tan_px) ** 2)
        A, g = J.T @ (w[:, None] * J), J.T @ (w * r)
        while lam < 1e8:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-9 * np.eye(4), g)
            rn = resid(p + step)
            if cost(rn) < c:
                p, r, c, lam = p + step, rn, cost(rn), lam * 0.3
                break
            lam *= 10
        if lam >= 1e8 or np.abs(step).max() < 1e-7:
            break
    return states(p)


def _agrees(med, p80, cnt):
    return cnt >= 2 and med < SIL_MEDIAN_PX and p80 < SIL_P80_PX


def _bring_units(ctx, units, placed_now, R, d, member):
    """Observers just placed (by resection) bring their units along; every member must agree with the placed cameras'
    silhouettes (checked again after the adjustment and in the final verification). Returns the members added."""
    added = []
    placed = member.copy()
    placed[np.asarray(placed_now, int)] = True
    for c in placed_now:
        u = units.get(c)
        if u is None:
            continue
        Rn, dn = _move_unit(u, c, R[c], d[c], R, d, placed)
        cams = np.r_[np.nonzero(member)[0], placed_now].astype(int)
        for k in u["members"]:
            if member[k] or k in placed_now or k in added:
                continue
            if _agrees(*_silhouette_agreement(ctx.sil, Rn, dn, [k], cams)[0]):
                R[k], d[k] = Rn[k], dn[k]
                added.append(int(k))
    return added


def _register_by_silhouette(ctx, c, unit, R, d, member):
    """Place observer c, which no feature link reaches, from the silhouettes, together with its unit (if any).
    Candidates: the hull search's best fits for c, and the best one's mirror twins (refined on the tangency; a twin that
    refines back near the start is no rival). Each candidate moves the whole unit (if the unit fits nowhere as a whole:
    c alone) and is judged, in this order, by the hull test (_coverage over the same cameras for every candidate: the
    placed observers and the whole unit at the candidate's pose; the worst coverage of any of their silhouettes), by
    how many unit members agree with the placed silhouettes, and, for near ties, by the feature matches that agree
    with it. A twin is not a rigid move of a unit with two or more viewing directions, so units are rarely ambiguous.
    If a distinct candidate (a twin) is as good on all three, the placement is ambiguous: it is not made, except for the
    origin (placed, with a warning). Writes R, d of the placed observers; returns (placed, reason if none)."""
    others = np.nonzero(member)[0]
    res = _hull_register(c, member, R, d, ctx.small, ctx.size, ctx.reach, ctx.clipped, ctx.sil, ctx.K, HULL_DS, ctx.rng,
                         unit=unit)
    if res is None:
        return [], "the placed observers' hull is too small for a silhouette search"
    Xh, cands = res
    if not cands:
        return [], "no pose fits its silhouette"
    alts = [(o[3], o[4]) for o in cands]                    # the refined starts, and the best one's twins
    for Rt in _twins(cands[0][3], Xh):
        if _angle_deg(Rt, cands[0][3]) < TWIN_MIN_DEG:      # a twin this close is no mirror ambiguity, just a nearby pose
            continue
        r = _refine_on_silhouettes(c, Rt, cands[0][4], R, d, others, ctx.sil, ctx.K)
        if r is not None and _angle_deg(r[3], cands[0][3]) >= TWIN_MIN_DEG:     # (it can refine back near the start)
            alts.append((r[3], r[4]))
    def score(unit):
        todo = [int(k) for k in (unit["members"] if unit is not None else [c]) if not member[k]]
        scored = []
        for Rc, dc in alts:
            Rn, dn = _move_unit(unit, c, Rc, dc, R, d, member)
            if len(todo) > 1:                               # the whole unit on the tangency (one turn, one scale)
                Rn, dn = _refine_gauge(todo, Rn, dn, others, ctx.sil)
            agree = _silhouette_agreement(ctx.sil, Rn, dn, todo, others)
            ok = [k for k, a in zip(todo, agree) if _agrees(*a)]
            if c not in ok:
                continue
            # the same cameras for every candidate (the whole unit, agreeing or not): coverage falls with every camera
            # added, so a set that depends on the candidate would favour the one fewer members agree with
            loo, full, harm = _coverage(Rn, dn, np.r_[others, todo], ctx.small, ctx.K, HULL_DS, ctx.reach, ctx.clipped)
            scored.append(dict(R=Rn, d=dn, ok=ok, F=_feature_support(ok, Rn, dn, member, ctx.feats, ctx.matches, ctx.K),
                               cov=float(full.min()), att=Rn[c]))
        return scored

    scored = score(unit)
    if not scored and unit is not None:                     # a unit bent by a wrong link (its states came from an
        scored = score(None)                                # adjustment) may fit nowhere as a whole: c alone
    if not scored:
        m = cands[0]
        return [], f"silhouette fit too loose (tangency median {m[0]:.1f}, p80 {m[1]:.1f} px)"
    def apart(a, b):
        return _angle_deg(a["att"], b["att"]) > 20
    # the hull test first (a wrong pose carves away what the placed observers see): the candidates within TWIN_MARGIN
    # of the best coverage; among those the most unit members agreeing; features break near-ties only (on this
    # symmetric aircraft, doppelganger matches can support a twin, and a twin can make a whole unit agree on the
    # tangency). A candidate outside the window neither takes over nor makes the placement ambiguous.
    top = max(s["cov"] for s in scored)
    window = [s for s in scored if s["cov"] >= top - TWIN_MARGIN]
    best = max(window, key=lambda s: (len(s["ok"]), s["cov"]))
    near = [s for s in window if apart(s, best) and len(s["ok"]) == len(best["ok"])]
    for s in near:
        if s["F"] >= best["F"] + TWIN_FEAT:
            best = s
    rivals = [s for s in scored if apart(s, best)]
    tie = [s for s in window if apart(s, best) and len(s["ok"]) >= len(best["ok"]) and abs(best["F"] - s["F"]) < TWIN_FEAT]
    DEBUG.setdefault("silreg", []).append(dict(c=c, unit=[int(k) for k in (unit["members"] if unit is not None else [c])],
                                               best=(len(best["ok"]), best["F"], best["cov"]),
                                               rivals=[(len(s["ok"]), s["F"], s["cov"]) for s in rivals], tie=len(tie)))
    if tie and 0 not in best["ok"]:
        return [], "mirror-ambiguous silhouette placement (a twin pose fits as well)"
    if tie:
        ctx.warn[0] = "placed from its silhouette; a mirror-twin pose fits as well"
    for k in best["ok"]:
        R[k], d[k] = best["R"][k], best["d"][k]
    return best["ok"], None


# ----------------------------------------------------------------------------- 9. the estimator

def _guided_pairs(R, member, matches, k=GUIDED_K, max_deg=GUIDED_DEG):
    """For every placed observer, its k placed neighbours with the closest viewing directions (within max_deg) that
    have not been matched with it yet: the pairs the current estimate says overlap."""
    axis = R[:, 2, :]
    cos = axis @ axis.T
    cos[~member] = cos[:, ~member] = -2.0
    np.fill_diagonal(cos, -2.0)
    pairs = set()
    for i in np.nonzero(member)[0]:
        for j in np.argsort(-cos[i])[:3 * k]:
            if cos[i, j] < np.cos(np.radians(max_deg)):
                break
            key = (min(i, int(j)), max(i, int(j)))
            if key not in matches:
                pairs.add(key)
                if sum(1 for p in pairs if i in p) >= k:
                    break
    return sorted(pairs)


class _State:
    """The reconstruction: states R (n,3,3), d (n,), who is placed (member), and the tracks with their points."""

    def __init__(self, R, d, member):
        self.R, self.d, self.member = R, d, member
        self.trk = self.cam = self.kidx = self.kp = self.X = self.ok = None

    def retrack(self, ctx):
        self.trk, self.cam, self.kidx, self.kp = _tracks(ctx.feats, _track_matches(ctx.feats, ctx.matches, ctx.edges, self.R,
                                                                                    self.d, self.member, ctx.K), ctx.n)

    def adjust(self, ctx, sigmas):
        self.R, self.d, self.X, self.ok, _ = _adjust(self.R, self.d, self.cam, self.trk, self.kp, ctx.K, self.member,
                                                     ctx.anchor, ctx.sil, ctx.rng, sigmas=sigmas)


def _grow(ctx, st, units, rounds=None):
    """Rounds that place the observers left out: resection from feature matches (their units come along); if that
    places nobody, the silhouettes (the origin's unit first, then the observers not tried yet; one attempt per unit,
    again once the hull has grown, at most HULL_TRIES per observer except for the origin's unit). Each round re-tracks
    and adjusts. Large swarms also match the pairs the estimate says overlap, and the unplaced with similar placed
    observers."""
    rounds = GROW_ROUNDS if rounds is None else rounds
    tried, tries = {}, {}                                   # silhouette registration: member count at the last try,
    for _ in range(rounds):                                 # and how many tries
        if ctx.n > ALL_PAIRS_MAX:
            ctx.clock.stage("mapping")
            more = _guided_pairs(st.R, st.member, ctx.matches) + _reach_pairs(ctx.score, st.member, ctx.matches)
            ctx.matches.update(_match_batches(sorted(set(more)), ctx.feats, ctx.best_v, ctx.skipped))
            ctx.clock.stage("matching")
        added = _register(ctx.feats, ctx.matches, st.member, st.R, st.d, st.X, st.cam, st.trk, st.kidx, st.ok, ctx.K, ctx.sil,
                          ctx.clipped, ctx.rng)
        added += _bring_units(ctx, units, added, st.R, st.d, st.member)
        if not added and st.member.sum() >= 4:              # nothing links by features: the silhouettes

            def mem(k):
                return units[k]["members"] if k in units else [k]
            todo = [int(c) for c in np.nonzero(~st.member)[0]   # again once the hull grew
                    if st.member.sum() > tried.get(c, -9) and (tries.get(c, 0) < HULL_TRIES or 0 in mem(c))]
            done, attempts = set(), 0

            def order(k):                                   # the origin's unit, the untried, large units, large silhouettes
                return (0 not in mem(k), k in tried, -len(mem(k)), k != 0, -ctx.size[k])
            for c in sorted(todo, key=order):
                u = units.get(c)
                if c in done or st.member[c]:
                    continue
                attempts += 1
                if attempts > HULL_PER_ROUND:
                    break
                group = [int(k) for k in u["members"]] if u is not None else [c]
                done.update(group)
                for k in group:
                    tried[k] = st.member.sum()
                    tries[k] = tries.get(k, 0) + 1
                placed, why = _register_by_silhouette(ctx, c, u, st.R, st.d, st.member)
                DEBUG.setdefault("hullreg", []).append((c, placed, why))
                if why is not None:
                    ctx.notes[c] = why
                for k in placed:                                # provisional: checked again after the adjustment
                    st.member[k] = True
                    added.append(k)
        DEBUG.setdefault("grow", []).append(added)
        if not added:
            break
        st.member[added] = True
        st.retrack(ctx)
        st.adjust(ctx, (8.0, 2.0))


def _verify(ctx, st):
    """Every placed observer must agree with the others. First the hull test (_coverage): no observer may carve away
    much of what the others see (HARM_MAX: a mirror twin, which doppelganger matches of this symmetric aircraft can hold
    and which passes the tangency test), and one that fewer than STRONG_OBS feature observations hold must also lie
    inside the hull the others carve (relative to the swarm's median: the coarse carving loses ~10 % even at the truth;
    well-held views can fall below it through the carving alone). Its offenders go first: a twin bends everyone's
    tangency. Then the epipolar tangency (median and 80th percentile; for a silhouette cut off by its frame, which
    disagrees in places, the median is enough when many feature observations hold the camera; one that few feature
    observations support must agree closely). The worst offender is dropped and the check repeated; the gauge anchor too
    (the best-supported camera left takes over). Returns the dropped observers (reasons in ctx.notes)."""
    dropped, hull_due = [], True                            # the hull test again only after a drop it caused
    for _ in range(ctx.n):
        support = np.bincount(st.cam[st.ok], minlength=ctx.n)
        cams = np.nonzero(st.member)[0]
        if len(cams) < 2:
            break
        bad = []
        if hull_due and len(cams) >= HULL_TEST_MIN:
            loo, full, harm = _coverage(st.R, st.d, cams, ctx.small, ctx.K, HULL_DS, ctx.reach, ctx.clipped)
            DEBUG["coverage"] = dict(cams=cams, loo=loo, full=full, harm=harm)
            ref = float(np.median(loo))
            for q, c in enumerate(cams):
                weak = support[c] < STRONG_OBS
                if (weak and loo[q] < ref - LOO_DROP) or harm[q] > HARM_MAX:
                    bad.append((max((ref - loo[q]) / LOO_DROP if weak else 0.0, harm[q] / HARM_MAX), c,
                                f"fails the hull test (its silhouette {100 * loo[q]:.0f} % inside the others' hull, "
                                f"median {100 * ref:.0f} %; it uncovers {100 * harm[q]:.0f} % of another silhouette)"))
        if not bad:
            agree = _silhouette_agreement(ctx.sil, st.R, st.d, cams, cams)
            for c, (med, p80, cnt) in zip(cams, agree):
                if not (med < SIL_MEDIAN_PX and (p80 < SIL_P80_PX or (ctx.clipped[c] and support[c] >= STRONG_OBS))):
                    bad.append((p80, c, f"disagrees with the others' silhouettes (tangency median {med:.1f}, p80 {p80:.1f} px)"))
                elif support[c] < MIN_CAM_OBS and not (med < HULL_MEDIAN_PX and p80 < HULL_P80_PX and cnt >= 3):
                    bad.append((p80, c, f"{support[c]} feature observations and a loose silhouette fit "
                                        f"(tangency median {med:.1f}, p80 {p80:.1f} px)"))
        if not bad:
            break
        _, c, why = max(bad)
        hull_due = why.startswith("fails the hull test")
        st.member[c] = False
        dropped.append(int(c))
        ctx.notes[int(c)] = why
        if c == ctx.anchor:                                 # a new gauge: the best-supported camera left
            left = np.nonzero(st.member)[0]
            ctx.anchor = int(left[np.argmax(support[left])]) if len(left) else 0
    return dropped


def _units_of(ctx, st, cams):
    """The dropped observers as units: groups linked by verified pairs that agree with their current states (a group
    attached through a doppelganger link is wrong as a whole but right inside)."""
    cams = set(int(c) for c in cams)
    parent = {c: c for c in cams}

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c
    for e in ctx.edges:
        i, j = e["i"], e["j"]
        if i in cams and j in cams:
            dR = st.R[j] @ st.R[i].T @ e["R"].T
            if np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 5.0:
                parent[root(i)] = root(j)
    groups = {}
    for c in cams:
        groups.setdefault(root(c), []).append(c)
    units = {}
    for g in groups.values():
        if len(g) >= 2:
            u = dict(members=np.array(sorted(g)), R=st.R.copy(), d=st.d.copy())
            for c in g:
                units[c] = u
    return units


def _result(ctx, st, clock, status=None):
    """Relative to the origin (R_rel = R_i R_0^T, d_rel = d_i / d_0), with the status and a reason for every observer
    not returned. If the origin is not placed, only the origin is returned (the contract is relative to it); the
    placed group is then given, in the gauge of its best-linked member, as "unanchored"."""
    n = ctx.n
    member = st.member if st is not None else np.eye(n, dtype=bool)[0]
    R = st.R if st is not None else np.tile(np.eye(3), (n, 1, 1))
    d = st.d if st is not None else np.ones(n)
    out = dict(timings=clock.timings)
    if member[0]:
        reg = np.nonzero(member)[0]
        out.update(R_rel=R[reg] @ R[0].T, d_rel=d[reg] / d[0], registered=reg.tolist())
        status = status or ("ok" if member.all() else "partial")
    else:
        reg = np.array([0])
        out.update(R_rel=np.eye(3)[None], d_rel=np.ones(1), registered=[0])
        status = status or "origin_unplaced"
        if member.any():
            g = np.nonzero(member)[0]
            out["unanchored"] = dict(registered=g.tolist(), R=R[g] @ R[ctx.anchor].T, d=d[g] / d[ctx.anchor])
    out["status"] = status
    default = "no verified feature link and no silhouette fit"
    feats = getattr(ctx, "feats", None)
    if feats is not None:                                   # pairs with fewer than 64 keypoints are never matched
        for c in range(n):
            k = len(feats[c]["vers"][0]["kp"])
            if k < 64 and c not in ctx.notes:
                ctx.notes[c] = f"only {k} keypoints on the silhouette (64 needed to match); no silhouette fit"
    out["dropped"] = {int(c): ctx.notes.get(int(c), default) for c in range(n) if c not in set(reg.tolist())}
    if not member[0]:                                       # the origin is returned (the reference), but unplaced
        out["origin_reason"] = ctx.notes.get(0, default)
        for c in np.nonzero(member)[0]:
            out["dropped"][int(c)] = "placed, but in a group the origin could not be attached to"
    out["warnings"] = dict(ctx.warn)
    if _VIZ is not None and st is not None and member[0]:
        out["viz"] = _viz_result(ctx, st)
    out["seconds"] = time.perf_counter() - clock.t0
    return out


def _viz_result(ctx, st):
    """For a video of the method (viz=True): a few illustrative matched pairs (inlier pixels, original images), the
    view graph, the registered cameras' states after each stage (relative to the final origin, the earlier stages
    turned so that the anchor agrees with its final state), coloured points in the origin's gauge (hub at 0, R_0 = I,
    d_0 = 1) and the bundle adjustments' RMS (px) per accepted step."""
    reg = np.nonzero(st.member)[0]
    R0, d0 = st.R[0], st.d[0]
    a = ctx.anchor
    names, kinds, kp0, kp1 = [], [], [], []
    edges = ctx.edges
    ang = np.array([np.degrees(np.linalg.norm(logmap(e["R"]))) for e in edges]) if edges else np.zeros(0)
    nin = np.array([len(e["m"]) for e in edges]) if edges else np.zeros(0)
    picks = []
    if len(edges):
        strong = np.nonzero(nin >= 50)[0] if (nin >= 50).any() else np.arange(len(edges))
        picks.append(("close", int(strong[np.argmin(ang[strong])])))
        picks.append(("wide baseline", int(strong[np.argmax(ang[strong])])))
        if ctx.best_v is not None:
            rolled = [k for k, e in enumerate(edges) if ctx.best_v[e["i"], e["j"]] != 0]
            if rolled:
                k = max(rolled, key=lambda k: nin[k])
                picks.append((f"rescued by the roll fix ({'+' if ctx.best_v[edges[k]['i'], edges[k]['j']] == 1 else '-'}"
                              f"{ROLL_DEG:.0f} deg version)", k))
    for kind, k in picks:
        e = edges[k]
        names.append((ctx.names[e["i"]], ctx.names[e["j"]]))
        kinds.append(kind)
        kp0.append(ctx.feats[e["i"]]["kp_np"][e["m"][:, 0]])
        kp1.append(ctx.feats[e["j"]]["kp_np"][e["m"][:, 1]])
    stages, SR, Sd = [], [], []
    for name, Rs, ds in ctx.viz_stages + [("final bundle adjustment", st.R, st.d)]:
        ok = np.isfinite(ds) & (ds > 0)
        if not ok[a]:
            continue
        Ra = Rs @ Rs[a].T @ st.R[a]                         # the stage's gauge turned onto the final one at the anchor
        Rr = np.where(ok[reg, None, None], Ra[reg] @ R0.T, np.nan)
        dr = np.where(ok[reg], ds[reg] / ds[a] * st.d[a] / d0, np.nan)
        stages.append(name)
        SR.append(Rr)
        Sd.append(dr)
    use = st.ok & st.member[st.cam]
    pts = np.unique(st.trk[use])
    Xo = (st.X[pts] @ R0.T) / d0                            # the origin's gauge
    first = {}
    for o in np.nonzero(use)[0]:
        first.setdefault(int(st.trk[o]), o)
    rgb = np.zeros((len(pts), 3))
    by_cam = {}
    for q, t in enumerate(pts):
        o = first[int(t)]
        by_cam.setdefault(int(st.cam[o]), []).append((q, st.kp[o]))
    for c, lst in by_cam.items():
        img = cv2.imread(str(ctx.paths[c]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for q, p in lst:
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                rgb[q] = img[y, x, ::-1] / 255.0
    return dict(pair_names=names, pair_kinds=kinds, pair_kp0=kp0, pair_kp1=kp1,
                edges=np.array([(e["i"], e["j"]) for e in edges], int).reshape(-1, 2),
                edge_inliers=nin.astype(int), stage_names=stages, stage_R_rel=np.array(SR), stage_d_rel=np.array(Sd),
                registered=reg.tolist(), points=Xo, points_rgb=rgb, history=list(_VIZ["history"]))


def estimate(image_paths, mask_paths, K, progress=None, viz=False):
    """image_paths[0] is the origin. Returns R_rel (M,3,3), d_rel (M,), registered (indices, origin first),
    timings {stage: s}, seconds, status ("ok": all placed; "partial": the origin and possibly others; "origin_unplaced":
    the origin could not be placed, only it is returned; "origin_unseen": its silhouette is (nearly) empty;
    "no_links": no pair of observers could be linked), dropped {index: reason} for every observer not returned,
    warnings {index: text}; with viz=True also "viz" (see _viz_result; viz=False records nothing)."""
    global _VIZ
    clock = _Clock(progress)                                # the warm-up, if the caller skipped it, is in "load"
    if not _NET:
        warm_up()
    if _DEVICE.type == "cuda":                              # a long-running caller: start from an unfragmented cache
        torch.cuda.empty_cache()
    DEBUG.clear()
    _VIZ = dict(history=[]) if viz else None
    try:
        with ThreadPoolExecutor(THREADS) as pool:
            data = list(pool.map(_load, zip(image_paths, mask_paths)))
        clock.stage("load")
        n_all = len(data)
        seen = np.nonzero([m.sum() >= MIN_MASK_PX for _, m, _ in data])[0]     # observers that see the target
        ctx = SimpleNamespace(n=n_all, notes={}, warn={}, anchor=0)
        tiny = {int(i): f"silhouette smaller than {MIN_MASK_PX} px (does not see the target)"
                for i in range(n_all) if i not in set(seen.tolist())}
        if len(seen) == 0 or seen[0] != 0:
            ctx.notes.update({i: "not attempted: the origin's silhouette is (nearly) empty" for i in range(n_all)})
            ctx.notes.update(tiny)
            return _result(ctx, None, clock, status="origin_unseen")
        out = _estimate([data[i] for i in seen], np.asarray(K, float), clock,
                        [image_paths[i] for i in seen], viz)
        out["registered"] = [int(seen[i]) for i in out["registered"]]
        if out["status"] == "ok" and len(out["registered"]) < n_all:     # observers excluded for their masks
            out["status"] = "partial"
        out["dropped"] = {int(seen[i]): r for i, r in out["dropped"].items()}
        out["dropped"].update(tiny)
        out["warnings"] = {int(seen[i]): r for i, r in out["warnings"].items()}
        if "unanchored" in out:
            out["unanchored"]["registered"] = [int(seen[i]) for i in out["unanchored"]["registered"]]
        if "viz" in out:
            v = out["viz"]
            v["edges"] = seen[v["edges"]] if len(v["edges"]) else v["edges"]
            v["registered"] = [int(seen[i]) for i in v["registered"]]
        return out
    finally:
        _VIZ = None


def _estimate(data, K, clock, paths, viz):
    """The pipeline on the loaded observers (the origin first)."""
    n = len(data)
    rng = np.random.default_rng(0)
    ctx = SimpleNamespace(n=n, K=K, rng=rng, clock=clock, notes={}, warn={}, skipped={}, anchor=0, best_v=None,
                          score=None, edges=[], paths=paths, names=[os.path.basename(str(p)) for p in paths],
                          viz_stages=[])
    with ThreadPoolExecutor(THREADS) as pool:
        outlines = list(pool.map(lambda gm: _outline(gm[1], K), data))
        ctx.small = torch.from_numpy(np.stack([x[2] for x in data])).to(_DEVICE)
        ctx.size = np.array([np.sqrt(x[1].sum() / np.pi) for x in data]) / K[0, 0]     # silhouette radius (normalised)
        ctx.reach = np.array([np.linalg.norm(o[0][0][:, :2], axis=1).max() for o in outlines])  # farthest outline point
        ctx.clipped = np.array([(o[2][:, 1] <= o[2][:, 2]).any() for o in outlines])             # touches its frame
        angles = _roll_angles(n)
        preps = list(pool.map(lambda job: _prepare(data[job[0]][0], data[job[0]][1], job[1]),
                              [(i, a) for i in range(n) for a in angles]))
        ctx.feats = feats = _group_versions(_features(preps), n, len(angles))
        del preps
        clock.stage("features")

        if n > ALL_PAIRS_MAX or len(angles) > 1:
            ctx.score, ctx.best_v = _similarity(feats)
        if n <= ALL_PAIRS_MAX:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:                                               # a shortlist, and the origin with everyone
            pairs = sorted(set(_shortlist(ctx.score)) | {(0, j) for j in range(1, n)})
        ctx.matches = matches = _match_batches(pairs, feats, ctx.best_v, ctx.skipped)
        if len(angles) > 1:                                 # the descriptor vote can pick the wrong version: for the
            _rematch_weak(ctx, pairs, len(angles))          # origin and weakly matched observers, try the others too
        clock.stage("matching")

        ctx.sil = sil = _Silhouettes(outlines, K[0, 0])
        ctx.edges = edges = _two_view(feats, matches, K, pool, sil)
        groups = _view_groups(n, edges, rng) if edges else []
        clock.stage("geometry")
        DEBUG.update(edges=edges, groups=[(np.nonzero(g["member"])[0], g["anchor"]) for g in groups], n_pairs=len(pairs),
                     skipped=dict(ctx.skipped))
        if not groups:
            for c in range(1, n):
                ctx.notes[c] = ctx.skipped.get((0, c), "no verified feature link")
            return _result(ctx, None, clock, status="no_links" if n > 1 else "ok")
        g0 = groups[0]
        ctx.anchor = g0["anchor"]
        units = {}                                          # observer -> its unit (the other groups)
        for g in groups[1:]:
            u = dict(members=np.nonzero(g["member"])[0], R=g["R"], d=g["d"])
            for c in u["members"]:
                units[int(c)] = u
        if viz:
            dn = np.where(g0["member"], g0["d"], np.nan)
            ctx.viz_stages = [("chained (spanning tree)", g0["R_chain"], dn), ("rotation averaging", g0["R"], dn)]

    st = _State(g0["R"].copy(), g0["d"].copy(), g0["member"].copy())
    trk, cam, kidx, kp = _tracks(feats, [(e["i"], e["j"], e["m"]) for e in g0["kept"]], n)
    st.trk, st.cam, st.kidx, st.kp = trk, cam, kidx, kp
    st.adjust(ctx, BA_SIGMAS)
    _grow(ctx, st, units)
    clock.stage("mapping")
    st.retrack(ctx)
    st.adjust(ctx, FINAL_SIGMAS)
    dropped = _verify(ctx, st)
    DEBUG["dropped"] = list(dropped)
    if dropped and REGROW:                                  # grow them back once, as units, then check again
        clock.stage("bundle")
        st.retrack(ctx)
        st.adjust(ctx, FINAL_SIGMAS)
        units2 = dict(units)
        units2.update(_units_of(ctx, st, dropped))
        _grow(ctx, st, units2)
        clock.stage("mapping")
        st.retrack(ctx)
        st.adjust(ctx, FINAL_SIGMAS)
        DEBUG["dropped2"] = _verify(ctx, st)
    if dropped:
        st.retrack(ctx)
        st.adjust(ctx, FINAL_SIGMAS)
    DEBUG.update(n_obs=int(st.ok.sum()), final=dict(R=st.R.copy(), d=st.d.copy(), member=st.member.copy(), anchor=ctx.anchor))
    clock.stage("bundle")
    return _result(ctx, st, clock)
