"""Visual hull ("shape from silhouettes") by voxel carving.

The idea in one sentence: a point can only belong to the object if every view that can see it
sees it on the object's silhouette (the white part of the mask). So we fill a box with voxels and
delete each voxel that some view sees on the background. What survives is the visual hull - the
largest shape that agrees with all the silhouettes.

Two passes:
  A. a coarse grid over a generous cube around the point the cameras look at -> finds the object
  B. a fine grid over the bounding box of what survived pass A -> the result

Some arithmetic (to_camera, look_at_point, initial_box and the dot products) is written out term by
term in the same order as the C++ version, so both keep exactly the same voxels. Matrix products or
np.linalg.solve would round differently.
"""
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from colmap import Point3D, focal, project, read_model, write_points3d_bin
from poses import camera_center, qvec_to_rotmat

SPHERE_RADIUS = math.sqrt(3) / 2   # radius of a voxel's bounding sphere (half its diagonal), in voxel sizes


@dataclass
class View:
    name: str
    camera: object          # colmap.Camera
    R: np.ndarray           # 3x3 world-to-camera rotation
    t: np.ndarray           # world-to-camera translation
    mask: np.ndarray        # H x W bool, True on the object
    image_path: Path = None  # colour image, only read when colouring the result

    @property
    def center(self):
        return camera_center(self.R, self.t)


@dataclass
class Hull:
    points: np.ndarray      # N x 3 voxel centres (surface voxels, or all of them with solid=True)
    normals: np.ndarray     # N x 3 outward unit normals; 0 where the occupancy gradient is 0
    colors: np.ndarray      # N x 3 uint8
    occupied: np.ndarray    # the carved pass-B grid (bool, dims)
    lo: np.ndarray          # corner of the pass-B grid
    voxel_size: float
    coverage: np.ndarray    # per view: fraction of the mask covered by the hull's projection


# ----------------------------------------------------------------------------- loading

def find_mask(masks_dir, image_name):
    """masks/<image name>, or the same stem with a .png/.jpg/.jpeg extension."""
    stem = Path(image_name).stem
    for name in (image_name, stem + ".png", stem + ".jpg", stem + ".jpeg"):
        if (Path(masks_dir) / name).exists():
            return Path(masks_dir) / name
    return None


def load_mask(path):
    """Silhouette as a bool array: grey value >= 128 (decoded the same way as the C++ version)."""
    with PILImage.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", image.size)                 # decode the brightness plane directly
        if image.mode.startswith("I"):                   # 16-bit grey: keep the high byte
            return (np.asarray(image).astype(np.int64) >> 8) >= 128
        return np.asarray(image.convert("L")) >= 128


def load_views(images_dir, masks_dir, sparse_dir):
    """One View per image of the sparse model (in image-id order) that has a mask."""
    cameras, images, _ = read_model(sparse_dir)
    views, skipped = [], []
    for image in sorted(images.values(), key=lambda im: im.id):
        mask_path = find_mask(masks_dir, image.name)
        if mask_path is None:
            skipped.append(image.name)
            continue
        camera = cameras[image.camera_id]
        focal(camera)  # fails early for camera models we cannot project
        mask = load_mask(mask_path)
        sx, sy = mask.shape[1] / camera.width, mask.shape[0] / camera.height
        if abs(sx - sy) > 0.01 * max(sx, sy):
            raise ValueError(f"{mask_path}: mask is {mask.shape[1]}x{mask.shape[0]} but the camera is "
                             f"{camera.width}x{camera.height} (different aspect ratio)")
        views.append(View(image.name, camera, qvec_to_rotmat(image.qvec), np.array(image.tvec, dtype=float),
                          mask, Path(images_dir) / image.name))
    if skipped:
        print(f"warning: skipped {len(skipped)} image(s) without a mask, e.g. {skipped[0]}")
    if not views:
        raise ValueError("no image has a matching mask")
    return views


# ----------------------------------------------------------------------------- geometry helpers

def to_camera(view, X):
    """Camera coordinates of world points X (N x 3): R @ X + t, written out term by term so it
    rounds exactly like the C++ version (X @ R.T + t does not)."""
    R, t = view.R, view.t
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0],
            R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1],
            R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2])


def footprints(view, X, voxel_size, width, height):
    """Pixel rectangles [x0, x1] x [y0, y1] that contain each voxel's projection.

    A voxel is replaced by its bounding sphere (radius r): the rectangle is (u, v) +- f*r/z. For lenses
    up to about 70 degrees horizontal field of view this contains the voxel's whole projection, so
    thin parts (a wing seen edge-on) are not carved away by accident.
    width/height may differ from the camera's (e.g. half-resolution masks).
    Returns (touches, inside, x0, x1, y0, y1): touches = in front of the camera and overlapping the
    image; inside = in front of the camera and entirely inside the image (clipped otherwise).
    """
    xc, yc, zc = to_camera(view, X)
    front = zc > 1e-12
    zc = np.where(front, zc, 1.0)                  # any positive depth; these voxels are ignored
    u, v = project(view.camera, xc, yc, zc)
    sx, sy = width / view.camera.width, height / view.camera.height
    fx, fy = focal(view.camera)
    u, v = u * sx, v * sy
    rx = fx * (SPHERE_RADIUS * voxel_size) / zc * sx   # projected sphere radius in pixels
    ry = fy * (SPHERE_RADIUS * voxel_size) / zc * sy
    touches = front & (u + rx >= 0) & (u - rx < width) & (v + ry >= 0) & (v - ry < height)
    inside = front & (u - rx >= 0) & (u + rx < width) & (v - ry >= 0) & (v + ry < height)
    x0 = np.clip(np.floor(u - rx), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.floor(u + rx), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(v - ry), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.floor(v + ry), 0, height - 1).astype(np.int64)
    return touches, inside, x0, x1, y0, y1


def summed_area_table(mask):
    """S[y, x] = number of object pixels in mask[:y, :x], so the object pixels in rows y0..y1 and
    columns x0..x1 are S[y1+1, x1+1] - S[y0, x1+1] - S[y1+1, x0] + S[y0, x0]: four lookups."""
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    S[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)
    return S


def make_grid(lo, hi, n):
    """Centres of cubic voxels filling the box [lo, hi], n voxels along its longest side.
    Order: x changes slowest, z fastest (like a numpy array indexed [i, j, k])."""
    voxel_size = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / n
    dims = [max(1, math.ceil((hi[a] - lo[a]) / voxel_size - 1e-9)) for a in range(3)]
    axes = [lo[a] + (np.arange(dims[a]) + 0.5) * voxel_size for a in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return centers, dims, voxel_size


def look_at_point(views):
    """The point closest (least squares) to every camera's viewing axis - where they all look."""
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for view in views:
        d, C = view.R[2], view.center              # viewing direction, camera centre
        dc = d[0] * C[0] + d[1] * C[1] + d[2] * C[2]
        for j in range(3):
            b[j] += C[j] - d[j] * dc
            for k in range(3):
                A[j][k] += (1.0 if j == k else 0.0) - d[j] * d[k]
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = A
    b0, b1, b2 = b
    det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    n = len(views)
    if det / (n * n * n) < 1e-6:                    # viewing axes (nearly) parallel
        print("warning: the cameras' viewing axes are nearly parallel; using the mean camera position")
        centers = [view.center for view in views]
        return np.array([sum(c[j] for c in centers) / n for j in range(3)])
    return np.array([  # Cramer's rule for the 3x3 system A p = b
        (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / det,
        (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / det,
        (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / det])


def initial_box(views, cube_size=0.0, shift=(0.0, 0.0, 0.0)):
    """Pass-A cube around the look-at point. Its automatic half-size is the radius each camera can
    see around that point (distance x tan of the half-diagonal field of view), smallest over views."""
    center = look_at_point(views)
    center = np.array([center[j] + shift[j] for j in range(3)])
    if cube_size > 0:
        half = cube_size
    else:
        candidates = []
        for view in views:
            d = view.center - center
            fx, fy = focal(view.camera)
            hx, hy = view.camera.width / (2.0 * fx), view.camera.height / (2.0 * fy)
            candidates.append(math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * math.sqrt(hx * hx + hy * hy))
        half = min(candidates)
    return center - half, center + half


# ----------------------------------------------------------------------------- carving

def carve(views, centers, voxel_size, min_votes, max_misses=0):
    """Indices of the voxels that survive carving. Each view says one of three things about a voxel:
      hit    - its footprint touches the silhouette
      miss   - its footprint lies entirely inside the image and on the background: the view SEES
               empty space there
      unseen - anything else (behind the camera, off the image, or partly off the image without
               touching the silhouette): this view knows nothing
    A voxel survives with at most max_misses misses and at least min_votes hits, so it may be unseen
    by at most V - min_votes views. Treating "unseen" as a miss (as the original did) deletes the parts
    of the object that leave the frame.
    """
    alive = np.arange(len(centers))                # voxels still standing
    hits = np.zeros(len(centers), np.int64)
    misses = np.zeros(len(centers), np.int64)
    for views_done, view in enumerate(views, start=1):
        height, width = view.mask.shape
        touches, inside, x0, x1, y0, y1 = footprints(view, centers[alive], voxel_size, width, height)
        S = summed_area_table(view.mask)
        object_pixels = S[y1 + 1, x1 + 1] - S[y0, x1 + 1] - S[y1 + 1, x0] + S[y0, x0]
        hits += touches & (object_pixels > 0)
        misses += inside & (object_pixels == 0)
        keep = (misses <= max_misses) & (hits + len(views) - views_done >= min_votes)  # can it still make it?
        alive, hits, misses = alive[keep], hits[keep], misses[keep]
    return alive


def surface_voxels(occupied):
    """Occupied voxels with at least one empty (or out-of-grid) face neighbour."""
    p = np.pad(occupied, 1)
    inside = (p[2:, 1:-1, 1:-1] & p[:-2, 1:-1, 1:-1] & p[1:-1, 2:, 1:-1] &
              p[1:-1, :-2, 1:-1] & p[1:-1, 1:-1, 2:] & p[1:-1, 1:-1, :-2])
    return occupied & ~inside


def normals_at(occupied, ijk):
    """Outward normals = minus the occupancy gradient (central differences) at voxels ijk.
    Zero where the gradient is zero (inside, or where a part is one voxel thin)."""
    p = np.pad(occupied, 1).astype(np.int8)
    i, j, k = ijk[:, 0] + 1, ijk[:, 1] + 1, ijk[:, 2] + 1
    n = np.stack([p[i - 1, j, k] - p[i + 1, j, k],    # minus the gradient, negated while still
                  p[i, j - 1, k] - p[i, j + 1, k],    # integers so that no -0.0 appears
                  p[i, j, k - 1] - p[i, j, k + 1]], axis=1).astype(float)
    length = np.sqrt(n[:, 0] * n[:, 0] + n[:, 1] * n[:, 1] + n[:, 2] * n[:, 2])
    normals = np.zeros_like(n)
    has = length > 0
    normals[has] = n[has] / length[has, None]
    return normals


def colorize(views, points, normals):
    """Colour of each point = average over the views that face it and see it on the silhouette,
    weighted by how directly they face it (cosine). Falls back to all views that see it."""
    weighted, weights = np.zeros((len(points), 3)), np.zeros(len(points))
    plain, counts = np.zeros((len(points), 3)), np.zeros(len(points))
    has_normal = (normals[:, 0] != 0) | (normals[:, 1] != 0) | (normals[:, 2] != 0)
    for view in views:
        if view.image_path is None or not Path(view.image_path).exists():
            continue
        with PILImage.open(view.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=float)
        xc, yc, zc = to_camera(view, points)
        ok = zc > 1e-12
        u, v = project(view.camera, xc, yc, np.where(ok, zc, 1.0))
        # the point's centre must fall on this view's silhouette ...
        mask_h, mask_w = view.mask.shape
        mx = np.floor(u * (mask_w / view.camera.width))
        my = np.floor(v * (mask_h / view.camera.height))
        ok &= (mx >= 0) & (mx < mask_w) & (my >= 0) & (my < mask_h)
        ok[ok] = view.mask[my[ok].astype(np.int64), mx[ok].astype(np.int64)]
        # ... and inside the colour image
        image_h, image_w = rgb.shape[:2]
        px = np.floor(u * (image_w / view.camera.width))
        py = np.floor(v * (image_h / view.camera.height))
        ok &= (px >= 0) & (px < image_w) & (py >= 0) & (py < image_h)
        idx = np.flatnonzero(ok)
        color = rgb[py[idx].astype(np.int64), px[idx].astype(np.int64)]
        # cosine between the outward normal and the direction to the camera
        w = view.center - points[idx]
        w = w / np.sqrt(w[:, 0] * w[:, 0] + w[:, 1] * w[:, 1] + w[:, 2] * w[:, 2])[:, None]
        n = normals[idx]
        cosine = n[:, 0] * w[:, 0] + n[:, 1] * w[:, 1] + n[:, 2] * w[:, 2]
        weight = np.where(has_normal[idx], cosine, 1.0)
        facing = weight > 0
        weighted[idx[facing]] += weight[facing, None] * color[facing]
        weights[idx[facing]] += weight[facing]
        plain[idx] += color
        counts[idx] += 1
    colors = np.full((len(points), 3), 128.0)
    some = counts > 0
    colors[some] = plain[some] / counts[some, None]
    facing = weights > 0
    colors[facing] = weighted[facing] / weights[facing, None]
    return np.clip(np.floor(colors + 0.5), 0, 255).astype(np.uint8)


def silhouette_coverage(views, points, voxel_size):
    """Per view: the fraction of the mask's object pixels covered by the hull's projection.

    If all poses and masks agree this is ~1.0 everywhere. A view whose pose or mask is wrong carves
    away parts that the other views need, so their coverage drops - this is the sanity check that
    would have caught the pose and mask bugs of the original pipeline.
    """
    coverage = []
    for view in views:
        height, width = view.mask.shape
        ok, _, x0, x1, y0, y1 = footprints(view, points, voxel_size, width, height)
        x0, x1, y0, y1 = x0[ok], x1[ok], y0[ok], y1[ok]
        # paint all footprint rectangles at once: a 2-D difference array, then running sums
        paint = np.zeros((height + 1, width + 1), np.int64)
        np.add.at(paint, (y0, x0), 1)
        np.add.at(paint, (y0, x1 + 1), -1)
        np.add.at(paint, (y1 + 1, x0), -1)
        np.add.at(paint, (y1 + 1, x1 + 1), 1)
        covered = paint.cumsum(axis=0).cumsum(axis=1)[:height, :width] > 0
        n_object = int(view.mask.sum())
        coverage.append(1.0 if n_object == 0 else int((covered & view.mask).sum()) / n_object)
    return np.array(coverage)


# ----------------------------------------------------------------------------- the whole thing

def visual_hull(views, voxel_num=64, passB_res=200, bbox_scale=1.0, cube_size=0.0,
                shift=(0.0, 0.0, 0.0), min_votes=0, max_misses=0, solid=False, color=True):
    """Two-pass visual hull. min_votes <= 0 means 90% of the views (all of them below 10 views)."""
    if voxel_num < 1 or passB_res < 1 or not bbox_scale > 0 or cube_size < 0 or max_misses < 0:
        raise ValueError("voxel_num and passB_res must be >= 1, bbox_scale > 0, cube_size and max_misses >= 0")
    V = len(views)
    min_votes = V - V // 10 if min_votes <= 0 else min(min_votes, V)
    print(f"{V} views: keep voxels on the silhouette in >= {min_votes} views and on the background in <= {max_misses}")

    # Pass A: coarse grid over the automatic (or --cube-size) box
    start = time.time()
    lo, hi = initial_box(views, cube_size, shift)
    centers, dims, size_a = make_grid(lo, hi, voxel_num)
    kept = carve(views, centers, size_a, min_votes, max_misses)
    print(f"pass A: {dims[0]}x{dims[1]}x{dims[2]} voxels of {size_a:.4g} in {_box(lo, hi)} "
          f"-> {len(kept)} kept ({time.time() - start:.1f} s)")
    if len(kept) == 0:
        raise RuntimeError("pass A kept no voxels - masks and poses disagree "
                           "(or the object is outside the box; try --cube-size)")
    ijk = np.stack(np.unravel_index(kept, dims), axis=1)
    if (ijk.min() == 0) or np.any(ijk.max(axis=0) == np.array(dims) - 1):
        print("warning: the hull touches the pass-A box; the object may be clipped - increase --cube-size")

    # Pass B: fine grid over the box around pass A's voxel centres, padded by one coarse voxel
    start = time.time()
    lo = centers[kept].min(axis=0) - size_a
    hi = centers[kept].max(axis=0) + size_a
    middle, half = (lo + hi) * 0.5, (hi - lo) * 0.5 * bbox_scale
    lo, hi = middle - half, middle + half
    centers, dims, size_b = make_grid(lo, hi, passB_res)
    kept = carve(views, centers, size_b, min_votes, max_misses)
    print(f"pass B: {dims[0]}x{dims[1]}x{dims[2]} voxels of {size_b:.4g} in {_box(lo, hi)} "
          f"-> {len(kept)} kept ({time.time() - start:.1f} s)")
    if len(kept) == 0:
        raise RuntimeError("pass B kept no voxels")

    occupied = np.zeros(dims, dtype=bool)
    occupied.reshape(-1)[kept] = True
    surface = surface_voxels(occupied)
    out = np.flatnonzero(occupied if solid else surface)
    points = centers[out]
    normals = normals_at(occupied, np.stack(np.unravel_index(out, dims), axis=1))
    colors = colorize(views, points, normals) if color else np.full((len(points), 3), 128, np.uint8)
    print(f"output: {len(points)} {'solid' if solid else 'surface'} voxels")

    coverage = silhouette_coverage(views, centers[np.flatnonzero(surface)], size_b)
    _report(views, coverage, min_votes)
    return Hull(points, normals, colors, occupied, lo, size_b, coverage)


def _box(lo, hi):
    return "[" + ", ".join(f"{a:.3g}..{b:.3g}" for a, b in zip(lo, hi)) + "]"


def _report(views, coverage, min_votes):
    worst = int(np.argmin(coverage))
    print(f"silhouette coverage: mean {coverage.mean():.3f}, min {coverage[worst]:.3f} ({views[worst].name})")
    low = [(c, v.name) for c, v in zip(coverage, views) if c < 0.9]
    if low:
        print(f"  {len(low)} view(s) below 0.9: " + ", ".join(f"{name} {c:.2f}" for c, name in sorted(low)[:10]))
        print("  low coverage = this view's pose or mask disagrees with the others")
    # A mask that touches the image border may cut the object off, and a part outside the frame in
    # more than V - min_votes views is carved. Lowering --min-votes lets it go unseen in more views.
    cut = [v.name for v in views if v.mask[0].any() or v.mask[-1].any() or v.mask[:, 0].any() or v.mask[:, -1].any()]
    if coverage[worst] < 0.99 and len(cut) > len(views) - min_votes and min_votes > 1:
        print(f"  {len(cut)} of {len(views)} masks touch the image border (e.g. {cut[0]}): if the object leaves "
              f"the frame there, lower --min-votes (try {min_votes - 1})")


# ----------------------------------------------------------------------------- output files

PLY_FIELDS = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]


def write_ply(path, points, normals, colors):
    """Binary PLY with the exact layout Inria's 3D Gaussian Splatting expects (it reads nx, ny, nz)."""
    vertices = np.empty(len(points), dtype=PLY_FIELDS)
    for column, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, column]
        vertices["n" + name] = normals[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, column]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header += [f"property {'float' if kind == '<f4' else 'uchar'} {name}" for name, kind in PLY_FIELDS]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header + ["end_header"]) + "\n").encode("ascii"))
        f.write(vertices.tobytes())


def export_to_colmap(sparse_dir, hull):
    """Store the hull as sparse/0/points3D.bin + points3D.ply, the starting point cloud that
    3DGS / NeRF code reads. The first time, existing files are kept as points3D_original.*"""
    sparse_dir = Path(sparse_dir)
    for name in ("points3D.bin", "points3D.ply"):
        current, backup = sparse_dir / name, sparse_dir / name.replace("points3D", "points3D_original")
        if current.exists() and not backup.exists():
            shutil.copy2(current, backup)
    points = {i + 1: Point3D(i + 1, tuple(xyz), tuple(int(c) for c in rgb))
              for i, (xyz, rgb) in enumerate(zip(hull.points, hull.colors))}
    write_points3d_bin(points, sparse_dir / "points3D.bin")
    write_ply(sparse_dir / "points3D.ply", hull.points, hull.normals, hull.colors)
