"""Read/write COLMAP sparse models and project points with COLMAP camera models.

A sparse model is a folder (usually <dataset>/sparse/0) with three files:
  cameras   - intrinsics (focal length, principal point, distortion) per camera
  images    - one pose per image: world-to-camera rotation (quaternion) + translation
  points3D  - a point cloud (COLMAP's triangulated points; here: the visual hull)
Each exists as .bin (little-endian binary) or .txt. Format: https://colmap.github.io/format.html
"""
import struct
from dataclasses import dataclass, field
from pathlib import Path

# model id -> (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str          # e.g. "SIMPLE_PINHOLE"
    width: int
    height: int
    params: list        # meaning depends on the model, see project()


@dataclass
class Image:
    id: int
    qvec: tuple         # (qw, qx, qy, qz) of the world-to-camera rotation R
    tvec: tuple         # world-to-camera translation t:  X_cam = R @ X_world + t
    camera_id: int
    name: str           # file name inside the images folder
    points2d: list = field(default_factory=list)   # [(x, y, point3d_id), ...]


@dataclass
class Point3D:
    id: int
    xyz: tuple
    rgb: tuple
    error: float = 0.0
    track: list = field(default_factory=list)      # [(image_id, point2d_index), ...]


# ----------------------------------------------------------------------------- projection

def focal(camera):
    """(fx, fy) in pixels."""
    p = camera.params
    if camera.model in ("PINHOLE", "OPENCV"):
        return p[0], p[1]
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0]
    raise ValueError(f"camera model {camera.model} is not supported; undistort the images first "
                     "(colmap image_undistorter) to get a PINHOLE model")


def project(camera, x, y, z):
    """Pixel coordinates (u, v) of camera-frame points (x, y, z) with z > 0.

    Works on floats or numpy arrays. Pixel (0, 0)'s top-left corner is (0, 0), so a point
    lands in pixel (floor(u), floor(v)) - COLMAP's convention.
    """
    a = x / z
    b = y / z
    p = camera.params
    fx, fy = focal(camera)
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        cx, cy = p[1], p[2]
    else:  # PINHOLE, OPENCV
        cx, cy = p[2], p[3]

    if camera.model == "SIMPLE_RADIAL":
        r2 = a * a + b * b
        radial = p[3] * r2
        a, b = a + a * radial, b + b * radial
    elif camera.model == "RADIAL":
        r2 = a * a + b * b
        radial = p[3] * r2 + p[4] * r2 * r2
        a, b = a + a * radial, b + b * radial
    elif camera.model == "OPENCV":
        k1, k2, p1, p2 = p[4], p[5], p[6], p[7]
        r2 = a * a + b * b
        radial = k1 * r2 + k2 * r2 * r2
        a, b = (a + a * radial + 2 * p1 * a * b + p2 * (r2 + 2 * a * a),
                b + b * radial + 2 * p2 * a * b + p1 * (r2 + 2 * b * b))
    return fx * a + cx, fy * b + cy


# ----------------------------------------------------------------------------- binary files

class _Reader:
    """Reads little-endian values one after another and fails loudly on truncated files."""

    def __init__(self, path):
        self.path, self.data, self.pos = path, Path(path).read_bytes(), 0

    def read(self, fmt):
        size = struct.calcsize("<" + fmt)
        if self.pos + size > len(self.data):
            raise ValueError(f"{self.path} is truncated")
        values = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += size
        return values

    def read_string(self):
        end = self.data.find(b"\0", self.pos)
        if end < 0:
            raise ValueError(f"{self.path} is truncated")
        text = self.data[self.pos:end].decode("utf-8")
        self.pos = end + 1
        return text


def read_cameras_bin(path):
    r = _Reader(path)
    cameras = {}
    for _ in range(r.read("Q")[0]):
        cam_id, model_id, width, height = r.read("IiQQ")
        name, n_params = CAMERA_MODELS[model_id]
        cameras[cam_id] = Camera(cam_id, name, width, height, list(r.read(f"{n_params}d")))
    return cameras


def read_images_bin(path):
    r = _Reader(path)
    images = {}
    for _ in range(r.read("Q")[0]):
        image_id, = r.read("I")
        qvec, tvec = r.read("4d"), r.read("3d")
        camera_id, = r.read("I")
        name = r.read_string()
        n_points, = r.read("Q")
        flat = r.read("ddq" * n_points)
        points2d = [flat[i:i + 3] for i in range(0, len(flat), 3)]
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name, points2d)
    return images


def read_points3d_bin(path):
    r = _Reader(path)
    points = {}
    for _ in range(r.read("Q")[0]):
        point_id, = r.read("Q")
        xyz, rgb, (error,) = r.read("3d"), r.read("3B"), r.read("d")
        track_length, = r.read("Q")
        flat = r.read("II" * track_length)
        points[point_id] = Point3D(point_id, xyz, rgb, error, [flat[i:i + 2] for i in range(0, len(flat), 2)])
    return points


def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            f.write(struct.pack("<IiQQ", cam.id, MODEL_IDS[cam.model], cam.width, cam.height))
            f.write(struct.pack(f"<{len(cam.params)}d", *cam.params))


def write_images_bin(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images.values():
            f.write(struct.pack("<I4d3dI", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode("utf-8") + b"\0")
            f.write(struct.pack("<Q", len(im.points2d)))
            for x, y, point3d_id in im.points2d:
                f.write(struct.pack("<ddq", x, y, point3d_id))


def write_points3d_bin(points, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points.values():
            f.write(struct.pack("<Q3d3Bd", pt.id, *pt.xyz, *pt.rgb, pt.error))
            f.write(struct.pack("<Q", len(pt.track)))
            for image_id, point2d_index in pt.track:
                f.write(struct.pack("<II", image_id, point2d_index))


# ----------------------------------------------------------------------------- text files

def _num(value):
    """Shortest text that reads back to exactly the same float (works for numpy floats too)."""
    return repr(float(value))


def _data_lines(path):
    """All lines except '#' comments (empty lines are kept: images.txt needs them)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if not line.startswith("#")]


def read_cameras_txt(path):
    cameras = {}
    for line in _data_lines(path):
        if line.strip():
            cam_id, model, width, height, *params = line.split()
            cameras[int(cam_id)] = Camera(int(cam_id), model, int(width), int(height), [float(p) for p in params])
    return cameras


def read_images_txt(path):
    """Two lines per image; the second (POINTS2D) line may be empty and must not be skipped."""
    lines, images, i = _data_lines(path), {}, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, *name = lines[i].split()
        values = lines[i + 1].split() if i + 1 < len(lines) else []
        points2d = [(float(values[k]), float(values[k + 1]), int(values[k + 2])) for k in range(0, len(values), 3)]
        images[int(image_id)] = Image(int(image_id), (float(qw), float(qx), float(qy), float(qz)),
                                      (float(tx), float(ty), float(tz)), int(camera_id), " ".join(name), points2d)
        i += 2
    return images


def read_points3d_txt(path):
    points = {}
    for line in _data_lines(path):
        if line.strip():
            v = line.split()
            track = [(int(v[k]), int(v[k + 1])) for k in range(8, len(v), 2)]
            points[int(v[0])] = Point3D(int(v[0]), tuple(map(float, v[1:4])), tuple(map(int, v[4:7])), float(v[7]), track)
    return points


def write_cameras_txt(cameras, path):
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"# Number of cameras: {len(cameras)}"]
    lines += [" ".join(map(str, [c.id, c.model, c.width, c.height, *map(_num, c.params)])) for c in cameras.values()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_txt(images, path):
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)",
             f"# Number of images: {len(images)}"]
    for im in images.values():
        lines.append(" ".join(map(str, [im.id, *map(_num, im.qvec), *map(_num, im.tvec), im.camera_id, im.name])))
        lines.append(" ".join(f"{_num(x)} {_num(y)} {pid}" for x, y, pid in im.points2d))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points3d_txt(points, path):
    lines = ["# 3D point list with one line of data per point:",
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
             f"# Number of points: {len(points)}"]
    for pt in points.values():
        track = [str(v) for pair in pt.track for v in pair]
        lines.append(" ".join([str(pt.id), *map(_num, pt.xyz), *map(str, pt.rgb), _num(pt.error), *track]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- whole models

READERS = {"cameras": (read_cameras_bin, read_cameras_txt),
           "images": (read_images_bin, read_images_txt),
           "points3D": (read_points3d_bin, read_points3d_txt)}
WRITERS = {"cameras": (write_cameras_bin, write_cameras_txt),
           "images": (write_images_bin, write_images_txt),
           "points3D": (write_points3d_bin, write_points3d_txt)}


def read_model(sparse_dir):
    """(cameras, images, points3d) dicts keyed by id. Prefers .bin, falls back to .txt;
    a missing points3D file gives an empty dict."""
    sparse_dir, model = Path(sparse_dir), []
    for part, (read_bin, read_txt) in READERS.items():
        if (sparse_dir / f"{part}.bin").exists():
            model.append(read_bin(sparse_dir / f"{part}.bin"))
        elif (sparse_dir / f"{part}.txt").exists():
            model.append(read_txt(sparse_dir / f"{part}.txt"))
        elif part == "points3D":
            model.append({})
        else:
            raise FileNotFoundError(f"no {part}.bin or {part}.txt in {sparse_dir}")
    return tuple(model)


def write_model(sparse_dir, cameras, images, points3d, text=False):
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for part, data in zip(WRITERS, (cameras, images, points3d)):
        write_bin, write_txt = WRITERS[part]
        if text:
            write_txt(data, sparse_dir / f"{part}.txt")
        else:
            write_bin(data, sparse_dir / f"{part}.bin")
