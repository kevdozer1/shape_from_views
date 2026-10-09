"""shape_from_views: camera poses from state estimates + a visual hull, as COLMAP files.

  python shape_from_views.py orbit      DATASET [focal_mm sensor_mm] [--rings "10,-4;12,0;10,4"] [--out DIR] [--write-states DIR]
  python shape_from_views.py estimate   DATASET [focal_mm sensor_mm] [--out DIR] [--origin PHOTO] [--distance D]
  python shape_from_views.py poses      DATASET distances.txt attitudes.txt [focal_mm sensor_mm] [--baseline YAW PITCH ROLL] [--photos LIST] [--out DIR]
  python shape_from_views.py hull       --data DATASET [--voxel-num 64] [--passB-res 200] [--min-votes N] [--max-misses N] [--solid] ...
  python shape_from_views.py subset     DATASET (--random N [--seed 42] | --every N) [--out DIR]
  python shape_from_views.py to-txt     SPARSE_DIR
  python shape_from_views.py transplant TARGET_SPARSE SOURCE_SPARSE [--by-index]

A DATASET folder holds images/, masks/ and sparse/0/ (the COLMAP layout NeRF/3DGS code expects).
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import visual_hull as vh
from colmap import (Camera, Image, read_model, write_cameras_bin, write_images_bin, write_model)
from poses import orbit_states, pose_from_state, rotmat_to_qvec, state_from_pose, to_differential

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------- shared helpers

def list_images(images_dir):
    """Sorted image names and their common (width, height)."""
    images_dir = Path(images_dir)
    names = sorted(p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not names:
        sys.exit(f"error: no .jpg/.jpeg/.png images in {images_dir}")
    sizes = set()
    for name in names:
        with PILImage.open(images_dir / name) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        sys.exit(f"error: the images have different sizes {sorted(sizes)}")
    return names, sizes.pop()


def render_camera(width, height, focal_mm, sensor_mm):
    """Pinhole camera of a Blender render: with sensor fit AUTO the sensor spans the longer side."""
    f = focal_mm / sensor_mm * max(width, height)
    return Camera(1, "SIMPLE_PINHOLE", width, height, [f, width / 2.0, height / 2.0])


def write_generated_model(out_dir, camera, names, states):
    """sparse/0 with one pose per (distance, yaw, pitch, roll) state and an empty point cloud.
    State i belongs to the i-th image in name order."""
    images = {}
    for i, (name, state) in enumerate(zip(names, states)):
        if not state[0] > 0:
            sys.exit(f"error: frame {i + 1} ({name}) has distance {state[0]}; distances must be > 0")
        R, t = pose_from_state(*state)
        images[i + 1] = Image(i + 1, rotmat_to_qvec(R), tuple(float(v) for v in t), camera.id, name)
    sparse_dir = Path(out_dir) / "sparse" / "0"
    stale = sparse_dir / "points3D.ply"
    if stale.exists():   # 3DGS reads points3D.ply before points3D.bin: an old hull would sneak in
        stale.replace(sparse_dir / "points3D_previous.ply")
        print(f"moved the old {stale.name} to points3D_previous.ply (it belongs to the previous poses)")
    write_model(sparse_dir, {camera.id: camera}, images, {})
    print(f"wrote {sparse_dir}: {len(names)} images of {camera.width}x{camera.height}, f = {camera.params[0]:.6g} px")
    print(f"  state 1 -> {names[0]}, state {len(names)} -> {names[-1]}")


def read_rows(path, columns):
    """Numbers from a text file, `columns` per line; empty lines and '#' comments are skipped."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.strip().startswith("#"):
            values = [float(v) for v in line.split()]
            if len(values) != columns:
                sys.exit(f"error: {path}:{number}: expected {columns} number(s), got {len(values)}")
            rows.append(values)
    return rows


def splitmix64(seed):
    """Tiny portable random number generator (the C++ version produces the same numbers)."""
    mask = (1 << 64) - 1
    state = seed & mask
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def shuffled(items, seed):
    """Fisher-Yates shuffle driven by splitmix64."""
    items, rng = list(items), splitmix64(seed)
    for i in range(len(items) - 1, 0, -1):
        j = next(rng) % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items


def backup_once(path):
    """Copy x.bin to x_original.bin unless that backup already exists."""
    path = Path(path)
    backup = path.with_name(path.stem + "_original" + path.suffix)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


# ----------------------------------------------------------------------------- commands

def cmd_orbit(args):
    names, (width, height) = list_images(Path(args.dataset) / "images")
    rings = [tuple(float(v) for v in ring.split(",")) for ring in args.rings.split(";")]
    states = orbit_states(len(names), rings)
    write_generated_model(args.out or args.dataset, render_camera(width, height, args.focal_mm, args.sensor_mm), names, states)
    if args.write_states:
        baseline, differences = to_differential(states)
        out = Path(args.write_states)
        out.mkdir(parents=True, exist_ok=True)
        (out / "distances.txt").write_text("".join(f"{d!r}\n" for d, _, _, _ in differences))
        (out / "attitudes.txt").write_text("".join(f"{y!r} {p!r} {r!r}\n" for _, y, p, r in differences))
        print(f"wrote {out}/distances.txt and attitudes.txt; replay them with --baseline {baseline[0]!r} {baseline[1]!r} {baseline[2]!r}")


def read_photo_list(path, available):
    """Photo names, one per line (the rows of the state lists belong to these photos, in this order)."""
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [n for n in names if n not in available]
    if missing:
        sys.exit(f"error: {path} names photos that are not in images/: {missing[:3]}")
    return names


def cmd_poses(args):
    names, (width, height) = list_images(Path(args.dataset) / "images")
    if args.photos:                                   # the rows belong to these photos (e.g. the ones `estimate` placed)
        names = read_photo_list(args.photos, set(names))
    distances = [row[0] for row in read_rows(args.distances, 1)]
    attitudes = read_rows(args.attitudes, 3)
    if not (len(distances) == len(attitudes) == len(names)):
        sys.exit(f"error: {len(names)} {'listed photos' if args.photos else 'images'} but {len(distances)} distances "
                 f"and {len(attitudes)} attitudes")
    if args.baseline is None:
        print("warning: no --baseline given; frame 0 is assumed to be level with the target (yaw 0, pitch 0, roll 0)")
    yaw0, pitch0, roll0 = args.baseline or (0.0, 0.0, 0.0)
    states = [(d, yaw0 + dy, pitch0 + dp, roll0 + dr) for d, (dy, dp, dr) in zip(distances, attitudes)]
    write_generated_model(args.out or args.dataset, render_camera(width, height, args.focal_mm, args.sensor_mm), names, states)
    print(f"baseline (deg): yaw {yaw0}, pitch {pitch0}, roll {roll0}")


def relative_to_states(R_rel, d_rel, distance):
    """(distance, yaw, pitch, roll) rows from what `estimate` measures: each observer's attitude R_rel relative to
    observer 1 and its distance as a multiple of observer 1's. Observer 1 is the origin: yaw 0, pitch 0, roll 0 at
    `distance` (replay with --baseline 0 0 0). Neither choice changes the shape: the origin only turns the scene,
    the distance only scales it."""
    R0, _ = pose_from_state(distance, 0.0, 0.0, 0.0)
    return [state_from_pose(Rr @ R0, float(dr) * distance) for Rr, dr in zip(R_rel, d_rel)]


def write_state_lists(out, names, states):
    """distances.txt, attitudes.txt and photos.txt (which photo each row belongs to), rows in photo-name order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = dict(zip(names, states))
    order = sorted(rows)
    (out / "photos.txt").write_text("".join(f"{n}\n" for n in order))
    (out / "distances.txt").write_text("".join(f"{rows[n][0]!r}\n" for n in order))
    (out / "attitudes.txt").write_text("".join(f"{rows[n][1]!r} {rows[n][2]!r} {rows[n][3]!r}\n" for n in order))


def cmd_estimate(args):
    """State lists measured from the photos: every photo's attitude and distance relative to observer 1."""
    import estimate                                   # needs torch, lightglue, numba, opencv and scipy
    data = Path(args.dataset)
    names, (width, height) = list_images(data / "images")
    if args.origin:
        if args.origin not in names:
            sys.exit(f"error: --origin {args.origin} is not in {data / 'images'}")
        names.remove(args.origin)
        names.insert(0, args.origin)                  # observer 1 = the origin: every estimate is relative to it
    masks = [vh.find_mask(data / "masks", n) for n in names]
    if None in masks:
        sys.exit(f"error: no mask for {names[masks.index(None)]} in {data / 'masks'}")
    camera = render_camera(width, height, args.focal_mm, args.sensor_mm)
    f, cx, cy = camera.params
    K = np.array([[f, 0.0, cx - 0.5], [0.0, f, cy - 0.5], [0.0, 0.0, 1.0]])   # OpenCV pixel centres sit at integers
    estimate.warm_up()
    result = estimate.estimate([data / "images" / n for n in names], masks, K,
                               progress=lambda stage, s: print(f"  {stage:9s} {s:7.2f} s", flush=True))
    placed = [names[i] for i in result["registered"]]
    if not placed or placed[0] != names[0]:
        sys.exit(f"error: observer 1 ({names[0]}) could not be placed: {result.get('status', 'no reason given')}")
    out = Path(args.out) if args.out else data / "states"
    write_state_lists(out, placed, relative_to_states(result["R_rel"], result["d_rel"], args.distance))
    print(f"placed {len(placed)} of {len(names)} photos in {result['seconds']:.1f} s (observer 1 = {names[0]})")
    for index, reason in sorted(result.get("dropped", {}).items()):
        print(f"  not placed: {names[int(index)]} ({reason})")
    print(f"wrote {out}/distances.txt, attitudes.txt and photos.txt; replay them with\n"
          f"  python shape_from_views.py poses {data} {out}/distances.txt {out}/attitudes.txt "
          f"--baseline 0 0 0 --photos {out}/photos.txt")


def cmd_hull(args):
    data = Path(args.data)
    sparse_dir = Path(args.sparse) if args.sparse else data / "sparse" / "0"
    views = vh.load_views(args.images or data / "images", args.masks or data / "masks", sparse_dir)
    hull = vh.visual_hull(views, voxel_num=args.voxel_num, passB_res=args.passB_res, bbox_scale=args.bbox_scale,
                          cube_size=args.cube_size, shift=(args.shift_x, args.shift_y, args.shift_z),
                          min_votes=args.min_votes, max_misses=args.max_misses, solid=args.solid,
                          color=not args.no_color)
    out = Path(args.out_ply) if args.out_ply else data / "visual_hull.ply"
    vh.write_ply(out, hull.points, hull.normals, hull.colors)
    print(f"wrote {out}")
    if not args.no_export:
        vh.export_to_colmap(sparse_dir, hull)
        print(f"wrote {sparse_dir / 'points3D.bin'} and points3D.ply (originals kept as points3D_original.*)")


def cmd_subset(args):
    base = Path(args.dataset).resolve()
    cameras, images, _ = read_model(base / "sparse" / "0")
    by_name = {image.name: image for image in images.values()}
    candidates = sorted(p.name for p in (base / "images").iterdir()
                        if p.name in by_name and vh.find_mask(base / "masks", p.name))
    every = args.every is not None
    n = args.every if every else args.random
    if n <= 0 or (not every and n > len(candidates)):
        sys.exit(f"error: cannot pick {n} of {len(candidates)} images that have a mask and a pose")
    chosen = candidates[::n] if every else sorted(shuffled(candidates, args.seed)[:n])
    out = Path(args.out) if args.out else base.parent / f"{base.name}_{'every' if every else 'any'}_{n}"
    if out.exists():
        sys.exit(f"error: {out} already exists - delete it or pass --out")
    for folder in ("images", "masks"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name in chosen:
        shutil.copy2(base / "images" / name, out / "images" / name)
        mask = vh.find_mask(base / "masks", name)
        shutil.copy2(mask, out / "masks" / mask.name)
    subset = {k + 1: Image(k + 1, by_name[name].qvec, by_name[name].tvec, by_name[name].camera_id, name)
              for k, name in enumerate(chosen)}
    used_cameras = {cam_id: cameras[cam_id] for cam_id in sorted({image.camera_id for image in subset.values()})}
    write_model(out / "sparse" / "0", used_cameras, subset, {})
    how = f"every {n}" if every else f"random {n}, seed {args.seed}"
    log = [f"subset of {base}: {how}"] + [f"{k + 1}: {name} (original index {candidates.index(name)})" for k, name in enumerate(chosen)]
    (out / "sparse" / "0" / "subset_log.txt").write_text("\n".join(log) + "\n")
    print(f"wrote {out} with {len(chosen)} images ({how})")


def cmd_to_txt(args):
    cameras, images, points = read_model(args.sparse_dir)
    write_model(args.sparse_dir, cameras, images, points, text=True)
    print(f"wrote cameras.txt, images.txt, points3D.txt in {args.sparse_dir}")


def cmd_transplant(args):
    target, source = Path(args.target), Path(args.source)
    target_cameras, target_images, _ = read_model(target)
    source_cameras, source_images, _ = read_model(source)
    targets = sorted(target_images.values(), key=lambda im: im.id)
    if args.by_index:
        print("WARNING: --by-index pairs images by position, not by name. That is only correct if both "
              "models contain the same frames in the same order - check the output!")
        ordered = sorted(source_images.values(), key=lambda im: im.id)
        if len(ordered) < len(targets):
            sys.exit("error: the source model has fewer images than the target")
        pairs = [(image, ordered[i * len(ordered) // len(targets)]) for i, image in enumerate(targets)]
    else:
        by_name = {image.name: image for image in source_images.values()}
        missing = [image.name for image in targets if image.name not in by_name]
        if missing:
            sys.exit(f"error: {len(missing)} target image(s) are not in the source model, e.g. {missing[:5]}")
        pairs = [(image, by_name[image.name]) for image in targets]
    for name in ("images.bin", "cameras.bin"):
        backup_once(target / name)
    for image, source_image in pairs:   # a pose is only valid together with its own intrinsics
        image.qvec, image.tvec, image.camera_id = source_image.qvec, source_image.tvec, source_image.camera_id
    used_cameras = {cam_id: source_cameras[cam_id] for cam_id in sorted({image.camera_id for image in targets})}
    write_cameras_bin(used_cameras, target / "cameras.bin")
    write_images_bin({image.id: image for image in targets}, target / "images.bin")
    print(f"copied {len(pairs)} poses and their cameras from {source} into {target} (originals kept as *_original.bin)")


# ----------------------------------------------------------------------------- argument parsing

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def lens_and_output(p):
        p.add_argument("focal_mm", type=float, nargs="?", default=40.0, help="lens focal length (default 40)")
        p.add_argument("sensor_mm", type=float, nargs="?", default=36.0, help="sensor width (default 36)")
        p.add_argument("--out", help="write sparse/0 under this folder instead of DATASET")

    p = commands.add_parser("orbit", help="poses for cameras circling the target on rings")
    p.add_argument("dataset")
    lens_and_output(p)
    p.add_argument("--rings", default="10,-4;12,0;10,4", help='"radius,height;..." (images split evenly)')
    p.add_argument("--write-states", help="also write distances.txt/attitudes.txt for these poses here")
    p.set_defaults(run=cmd_orbit)

    p = commands.add_parser("poses", help="poses from distance + differential attitude lists")
    p.add_argument("dataset")
    p.add_argument("distances")
    p.add_argument("attitudes")
    lens_and_output(p)
    p.add_argument("--baseline", type=float, nargs=3, metavar=("YAW", "PITCH", "ROLL"),
                   help="absolute attitude of frame 0 in degrees (the lists hold differences)")
    p.add_argument("--photos", help="the rows belong to the photos named in this file (default: all images, by name)")
    p.set_defaults(run=cmd_poses)

    p = commands.add_parser("estimate", help="state lists measured from the photos (observer 1 = the origin)")
    p.add_argument("dataset", help="folder with images/ and masks/")
    p.add_argument("focal_mm", type=float, nargs="?", default=40.0, help="lens focal length (default 40)")
    p.add_argument("sensor_mm", type=float, nargs="?", default=36.0, help="sensor width (default 36)")
    p.add_argument("--out", help="folder for distances.txt, attitudes.txt and photos.txt (default DATASET/states)")
    p.add_argument("--origin", help="the photo that is observer 1 (default: the first by name)")
    p.add_argument("--distance", type=float, default=1.0,
                   help="observer 1's distance to the target: sets the scale only, not the shape (default 1)")
    p.set_defaults(run=cmd_estimate)

    p = commands.add_parser("hull", help="visual hull from images + masks + sparse/0")
    p.add_argument("--data", required=True, help="dataset folder")
    p.add_argument("--images", help="default DATA/images")
    p.add_argument("--masks", help="default DATA/masks")
    p.add_argument("--sparse", help="default DATA/sparse/0")
    p.add_argument("--voxel-num", type=int, default=64, help="pass A voxels along the longest side")
    p.add_argument("--passB-res", type=int, default=200, help="pass B voxels along the longest side")
    p.add_argument("--bbox-scale", type=float, default=1.0, help="enlarge pass A's bounding box for pass B")
    p.add_argument("--cube-size", type=float, default=0.0, help="pass A half-size (0 = automatic)")
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--shift-z", type=float, default=0.0)
    p.add_argument("--min-votes", type=int, default=0,
                   help="views that must see a voxel on the silhouette (0 = 90%% of the views)")
    p.add_argument("--max-misses", type=int, default=0,
                   help="views allowed to see background at a kept voxel (raise for imperfect masks)")
    p.add_argument("--solid", action="store_true", help="output interior voxels too")
    p.add_argument("--no-color", action="store_true", help="skip colouring (faster)")
    p.add_argument("--out-ply", help="default DATA/visual_hull.ply")
    p.add_argument("--no-export", action="store_true", help="do not write points3D.bin/.ply into the sparse folder")
    p.set_defaults(run=cmd_hull)

    p = commands.add_parser("subset", help="copy a subset of a dataset with matching sparse/0")
    p.add_argument("dataset")
    pick = p.add_mutually_exclusive_group(required=True)
    pick.add_argument("--random", type=int, help="N random images")
    pick.add_argument("--every", type=int, help="every N-th image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    p.set_defaults(run=cmd_subset)

    p = commands.add_parser("to-txt", help="convert a binary sparse model to text")
    p.add_argument("sparse_dir")
    p.set_defaults(run=cmd_to_txt)

    p = commands.add_parser("transplant", help="copy poses (and their cameras) from another model, by image name")
    p.add_argument("target")
    p.add_argument("source")
    p.add_argument("--by-index", action="store_true", help="legacy: pair images by position (risky)")
    p.set_defaults(run=cmd_transplant)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        args.run(args)
    except (ValueError, RuntimeError, OSError) as error:   # bad options or data: a message, not a traceback
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
