# shape_from_views

Reconstruct an object's shape from posed photos and their silhouette masks (a *visual hull*), and
write it out as a COLMAP `sparse/0` folder. NeRF and 3D Gaussian Splatting code can then start from
it instead of running COLMAP.

This is the readable Python version of [sfm_demo_2.0](https://github.com/kevdozer1/sfm_demo_2.0),
which is written in C++. Both implement the same algorithm with the same options and defaults, and
on the same inputs they keep exactly the same voxels.

```
state estimates (distance + attitude per frame) ──► camera poses ─┐
                                    images + masks ───────────────┴─► visual hull ──► visual_hull.ply
                                                                                  └─► sparse/0/points3D.bin + .ply
```

## Quick start

```bash
pip install -r requirements.txt          # numpy + pillow (pytest for the tests)

# 1. camera poses: either from state lists ...
#    (the demo lists are sfm_demo_2.0/examples/anduril/distances.txt and attitudes.txt; for any
#    ring orbit, `orbit DATASET --write-states DIR` writes matching lists)
python shape_from_views.py poses data/anduril distances.txt attitudes.txt --baseline 0 -21.8014 0
#    ... or for cameras that circle the object on rings (radius,height per ring)
python shape_from_views.py orbit data/anduril --rings "10,-4;12,0;10,4"

# 2. the visual hull (writes data/anduril/visual_hull.ply and sparse/0/points3D.bin + .ply)
python shape_from_views.py hull --data data/anduril
```

A dataset folder looks like this (the layout NeRF/3DGS code expects):

```
data/anduril/
├── images/      0001.jpg ...  the photos; orbit/poses take the frame list and image size from them,
│                              hull only uses them for colour
├── masks/       0001.jpg ...  white = object, black = background (same names, or .png)
└── sparse/0/    cameras.bin, images.bin, points3D.bin   (COLMAP format)
```

No data? The test helper renders a toy plane into a dataset folder:

```bash
python -c "import pathlib, test_shape_from_views as t; t.make_dataset(pathlib.Path('toy'))"
python shape_from_views.py hull --data toy
```

The hull run ends with a sanity check:

```
silhouette coverage: mean 1.000, min 0.999 (0194.jpg)
```

Coverage is the fraction of each mask that the finished hull covers when projected back into that
view. If every pose and mask is right it is about 1.0 everywhere. A wrong pose or a mismatched mask
carves away parts that the other views need, and the affected views are listed by name. Look at
this number before you look at the point cloud.

## Conventions

Most of what went wrong in the original project was a convention mismatch, so here they are:

| what | convention |
|---|---|
| pose | world-to-camera: `X_cam = R @ X_world + t`; camera centre `C = -R.T @ t` |
| camera axes | **x right, y DOWN, z forward** (COLMAP / OpenCV), not OpenGL/Blender's y-up, z-backward |
| quaternion | `(qw, qx, qy, qz)` of R, as in COLMAP files |
| pixels | pixel (i, j) covers [i, i+1) x [j, j+1); a point at (u, v) lands in pixel (floor u, floor v) |
| generated poses | world +Z up, object at the origin, yaw 0 = camera on the -Y axis |
| `distances.txt` | one number per line: camera-to-object distance (> 0) |
| `attitudes.txt` | `d_yaw d_pitch d_roll` per line in degrees, *differences from frame 0*; absolute = `--baseline` + difference |

Yaw and pitch say where the camera sits around the object (azimuth and elevation), so the object's
apparent yaw has the opposite sign. Roll spins the camera about its own viewing direction, and a
generated camera always looks at the origin. Line k of the lists belongs to the k-th image in name
order. If you leave out `--baseline`, frame 0 is assumed to sit at yaw 0, pitch 0 (level with the
object) and roll 0. For the demo orbit the correct baseline is pitch -21.8014.

## How the visual hull works (`visual_hull.py`)

A point can be part of the object only if every view that can see it sees it on the silhouette.
The steps:

1. **Find where to look** (`look_at_point`, `initial_box`). The point closest to every camera's
   viewing axis is where the object is. The starting cube around it is as large as the cameras can
   see.
2. **Fill it with voxels** (`make_grid`): cubic voxels, `--voxel-num` along the longest side.
3. **Carve** (`carve`). Each view puts every voxel in one of three groups:
   * *hit*: the voxel's footprint touches the silhouette;
   * *miss*: the footprint is fully inside the image and only covers background, so the view
     *sees* empty space there;
   * *unseen*: anything else (behind the camera, off the image, or partly off it without touching
     the silhouette), so this view knows nothing.

   A voxel survives with at most `--max-misses` misses (default 0) and at least `--min-votes` hits.
   The default for `--min-votes` is 90% of the views, rounded up, so it is every view below 10
   views. A part of the object that leaves the frame in more views than that allows is still
   carved. Lower `--min-votes` then; the coverage report suggests a value when masks touch the
   image border.

   The footprint (`footprints`) is a rectangle around the voxel's projected centre, `(u, v) +- f*r/z`,
   where r is the radius of the voxel's bounding sphere. For lenses up to about 70 degrees of
   horizontal field of view it contains the whole voxel, so a thin part such as a wing seen edge-on
   is not carved by mistake. Whether a footprint touches the silhouette takes four lookups in a
   summed-area table (`summed_area_table`).
4. **Two passes**. Pass A uses a coarse grid over the big cube to find the object. Pass B uses a
   fine grid (`--passB-res`) over pass A's bounding box.
5. **Output**. Surface voxels only (`--solid` keeps the inside too), outward normals from the
   occupancy gradient, and colours averaged over the views that face each point. The PLY uses
   exactly the layout Inria's 3DGS loader reads: `x y z nx ny nz red green blue`.
6. **Check** (`silhouette_coverage`): the coverage report above.

## All tools

| command | does |
|---|---|
| `orbit DATASET [focal_mm sensor_mm] [--rings R,Z;...] [--out DIR] [--write-states DIR]` | poses for cameras on rings around the object; images are split evenly over the rings |
| `poses DATASET distances.txt attitudes.txt [focal_mm sensor_mm] [--baseline Y P R] [--out DIR]` | poses from state estimates |
| `hull --data DATASET [--voxel-num 64] [--passB-res 200] [--min-votes N] [--max-misses N] [--solid] [--no-color] [--out-ply PATH] [--no-export] ...` | the visual hull |
| `subset DATASET (--random N [--seed 42] \| --every N) [--out DIR]` | copy some images + masks with a matching `sparse/0` |
| `to-txt SPARSE_DIR` | binary COLMAP model to text |
| `transplant TARGET_SPARSE SOURCE_SPARSE [--by-index]` | copy poses and their intrinsics from another model, matched by image name |

Generated cameras get `f = focal_mm / sensor_mm * max(width, height)` pixels, which is Blender's
default "sensor fit: auto", with the principal point at the image centre.

## Reading the code

| file | lines | what |
|---|---|---|
| `poses.py` | ~130 | quaternions, the look-at camera, poses from states, ring orbits |
| `colmap.py` | ~300 | COLMAP `.bin` / `.txt` reading and writing, camera models and projection |
| `visual_hull.py` | ~430 | the algorithm above, the PLY writer and the export to `sparse/0` |
| `shape_from_views.py` | ~300 | the command line |
| `test_shape_from_views.py` | ~280 | tests |

Start with `visual_hull.visual_hull()` and follow the calls. Some arithmetic (`to_camera`,
`look_at_point`, the dot products) is written out term by term instead of using `@` or
`np.linalg.solve`, so it rounds exactly like the C++ version.

## Tests

```bash
python -m pytest -q
```

The unit tests cover the camera axes, quaternions, the OpenCV projection, COLMAP `.bin` / `.txt`
round trips, state replay, the subset shuffle (it matches the C++ pick), mask thresholds and bad
arguments.

The end-to-end tests ray-trace a toy plane (ellipsoid fuselage, thin wings, a tail fin) from 36
cameras, write a real dataset folder and run the command line on it. They check that:

* the hull contains the object;
* the hull is at most 2.5x the object's volume;
* every view's coverage is about 1;
* the old 180-degree-rolled camera convention is caught by the coverage check;
* parts that leave the frame in a few views survive;
* the colours come from the right parts (the wings come out blue);
* the generators reproduce the cameras the images were rendered with.

Limits worth knowing:

* A silhouette cannot see concavities, so the inside of a cup stays filled.
* A part thinner than about one pixel in some mask is not in the data at all.
