"""
body_model.py
=============
3D geometric models of space debris bodies.

Provides
--------
  DebrisModel         : container for vertices, faces, edges, keypoints
  ariane_model()      : Ariane 44L upper stage (hollow cylinder + nozzle)
  cubesat_3u_model()  : 3U CubeSat (rectangular box)
  custom_cylinder()   : generic solid/hollow cylinder

Coordinate convention
---------------------
  Body frame: x along symmetry axis (axial), y/z transverse.
  Origin at geometric centre of the body.

Usage
-----
  model = ariane_model()
  verts = model.vertices          # (N, 3) float64 [m]
  edges = model.edges             # (M, 2) int indices into verts
  faces = model.faces             # (F, 3 or 4) int — triangles/quads
  kpts  = model.keypoints         # dict {name: (3,) position [m]}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DebrisModel:
    """Container for a 3D debris body geometry."""

    name        : str
    vertices    : np.ndarray            # (N, 3)  [m]  body frame
    faces       : np.ndarray            # (F, k)  int  triangle or quad indices
    edges       : np.ndarray            # (M, 2)  int  wireframe edges
    face_normals: np.ndarray            # (F, 3)  outward unit normals
    keypoints   : Dict[str, np.ndarray] = field(default_factory=dict)
    metadata    : dict                  = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────────────
    @property
    def n_verts(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def keypoint_array(self) -> np.ndarray:
        """Stack of keypoint positions, shape (K, 3)."""
        return np.array(list(self.keypoints.values()), dtype=float)

    @property
    def keypoint_names(self) -> List[str]:
        return list(self.keypoints.keys())

    def transform(self, R: np.ndarray, t: np.ndarray = None) -> 'DebrisModel':
        """
        Return a new DebrisModel with vertices and normals rotated by R
        and optionally translated by t.
        """
        R = np.asarray(R, dtype=float)
        verts = (R @ self.vertices.T).T
        norms = (R @ self.face_normals.T).T
        kpts  = {k: R @ v for k, v in self.keypoints.items()}
        if t is not None:
            t = np.asarray(t, dtype=float).ravel()
            verts = verts + t
            kpts  = {k: v + t for k, v in kpts.items()}
        return DebrisModel(
            name=self.name,
            vertices=verts,
            faces=self.faces.copy(),
            edges=self.edges.copy(),
            face_normals=norms,
            keypoints=kpts,
            metadata=self.metadata.copy(),
        )

    def scale(self, s: float) -> 'DebrisModel':
        """Return a uniformly scaled copy."""
        return DebrisModel(
            name=self.name,
            vertices=self.vertices * s,
            faces=self.faces.copy(),
            edges=self.edges.copy(),
            face_normals=self.face_normals.copy(),
            keypoints={k: v * s for k, v in self.keypoints.items()},
            metadata=self.metadata.copy(),
        )

    def bounding_box(self) -> tuple:
        """Return (min_xyz, max_xyz) of vertex bounding box."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cylinder_mesh(radius: float, length: float,
                   n_circ: int = 32,
                   n_long: int = 8,
                   cap_start: bool = True,
                   cap_end:   bool = True) -> tuple:
    """
    Build a cylinder mesh centred at the origin, axis along x.

    Returns
    -------
    verts  : (N, 3)
    faces  : list of [i, j, k, l] quad indices (lateral) + [i, j, k] tri (caps)
    edges  : (M, 2) unique edge list
    normals: (F, 3) per-face outward normals
    """
    angles = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
    xs     = np.linspace(-length/2, length/2, n_long + 1)

    # Lateral surface vertices: (n_long+1) rings × n_circ points
    verts = []
    for x in xs:
        for a in angles:
            verts.append([x, radius * np.cos(a), radius * np.sin(a)])

    # Optionally add cap centre vertices
    idx_cap_neg = len(verts)
    if cap_start:
        verts.append([-length/2, 0.0, 0.0])
    idx_cap_pos = len(verts)
    if cap_end:
        verts.append([ length/2, 0.0, 0.0])

    verts = np.array(verts, dtype=float)

    def ring_idx(ix, ic):
        """Vertex index for ring ix (0..n_long), circle position ic (0..n_circ-1)."""
        return ix * n_circ + (ic % n_circ)

    faces   = []
    normals = []

    # Lateral quads
    for ix in range(n_long):
        for ic in range(n_circ):
            i0 = ring_idx(ix,     ic)
            i1 = ring_idx(ix,     ic + 1)
            i2 = ring_idx(ix + 1, ic + 1)
            i3 = ring_idx(ix + 1, ic)
            faces.append([i0, i1, i2, i3])
            # Outward normal = midpoint of quad projected onto cylinder surface
            a_mid = angles[ic] + np.pi / n_circ
            normals.append([0.0, np.cos(a_mid), np.sin(a_mid)])

    # Start cap (faces normal in -x)
    if cap_start:
        for ic in range(n_circ):
            i0 = ring_idx(0, ic)
            i1 = ring_idx(0, ic + 1)
            faces.append([idx_cap_neg, i1, i0])
            normals.append([-1.0, 0.0, 0.0])

    # End cap (faces normal in +x)
    if cap_end:
        last = n_long
        for ic in range(n_circ):
            i0 = ring_idx(last, ic)
            i1 = ring_idx(last, ic + 1)
            faces.append([idx_cap_pos, i0, i1])
            normals.append([1.0, 0.0, 0.0])

    normals = np.array(normals, dtype=float)

    # Build edge set from faces (unique undirected)
    edge_set = set()
    for f in faces:
        n_v = len(f)
        for k in range(n_v):
            a, b = f[k], f[(k+1) % n_v]
            edge_set.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edge_set), dtype=int)

    return verts, faces, edges, normals


def _box_mesh(lx: float, ly: float, lz: float) -> tuple:
    """
    Axis-aligned box (rectangular cuboid) centred at origin.
    lx along x, ly along y, lz along z.
    """
    hx, hy, hz = lx/2, ly/2, lz/2
    verts = np.array([
        [-hx, -hy, -hz],  # 0
        [ hx, -hy, -hz],  # 1
        [ hx,  hy, -hz],  # 2
        [-hx,  hy, -hz],  # 3
        [-hx, -hy,  hz],  # 4
        [ hx, -hy,  hz],  # 5
        [ hx,  hy,  hz],  # 6
        [-hx,  hy,  hz],  # 7
    ], dtype=float)

    faces = [
        [0, 1, 2, 3],  # -z face
        [4, 7, 6, 5],  # +z face
        [0, 4, 5, 1],  # -y face
        [3, 2, 6, 7],  # +y face
        [0, 3, 7, 4],  # -x face
        [1, 5, 6, 2],  # +x face
    ]
    normals = np.array([
        [ 0,  0, -1],
        [ 0,  0,  1],
        [ 0, -1,  0],
        [ 0,  1,  0],
        [-1,  0,  0],
        [ 1,  0,  0],
    ], dtype=float)

    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],  # bottom ring
        [4,5],[5,6],[6,7],[7,4],  # top ring
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ], dtype=int)

    return verts, faces, edges, normals


# ── Public model constructors ─────────────────────────────────────────────────

def ariane_model(n_circ: int = 24, n_long: int = 6) -> DebrisModel:
    """
    Ariane 44L H10 upper stage — hollow cylinder with nozzle stub.

    Dimensions
    ----------
      Body cylinder : radius 1.4 m, length 8.0 m
      Nozzle cone   : approximated as a shorter cylinder (radius 0.6 m, length 1.0 m)
                      attached to the -x end

    Keypoints (body frame, origin = cylinder centre)
    -------------------------------------------------
      nose        : +x end-cap centre
      nozzle_tip  : -x end of nozzle stub
      nozzle_rim  : -x rim of body cylinder (4 points at cardinal angles)
      mid_ring    : 4 points around mid-body circumference
    """
    R_body = 1.4
    L_body = 8.0
    R_nozzle = 0.6
    L_nozzle = 1.0

    # Main body
    v_body, f_body, e_body, n_body = _cylinder_mesh(
        R_body, L_body, n_circ=n_circ, n_long=n_long,
        cap_start=False, cap_end=True)

    # Nozzle stub — centred at x = -L_body/2 - L_nozzle/2
    v_noz, f_noz, e_noz, n_noz = _cylinder_mesh(
        R_nozzle, L_nozzle, n_circ=n_circ//2, n_long=2,
        cap_start=True, cap_end=False)

    # Offset nozzle vertices
    noz_offset = np.array([-L_body/2 - L_nozzle/2, 0.0, 0.0])
    v_noz = v_noz + noz_offset

    # Merge meshes
    offset = len(v_body)
    v_all  = np.vstack([v_body, v_noz])
    f_all  = f_body + [[i + offset for i in face] for face in f_noz]
    e_all  = np.vstack([e_body, e_noz + offset])
    n_all  = np.vstack([n_body, n_noz])

    # Deduplicate edges
    edge_set = set(map(tuple, e_all.tolist()))
    e_all = np.array(sorted(edge_set), dtype=int)

    # Face list → uniform array (pad triangles to quads with -1)
    max_v = max(len(f) for f in f_all)
    f_arr = np.full((len(f_all), max_v), -1, dtype=int)
    for i, f in enumerate(f_all):
        f_arr[i, :len(f)] = f

    # Keypoints
    angles_4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    kpts = {
        'nose'        : np.array([ L_body/2, 0.0, 0.0]),
        'nozzle_tip'  : np.array([-L_body/2 - L_nozzle, 0.0, 0.0]),
        'body_centre' : np.array([0.0, 0.0, 0.0]),
    }
    for j, a in enumerate(angles_4):
        kpts[f'nozzle_rim_{j}'] = np.array([
            -L_body/2, R_body * np.cos(a), R_body * np.sin(a)])
        kpts[f'mid_ring_{j}']   = np.array([
            0.0, R_body * np.cos(a), R_body * np.sin(a)])
        kpts[f'nose_rim_{j}']   = np.array([
            L_body/2, R_body * np.cos(a), R_body * np.sin(a)])

    return DebrisModel(
        name='Ariane 44L upper stage',
        vertices=v_all,
        faces=f_arr,
        edges=e_all,
        face_normals=n_all,
        keypoints=kpts,
        metadata={'radius': R_body, 'length': L_body,
                  'mass_kg': 1200, 'n_circ': n_circ},
    )


def cubesat_3u_model() -> DebrisModel:
    """
    3U CubeSat (10 × 10 × 30 cm).
    x along long axis, y/z transverse.
    Keypoints: 8 corners + 2 face centres + centre.
    """
    lx, ly, lz = 0.30, 0.10, 0.10
    verts, faces, edges, normals = _box_mesh(lx, ly, lz)

    f_arr = np.array(faces, dtype=int)

    hx, hy, hz = lx/2, ly/2, lz/2
    kpts = {
        'corner_000': np.array([-hx, -hy, -hz]),
        'corner_100': np.array([ hx, -hy, -hz]),
        'corner_110': np.array([ hx,  hy, -hz]),
        'corner_010': np.array([-hx,  hy, -hz]),
        'corner_001': np.array([-hx, -hy,  hz]),
        'corner_101': np.array([ hx, -hy,  hz]),
        'corner_111': np.array([ hx,  hy,  hz]),
        'corner_011': np.array([-hx,  hy,  hz]),
        'face_pos_x': np.array([ hx,  0.,  0.]),
        'face_neg_x': np.array([-hx,  0.,  0.]),
        'centre'    : np.array([ 0.,  0.,  0.]),
    }

    return DebrisModel(
        name='3U CubeSat',
        vertices=verts,
        faces=f_arr,
        edges=edges,
        face_normals=normals,
        keypoints=kpts,
        metadata={'lx': lx, 'ly': ly, 'lz': lz, 'mass_kg': 4},
    )


def custom_cylinder(radius: float, length: float,
                    n_circ: int = 24) -> DebrisModel:
    """
    Generic solid cylinder for custom debris bodies.

    Parameters
    ----------
    radius  : float  [m]
    length  : float  [m]
    n_circ  : int    Circumferential resolution
    """
    v, f_list, e, n = _cylinder_mesh(radius, length,
                                      n_circ=n_circ, n_long=6,
                                      cap_start=True, cap_end=True)

    max_v = max(len(f) for f in f_list)
    f_arr = np.full((len(f_list), max_v), -1, dtype=int)
    for i, f in enumerate(f_list):
        f_arr[i, :len(f)] = f

    angles_4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    kpts = {
        'nose'        : np.array([ length/2, 0., 0.]),
        'tail'        : np.array([-length/2, 0., 0.]),
        'body_centre' : np.array([0., 0., 0.]),
    }
    for j, a in enumerate(angles_4):
        kpts[f'rim_pos_{j}'] = np.array([length/2,  radius*np.cos(a), radius*np.sin(a)])
        kpts[f'rim_neg_{j}'] = np.array([-length/2, radius*np.cos(a), radius*np.sin(a)])

    return DebrisModel(
        name=f'Custom cylinder r={radius:.2f}m L={length:.2f}m',
        vertices=v,
        faces=f_arr,
        edges=e,
        face_normals=n,
        keypoints=kpts,
        metadata={'radius': radius, 'length': length},
    )
