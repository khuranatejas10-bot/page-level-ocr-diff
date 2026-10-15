"""
Text-block and text-line grouping.
Paper 1: Delaunay triangulation + scale-adaptive edge pruning for block grouping,
         then orientation/scale-aware polynomial fitting for line grouping.
"""
import math
import warnings
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

try:
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not available; falling back to proximity-based grouping.")

from .components import TextComponent


@dataclass
class TextLine:
    """An ordered list of CCs forming a text line."""
    id: int
    block_id: int
    components: List[TextComponent] = field(default_factory=list)
    poly_coeffs: Optional[np.ndarray] = None  # Degree-k polynomial y=f(x) in skew-corrected space
    orientation_deg: float = 0.0
    scale: float = 20.0
    text: str = ""
    confidence: float = 1.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        if not self.components:
            return (0, 0, 0, 0)
        xs = [c.bbox[0] for c in self.components]
        ys = [c.bbox[1] for c in self.components]
        x2s = [c.bbox[0] + c.bbox[2] for c in self.components]
        y2s = [c.bbox[1] + c.bbox[3] for c in self.components]
        return (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))

    @property
    def center(self) -> Tuple[float, float]:
        bx, by, bw, bh = self.bbox
        return (bx + bw / 2, by + bh / 2)

    def eval_poly_at(self, x: float) -> float:
        """Evaluate the fitted polynomial at x (in skew-corrected space)."""
        if self.poly_coeffs is None:
            return self.center[1]
        return float(np.polyval(self.poly_coeffs, x))


@dataclass
class TextBlock:
    """A group of CCs forming a text block (column/paragraph)."""
    id: int
    components: List[TextComponent] = field(default_factory=list)
    lines: List[TextLine] = field(default_factory=list)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        if not self.components:
            return (0, 0, 0, 0)
        xs = [c.bbox[0] for c in self.components]
        ys = [c.bbox[1] for c in self.components]
        x2s = [c.bbox[0] + c.bbox[2] for c in self.components]
        y2s = [c.bbox[1] + c.bbox[3] for c in self.components]
        return (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))


# ── Block grouping via Delaunay triangulation ──────────────────────────────

def group_components_into_blocks(
    components: List[TextComponent],
    epsilon: float = 2.0,
    scale_diff_thresh: float = 50.0,
) -> List[TextBlock]:
    """
    Paper 1 Section III-D: Delaunay triangulation + scale-adaptive edge pruning.
    Edge (p, q) is removed when dist(p, q) >= epsilon * min(scale_p, scale_q).

    Falls back to simple proximity grouping when scipy is unavailable or < 4 CCs.

    Returns list of TextBlock objects, sorted top-to-bottom, left-to-right.
    """
    if len(components) == 0:
        return []
    if len(components) == 1:
        b = TextBlock(id=0, components=components[:])
        return [b]

    if _HAS_SCIPY and len(components) >= 4:
        blocks = _delaunay_grouping(components, epsilon, scale_diff_thresh)
    else:
        blocks = _proximity_grouping(components, epsilon)

    # Sort blocks top-to-bottom, left-to-right
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    for i, b in enumerate(blocks):
        b.id = i
    return blocks


def _delaunay_grouping(components: List[TextComponent], epsilon: float,
                       scale_diff_thresh: float) -> List[TextBlock]:
    """Delaunay triangulation with scale-adaptive edge pruning."""
    centers = np.array([c.center for c in components], dtype=np.float64)
    # Add tiny jitter to avoid degenerate triangulation
    jitter = np.random.normal(0, 1e-3, centers.shape)
    try:
        tri = Delaunay(centers + jitter)
    except Exception:
        return _proximity_grouping(components, epsilon)

    n = len(components)
    adj: List[Set[int]] = [set() for _ in range(n)]

    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            if u >= n or v >= n:
                continue
            pu, pv = components[u], components[v]
            dist = float(np.linalg.norm(centers[u] - centers[v]))
            max_dist = epsilon * min(pu.scale, pv.scale)
            scale_diff = abs(pu.scale - pv.scale)
            if dist < max_dist and scale_diff < scale_diff_thresh:
                adj[u].add(v)
                adj[v].add(u)

    return _connected_components_from_adj(components, adj)


def _proximity_grouping(components: List[TextComponent], epsilon: float) -> List[TextBlock]:
    """Simple proximity-based grouping fallback."""
    n = len(components)
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = components[i], components[j]
            dist = math.hypot(pi.center[0] - pj.center[0], pi.center[1] - pj.center[1])
            max_dist = epsilon * min(pi.scale, pj.scale)
            if dist < max_dist:
                adj[i].add(j)
                adj[j].add(i)
    return _connected_components_from_adj(components, adj)


def _connected_components_from_adj(components: List[TextComponent],
                                   adj: List[Set[int]]) -> List[TextBlock]:
    """BFS to find connected subgraphs."""
    n = len(components)
    visited = [False] * n
    blocks = []
    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        group = [start]
        while queue:
            curr = queue.pop(0)
            for nb in adj[curr]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
                    group.append(nb)
        b = TextBlock(id=len(blocks), components=[components[i] for i in group])
        blocks.append(b)
    return blocks


# ── Line grouping within blocks ────────────────────────────────────────────

def group_block_into_lines(
    block: TextBlock,
    max_poly_degree: int = 4,
    deviation_scale_factor: float = 0.25,  # Paper: s_bar / 4
) -> List[TextLine]:
    """
    Paper 1 Section III-E: bottom-up grouping with polynomial curve fitting.
    CCs are assigned to lines whose fitted polynomial deviates by at most
    (avg_scale * deviation_scale_factor) from each CC center.

    Returns lines sorted by baseline position (top-to-bottom in the block).
    """
    comps = block.components
    if not comps:
        return []

    # Dominant block orientation for skew correction
    orient_vals = np.array([c.orientation_deg for c in comps])
    dominant_angle = float(np.median(orient_vals))
    angle_rad = math.radians(-dominant_angle)  # negative = correct skew

    # Compute block center for rotation
    cx_mean = np.mean([c.center[0] for c in comps])
    cy_mean = np.mean([c.center[1] for c in comps])

    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    def rotate(px, py):
        dx, dy = px - cx_mean, py - cy_mean
        return cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy

    # Sort CCs by rotated y (top to bottom), then x (left to right)
    rot_coords = [rotate(c.center[0], c.center[1]) for c in comps]
    order = sorted(range(len(comps)), key=lambda i: (rot_coords[i][1], rot_coords[i][0]))
    comps_sorted = [comps[i] for i in order]
    rot_sorted = [rot_coords[i] for i in order]

    lines: List[TextLine] = []
    used = [False] * len(comps_sorted)
    line_id_counter = 0

    for seed_idx in range(len(comps_sorted)):
        if used[seed_idx]:
            continue
        used[seed_idx] = True
        line_comps = [comps_sorted[seed_idx]]
        line_rots = [rot_sorted[seed_idx]]

        changed = True
        while changed:
            changed = False
            s_bar = float(np.mean([c.scale for c in line_comps]))
            threshold = s_bar * deviation_scale_factor

            # Fit polynomial to current line
            if len(line_comps) >= 2:
                xs = np.array([r[0] for r in line_rots])
                ys = np.array([r[1] for r in line_rots])
                deg = min(max_poly_degree, len(line_comps) - 1)
                try:
                    coeffs = np.polyfit(xs, ys, deg=deg)
                except np.linalg.LinAlgError:
                    coeffs = np.array([np.mean(ys)])
            else:
                coeffs = None

            for cand_idx in range(len(comps_sorted)):
                if used[cand_idx]:
                    continue
                cx_r, cy_r = rot_sorted[cand_idx]
                cand_scale = comps_sorted[cand_idx].scale

                # Vertical proximity check
                y_center_line = float(np.mean([r[1] for r in line_rots]))
                v_dist = abs(cy_r - y_center_line)
                if v_dist > s_bar * 0.75:
                    continue

                # Polynomial deviation check
                if coeffs is not None:
                    y_pred = float(np.polyval(coeffs, cx_r))
                    rmse = abs(cy_r - y_pred)
                else:
                    rmse = v_dist

                if rmse <= threshold + 1e-3:
                    # Verify adding this CC doesn't violate RMSE for whole group
                    test_comps = line_comps + [comps_sorted[cand_idx]]
                    test_rots = line_rots + [(cx_r, cy_r)]
                    if _check_rmse(test_rots, max_poly_degree, s_bar, deviation_scale_factor):
                        used[cand_idx] = True
                        line_comps.append(comps_sorted[cand_idx])
                        line_rots.append((cx_r, cy_r))
                        changed = True

        if line_comps:
            # Sort line CCs left-to-right by rotated x
            line_order = sorted(range(len(line_comps)), key=lambda i: line_rots[i][0])
            line_comps_sorted = [line_comps[i] for i in line_order]
            line_rots_sorted = [line_rots[i] for i in line_order]

            # Final polynomial fit
            if len(line_comps_sorted) >= 2:
                xs = np.array([r[0] for r in line_rots_sorted])
                ys = np.array([r[1] for r in line_rots_sorted])
                deg = min(max_poly_degree, len(line_comps_sorted) - 1)
                try:
                    coeffs = np.polyfit(xs, ys, deg=deg)
                except np.linalg.LinAlgError:
                    coeffs = np.array([np.mean(ys)])
            else:
                coeffs = np.array([line_rots_sorted[0][1]])

            avg_scale = float(np.mean([c.scale for c in line_comps_sorted]))
            avg_orient = float(np.mean([c.orientation_deg for c in line_comps_sorted]))

            line = TextLine(
                id=line_id_counter,
                block_id=block.id,
                components=line_comps_sorted,
                poly_coeffs=coeffs,
                orientation_deg=avg_orient,
                scale=avg_scale,
            )
            lines.append(line)
            line_id_counter += 1

    # Sort lines by baseline: evaluate polynomial at block center x
    block_cx = float(np.mean([c.center[0] for c in block.components])) if block.components else 0
    rot_cx, _ = rotate(block_cx, cy_mean)

    def line_baseline_y(line: TextLine) -> float:
        if line.poly_coeffs is not None and len(line.poly_coeffs) > 0:
            return float(np.polyval(line.poly_coeffs, rot_cx))
        if line.components:
            return line.center[1]
        return 0.0

    lines.sort(key=line_baseline_y)
    for i, ln in enumerate(lines):
        ln.id = i
    return lines


def _check_rmse(rots: List[Tuple[float, float]], max_deg: int,
                s_bar: float, factor: float) -> bool:
    """Check if fitting a polynomial to given points satisfies RMSE threshold."""
    if len(rots) < 2:
        return True
    xs = np.array([r[0] for r in rots])
    ys = np.array([r[1] for r in rots])
    deg = min(max_deg, len(rots) - 1)
    try:
        coeffs = np.polyfit(xs, ys, deg=deg)
        y_pred = np.polyval(coeffs, xs)
        rmse = float(np.sqrt(np.mean((ys - y_pred) ** 2)))
        return rmse <= s_bar * factor + 1e-3
    except np.linalg.LinAlgError:
        return False


def build_layout(
    components: List[TextComponent],
    epsilon: float = 2.0,
    max_poly_degree: int = 4,
    deviation_scale_factor: float = 0.25,
) -> List[TextBlock]:
    """
    Full layout analysis: block grouping → line grouping within each block.
    Returns list of TextBlock objects, each with .lines populated.
    """
    blocks = group_components_into_blocks(components, epsilon=epsilon)
    for block in blocks:
        block.lines = group_block_into_lines(
            block,
            max_poly_degree=max_poly_degree,
            deviation_scale_factor=deviation_scale_factor,
        )
    return blocks
