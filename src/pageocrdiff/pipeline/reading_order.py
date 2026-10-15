"""
Reading order determination across blocks and within lines.
Paper 1 Section V: intra-line order by rotated x, inter-line by baseline height,
inter-block by top-to-bottom / left-to-right column ordering (XY-cut policy).
"""
import numpy as np
from typing import List, Tuple
from .grouping import TextBlock, TextLine


def determine_reading_order(blocks: List[TextBlock],
                            page_width: int = 0,
                            language_rtl: bool = False) -> List[TextBlock]:
    """
    Assigns reading order indices to blocks, lines, and components.
    Policy: multi-column XY-cut followed by top-to-bottom / left-to-right.

    Paper 1 inter-line ordering: sort lines within a block by their polynomial
    baseline evaluated at the block's average x-coordinate f_i(x_avg).

    Returns the same block list with .lines sorted in reading order.
    """
    if not blocks:
        return blocks

    # XY-cut column detection: split blocks into columns by horizontal whitespace
    column_groups = _xy_cut_columns(blocks, page_width)

    global_block_order = 0
    ordered_blocks: List[TextBlock] = []
    for col_blocks in column_groups:
        # Within each column: top-to-bottom
        col_blocks.sort(key=lambda b: b.bbox[1])
        for block in col_blocks:
            block.id = global_block_order
            global_block_order += 1

            # Intra-block line ordering: by baseline at block center x
            bx, by, bw, bh = block.bbox
            x_avg = bx + bw / 2.0

            def line_key(line: TextLine) -> float:
                if line.poly_coeffs is not None and len(line.poly_coeffs) > 0:
                    return float(np.polyval(line.poly_coeffs, x_avg))
                return line.center[1]

            block.lines.sort(key=line_key)

            # Intra-line component ordering: left-to-right (or RTL if configured)
            for line in block.lines:
                if language_rtl:
                    line.components.sort(key=lambda c: -c.center[0])
                else:
                    line.components.sort(key=lambda c: c.center[0])

            ordered_blocks.append(block)

    return ordered_blocks


def _xy_cut_columns(blocks: List[TextBlock],
                    page_width: int) -> List[List[TextBlock]]:
    """
    Simple XY-cut column detection: project block bboxes onto x-axis,
    find gaps, split into column groups, then sort each column top-to-bottom.
    """
    if not blocks or page_width <= 0:
        return [sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1]))]

    # Build x-projection occupancy
    occupancy = np.zeros(page_width, dtype=np.int32)
    for b in blocks:
        bx, by, bw, bh = b.bbox
        x1 = max(0, bx)
        x2 = min(page_width, bx + bw)
        if x2 > x1:
            occupancy[x1:x2] += 1

    # Find gaps (zero occupancy) wider than a threshold
    gap_threshold = max(10, page_width // 20)
    in_gap = False
    gap_start = 0
    column_splits = []
    for x in range(page_width):
        if occupancy[x] == 0:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                gap_width = x - gap_start
                if gap_width >= gap_threshold:
                    column_splits.append((gap_start + gap_start + gap_width) // 2)
                in_gap = False

    if not column_splits:
        return [sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))]

    # Assign blocks to columns
    boundaries = [0] + column_splits + [page_width + 1]
    n_cols = len(boundaries) - 1
    columns: List[List[TextBlock]] = [[] for _ in range(n_cols)]
    for b in blocks:
        bx, _, bw, _ = b.bbox
        block_cx = bx + bw / 2
        assigned = n_cols - 1
        for col_i in range(n_cols):
            if boundaries[col_i] <= block_cx < boundaries[col_i + 1]:
                assigned = col_i
                break
        columns[assigned].append(b)

    return [col for col in columns if col]


def extract_ordered_text(blocks: List[TextBlock]) -> str:
    """Reconstruct full document text in reading order."""
    lines_out = []
    for block in blocks:
        for line in block.lines:
            if line.text:
                lines_out.append(line.text)
            else:
                # Reconstruct from component metadata if available
                words = []
                for comp in line.components:
                    pass  # Text assigned later by recognizer
                if words:
                    lines_out.append(" ".join(words))
    return "\n".join(lines_out)
