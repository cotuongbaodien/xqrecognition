"""
FEN notation generator for Xiangqi Recognition System.
Converts board state to FEN string representation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from .settings import (
    GRID_COLS,
    GRID_ROWS,
    CLASS_TO_FEN,
    STARTING_FEN,
)
from .board_detector import Grid
from .piece_detector import DetectedPiece


def render_fen_ascii(fen: str) -> str:
    """Render a FEN string as a 10x9 ASCII board for visual inspection."""
    rows = fen.split(' ')[0].split('/')
    lines = ['   a b c d e f g h i', '  +-+-+-+-+-+-+-+-+-+']
    for i, row in enumerate(rows):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend(['.'] * int(ch))
            else:
                cells.append(ch)
        cells = cells[:9] + ['.'] * (9 - len(cells))
        lines.append(f'{i} |' + '|'.join(cells) + '|')
    lines.append('  +-+-+-+-+-+-+-+-+-+')
    return '\n'.join(lines)


@dataclass
class BoardState:
    """Represents the state of a Xiangqi board."""
    board: List[List[Optional[str]]]  # 10 rows x 9 cols, None for empty
    pieces: List[Tuple[int, int, str]]  # (row, col, fen_symbol)

    def __post_init__(self):
        if len(self.board) != GRID_ROWS:
            raise ValueError(f"Board must have {GRID_ROWS} rows")
        for row in self.board:
            if len(row) != GRID_COLS:
                raise ValueError(f"Each row must have {GRID_COLS} columns")

    def get_piece(self, row: int, col: int) -> Optional[str]:
        """Get the piece at a given position."""
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[str]):
        """Set a piece at a given position."""
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            self.board[row][col] = piece

    def is_empty(self, row: int, col: int) -> bool:
        """Check if a position is empty."""
        return self.get_piece(row, col) is None

    def count_pieces(self) -> Dict[str, int]:
        """Count pieces by type."""
        counts = {}
        for row in self.board:
            for piece in row:
                if piece:
                    counts[piece] = counts.get(piece, 0) + 1
        return counts


class FENGenerator:
    """
    Generates FEN notation from board state.

    FEN Format for Xiangqi:
    - Start from row 0 (black side) to row 9 (red side)
    - Each row separated by "/"
    - Numbers indicate consecutive empty spaces
    - Uppercase letters for red pieces, lowercase for black
    """

    CLASS_TO_FEN = CLASS_TO_FEN

    def __init__(self):
        pass

    def map_pieces_to_grid(
        self,
        pieces: List[DetectedPiece],
        grid: Grid,
        max_distance_ratio: float = 0.6
    ) -> BoardState:
        """
        Map detected pieces to grid positions.

        Args:
            pieces: List of detected pieces.
            grid: The board grid.
            max_distance_ratio: Max distance to grid cell as ratio of cell size.
                Pieces further than this are discarded as false positives.

        Returns:
            BoardState object representing the board.
        """
        board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        # Track confidence per cell for collision resolution
        board_confidence = [[0.0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        piece_positions = []

        max_dist = max(grid.cell_width, grid.cell_height) * max_distance_ratio

        # Generals first, then confidence descending. The general is the one
        # piece a position cannot lose: on tilted photos perspective can snap
        # a general and its neighbour onto the same cell, and pure-confidence
        # order let a 0.91 advisor erase a 0.90 general → kingless FEN broke
        # the mobile app with "in check" on both sides (2026-06-12).
        sorted_pieces = sorted(
            pieces,
            key=lambda p: (p.fen_symbol not in ("K", "k"), -p.confidence),
        )

        for piece in sorted_pieces:
            cx, cy = piece.center
            row, col = grid.get_nearest_cell(cx, cy)

            if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
                continue

            # Distance check: discard pieces too far from any grid cell
            if max_dist > 0:
                px, py = grid.points[row, col]
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist > max_dist:
                    continue

            # Placement order = priority order, so an occupied cell always
            # holds a piece that outranks this one. Instead of dropping the
            # loser (it IS on the board, just perspective-squeezed), shift it
            # to the nearest free cell still within snapping distance.
            if board[row][col] is not None:
                alt = self._nearest_free_cell(grid, cx, cy, board, max_dist)
                if alt is None:
                    continue
                row, col = alt

            board[row][col] = piece.fen_symbol
            board_confidence[row][col] = piece.confidence
            piece_positions.append((row, col, piece.fen_symbol))

        return BoardState(board=board, pieces=piece_positions)

    @staticmethod
    def _nearest_free_cell(grid, cx, cy, board, max_dist):
        """Nearest unoccupied grid cell within max_dist of (cx, cy), or None."""
        pts = grid.points.reshape(-1, 2)
        dists = np.sqrt(((pts - np.array([cx, cy])) ** 2).sum(axis=1))
        for idx in np.argsort(dists):
            if max_dist > 0 and dists[idx] > max_dist:
                return None
            row, col = divmod(int(idx), GRID_COLS)
            if board[row][col] is None:
                return row, col
        return None

    def map_pieces_to_grid_by_interpolation(
        self,
        pieces: List[DetectedPiece],
        image_width: int,
        image_height: int,
        margin_ratio: float = 0.05
    ) -> BoardState:
        """
        Map pieces to grid using interpolation based on detected piece positions.
        Fallback method when board detection fails.

        Args:
            pieces: List of detected pieces.
            image_width: Width of the image.
            image_height: Height of the image.
            margin_ratio: Fallback margin ratio if piece-based estimation fails.

        Returns:
            BoardState object.
        """
        if not pieces:
            board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
            return BoardState(board=board, pieces=[])

        centers_x = [p.center[0] for p in pieces]
        centers_y = [p.center[1] for p in pieces]

        min_x, max_x = min(centers_x), max(centers_x)
        min_y, max_y = min(centers_y), max(centers_y)

        cell_width = (max_x - min_x) / (GRID_COLS - 1) if max_x > min_x else 1
        cell_height = (max_y - min_y) / (GRID_ROWS - 1) if max_y > min_y else 1

        board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        piece_positions = []

        sorted_pieces = sorted(pieces, key=lambda p: p.confidence, reverse=True)

        for piece in sorted_pieces:
            cx, cy = piece.center

            col = round((cx - min_x) / cell_width)
            row = round((cy - min_y) / cell_height)

            col = max(0, min(GRID_COLS - 1, col))
            row = max(0, min(GRID_ROWS - 1, row))

            if board[row][col] is None:
                board[row][col] = piece.fen_symbol
                piece_positions.append((row, col, piece.fen_symbol))

        return BoardState(board=board, pieces=piece_positions)

    def generate_fen(self, board_state: BoardState, turn: str = "w") -> str:
        """
        Generate FEN string from board state.

        Args:
            board_state: BoardState object.
            turn: Whose turn to move ('w' for red/white, 'b' for black).

        Returns:
            FEN string with turn indicator.
        """
        fen_rows = []

        for row in range(GRID_ROWS):
            fen_row = ""
            empty_count = 0

            for col in range(GRID_COLS):
                piece = board_state.board[row][col]

                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece

            if empty_count > 0:
                fen_row += str(empty_count)

            fen_rows.append(fen_row)

        fen_board = "/".join(fen_rows)
        return f"{fen_board} {turn}"

    def parse_fen(self, fen: str) -> Tuple[BoardState, str]:
        """
        Parse a FEN string into a BoardState.

        Args:
            fen: FEN string (with or without turn indicator).

        Returns:
            Tuple of (BoardState object, turn indicator).
        """
        board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        piece_positions = []

        parts = fen.strip().split(" ")
        fen_board = parts[0]
        turn = parts[1] if len(parts) > 1 else "w"

        rows = fen_board.split("/")
        if len(rows) != GRID_ROWS:
            raise ValueError(f"FEN must have {GRID_ROWS} rows, got {len(rows)}")

        for row_idx, row_str in enumerate(rows):
            col_idx = 0
            for char in row_str:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    if col_idx < GRID_COLS:
                        board[row_idx][col_idx] = char
                        piece_positions.append((row_idx, col_idx, char))
                    col_idx += 1

        return BoardState(board=board, pieces=piece_positions), turn

    def validate_fen(self, fen: str) -> Tuple[bool, List[str]]:
        """
        Validate a FEN string for Xiangqi rules.

        Args:
            fen: FEN string to validate.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []

        try:
            board_state, turn = self.parse_fen(fen)
        except Exception as e:
            return False, [str(e)]

        piece_counts = board_state.count_pieces()

        max_pieces = {
            'k': 1, 'K': 1,  # Generals
            'a': 2, 'A': 2,  # Advisors
            'b': 2, 'B': 2,  # Elephants
            'n': 2, 'N': 2,  # Knights
            'r': 2, 'R': 2,  # Rooks
            'c': 2, 'C': 2,  # Cannons
            'p': 5, 'P': 5,  # Pawns
        }

        for piece, max_count in max_pieces.items():
            count = piece_counts.get(piece, 0)
            if count > max_count:
                errors.append(f"Too many {piece}: {count} > {max_count}")

        if piece_counts.get('k', 0) != 1:
            errors.append("Missing black general (k)")
        if piece_counts.get('K', 0) != 1:
            errors.append("Missing red general (K)")

        return len(errors) == 0, errors

    def compare_fen(self, fen1: str, fen2: str) -> Dict:
        """
        Compare two FEN strings and report differences.

        Args:
            fen1: First FEN string.
            fen2: Second FEN string.

        Returns:
            Dictionary with comparison results.
        """
        board1, _ = self.parse_fen(fen1)
        board2, _ = self.parse_fen(fen2)

        differences = []
        matching = 0
        total = 0

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                p1 = board1.board[row][col]
                p2 = board2.board[row][col]

                if p1 or p2:
                    total += 1
                    if p1 == p2:
                        matching += 1
                    else:
                        differences.append({
                            "position": (row, col),
                            "fen1": p1,
                            "fen2": p2,
                        })

        accuracy = matching / total if total > 0 else 1.0

        return {
            "fen1": fen1,
            "fen2": fen2,
            "total_pieces": total,
            "matching": matching,
            "accuracy": accuracy,
            "differences": differences,
        }

    @staticmethod
    def get_starting_fen() -> str:
        """Get the standard starting position FEN."""
        return STARTING_FEN

    def detect_board_orientation(self, board_state: BoardState) -> str:
        """
        Detect 180° vertical orientation. Standard Xiangqi: black at top
        (rows 0-4), red at bottom (rows 5-9).

        Uses a MAJORITY VOTE of three independent signals so a single
        misread piece — especially a misclassified general, which is the
        exact failure mode that used to flip the whole board — cannot
        decide orientation on its own:
          1. General (K/k) rows.
          2. Colour centroid of ALL pieces (red should be the lower half).
          3. Palace pieces (K/A vs k/a) sitting in their expected half.

        Returns 'standard' or 'flipped'.
        """
        red_gen = black_gen = None
        red_rows, black_rows = [], []        # every piece, by colour
        red_palace, black_palace = [], []    # generals + advisors only

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                p = board_state.board[row][col]
                if not p:
                    continue
                if p == 'K':
                    red_gen = row
                elif p == 'k':
                    black_gen = row
                (red_rows if p.isupper() else black_rows).append(row)
                if p in ('K', 'A'):
                    red_palace.append(row)
                elif p in ('k', 'a'):
                    black_palace.append(row)

        votes = []

        # Signal 1: general positions
        if red_gen is not None and black_gen is not None:
            votes.append('standard' if red_gen > black_gen else 'flipped')
        elif red_gen is not None:
            votes.append('standard' if red_gen >= 5 else 'flipped')
        elif black_gen is not None:
            votes.append('standard' if black_gen < 5 else 'flipped')

        # Signal 2: colour centroid of all pieces (robust to a few misreads)
        if red_rows and black_rows:
            red_avg = sum(red_rows) / len(red_rows)
            black_avg = sum(black_rows) / len(black_rows)
            votes.append('standard' if red_avg > black_avg else 'flipped')

        # Signal 3: how many palace pieces fall in their expected half
        if red_palace or black_palace:
            std_ok = sum(r >= 5 for r in red_palace) + sum(r < 5 for r in black_palace)
            flp_ok = sum(r < 5 for r in red_palace) + sum(r >= 5 for r in black_palace)
            if std_ok != flp_ok:
                votes.append('standard' if std_ok > flp_ok else 'flipped')

        if not votes:
            return 'standard'
        return 'flipped' if votes.count('flipped') > votes.count('standard') else 'standard'

    def flip_board(self, board_state: BoardState) -> BoardState:
        """
        Flip the board 180 degrees.
        Used to normalize board orientation.
        """
        new_board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        new_positions = []

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece:
                    new_row = GRID_ROWS - 1 - row
                    new_col = GRID_COLS - 1 - col
                    new_board[new_row][new_col] = piece
                    new_positions.append((new_row, new_col, piece))

        return BoardState(board=new_board, pieces=new_positions)

    def mirror_board_horizontal(self, board_state: BoardState) -> BoardState:
        """Mirror the board left-right (reverse columns)."""
        new_board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        new_positions = []

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece:
                    new_col = GRID_COLS - 1 - col
                    new_board[row][new_col] = piece
                    new_positions.append((row, new_col, piece))

        return BoardState(board=new_board, pieces=new_positions)

    def detect_needs_mirror(self, image: np.ndarray, bbox) -> bool:
        """
        Detect if the board image needs horizontal mirroring.

        Uses multiple image features: gradient direction in the river area,
        full board gradient, and text density in corners. These features
        are combined into a single score.

        Args:
            image: Original image (BGR).
            bbox: Board bounding box (x1, y1, x2, y2).

        Returns:
            True if the board needs horizontal mirroring.
        """
        if bbox is None:
            return False

        x1, y1, x2, y2 = [int(v) for v in bbox]
        board = image[y1:y2, x1:x2]
        bh, bw = board.shape[:2]

        if bh < 20 or bw < 20:
            return False

        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

        # Feature 1: Full board horizontal gradient moment
        sobelx_full = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        moment_full = np.mean(sobelx_full)

        # Feature 2: River area gradient moment
        ry1 = int(bh * 0.43)
        ry2 = int(bh * 0.57)
        sobelx_river = cv2.Sobel(gray[ry1:ry2], cv2.CV_64F, 1, 0, ksize=3)
        moment_river = np.mean(sobelx_river)

        # Feature 3: Top edge text density (left vs right)
        tl = gray[:int(bh * 0.08), :int(bw * 0.3)]
        tr = gray[:int(bh * 0.08), int(bw * 0.7):]
        edges_tl = np.sum(cv2.Canny(tl, 50, 150) > 0)
        edges_tr = np.sum(cv2.Canny(tr, 50, 150) > 0)
        text_top = (edges_tl - edges_tr) / max(edges_tl + edges_tr, 1)

        # Feature 4: Bottom edge text density (left vs right)
        bl = gray[int(bh * 0.92):, :int(bw * 0.3)]
        br = gray[int(bh * 0.92):, int(bw * 0.7):]
        edges_bl = np.sum(cv2.Canny(bl, 50, 150) > 0)
        edges_br = np.sum(cv2.Canny(br, 50, 150) > 0)
        text_bot = (edges_bl - edges_br) / max(edges_bl + edges_br, 1)

        # Combined score: positive = needs mirror
        score = moment_full + moment_river * 0.5 + text_top * 2 + text_bot * 2

        return score > 0.8

    def normalize_board_orientation(
        self,
        board_state: BoardState,
        image: np.ndarray = None,
        bbox=None
    ) -> BoardState:
        """
        Normalize board to standard orientation.
        Handles both vertical flip (red/black swap) and horizontal mirror.

        Args:
            board_state: Board state to normalize.
            image: Original image for mirror detection (optional).
            bbox: Board bounding box for mirror detection (optional).
        """
        # Step 1: Fix vertical orientation (red at bottom, black at top)
        orientation = self.detect_board_orientation(board_state)
        if orientation == 'flipped':
            board_state = self.flip_board(board_state)

        # Step 2: Check if horizontal mirror is needed using image analysis
        if image is not None and bbox is not None:
            if self.detect_needs_mirror(image, bbox):
                board_state = self.mirror_board_horizontal(board_state)

        return board_state

    def normalize_mirror(
        self,
        board_state: BoardState,
        image: np.ndarray = None,
        bbox=None
    ) -> BoardState:
        """Check and apply horizontal mirror only (vertical already handled)."""
        if image is not None and bbox is not None:
            if self.detect_needs_mirror(image, bbox):
                board_state = self.mirror_board_horizontal(board_state)
        return board_state

    def board_to_ascii(self, board_state: BoardState) -> str:
        """
        Convert board state to ASCII representation.

        Args:
            board_state: BoardState object.

        Returns:
            ASCII string representation of the board.
        """
        lines = []
        lines.append("  0 1 2 3 4 5 6 7 8")
        lines.append("  -----------------")

        for row in range(GRID_ROWS):
            row_str = f"{row}|"
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                row_str += (piece if piece else ".") + " "
            lines.append(row_str)

        lines.append("  -----------------")

        return "\n".join(lines)
