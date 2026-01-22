"""
FEN notation generator for Xiangqi Recognition System.
Converts board state to FEN string representation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    GRID_COLS,
    GRID_ROWS,
    CLASS_TO_FEN,
    STARTING_FEN,
)
from .board_detector import Grid
from .piece_detector import DetectedPiece


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
        grid: Grid
    ) -> BoardState:
        """
        Map detected pieces to grid positions.

        Args:
            pieces: List of detected pieces.
            grid: The board grid.

        Returns:
            BoardState object representing the board.
        """
        board = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        piece_positions = []

        for piece in pieces:
            cx, cy = piece.center
            row, col = grid.get_nearest_cell(cx, cy)

            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                if board[row][col] is None:
                    board[row][col] = piece.fen_symbol
                    piece_positions.append((row, col, piece.fen_symbol))

        return BoardState(board=board, pieces=piece_positions)

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
        Detect board orientation based on piece positions.
        Standard Xiangqi: Black at top (row 0-4), Red at bottom (row 5-9).

        Returns:
            'standard' if black is at top, 'flipped' if red is at top.
        """
        red_y_sum = 0
        red_count = 0
        black_y_sum = 0
        black_count = 0

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece:
                    if piece.isupper():  # Red piece
                        red_y_sum += row
                        red_count += 1
                    else:  # Black piece
                        black_y_sum += row
                        black_count += 1

        if red_count == 0 or black_count == 0:
            return 'standard'

        red_avg_row = red_y_sum / red_count
        black_avg_row = black_y_sum / black_count

        if red_avg_row < black_avg_row:
            return 'flipped'
        return 'standard'

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

    def normalize_board_orientation(self, board_state: BoardState) -> BoardState:
        """
        Normalize board to standard orientation (black at top, red at bottom).
        """
        orientation = self.detect_board_orientation(board_state)
        if orientation == 'flipped':
            return self.flip_board(board_state)
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
