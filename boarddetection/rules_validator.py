"""
Xiangqi game rules validator for post-processing detection results.
Validates and corrects board state based on Chinese Chess rules.
"""

from typing import List, Optional, Tuple, Dict
from .fen_generator import BoardState
from .settings import GRID_COLS, GRID_ROWS


# Maximum number of each piece type
MAX_PIECE_COUNTS = {
    'k': 1, 'K': 1,  # Generals
    'a': 2, 'A': 2,  # Advisors
    'b': 2, 'B': 2,  # Elephants
    'n': 2, 'N': 2,  # Knights
    'r': 2, 'R': 2,  # Rooks
    'c': 2, 'C': 2,  # Cannons
    'p': 5, 'P': 5,  # Pawns
}

# Valid positions for constrained pieces
# Standard orientation: black rows 0-4, red rows 5-9
VALID_POSITIONS = {
    # Red General: palace rows 7-9, cols 3-5
    'K': {(r, c) for r in range(7, 10) for c in range(3, 6)},
    # Black General: palace rows 0-2, cols 3-5
    'k': {(r, c) for r in range(0, 3) for c in range(3, 6)},
    # Red Advisors: 5 positions in red palace
    'A': {(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)},
    # Black Advisors: 5 positions in black palace
    'a': {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)},
    # Red Elephants: 7 positions in red half
    'B': {(5, 2), (5, 6), (7, 0), (7, 2), (7, 4), (7, 6), (7, 8),
          (9, 0), (9, 2), (9, 4), (9, 6), (9, 8)},
    # Black Elephants: 7 positions in black half
    'b': {(0, 0), (0, 2), (0, 4), (0, 6), (0, 8), (2, 0), (2, 2),
          (2, 4), (2, 6), (2, 8), (4, 2), (4, 6)},
}

# Pawns can only be in certain rows
# Red pawns: rows 3-9 (rows 5-9 on own side, 3-4 after crossing river)
# Black pawns: rows 0-6 (rows 0-4 on own side, 5-6 after crossing river)
# But after crossing river, pawns can move sideways, so all cols are valid
# Before crossing: only original columns (0,2,4,6,8)
PAWN_CONSTRAINTS = {
    'P': {'min_row': 3, 'max_row': 9, 'home_rows': range(5, 10), 'home_cols': {0, 2, 4, 6, 8}},
    'p': {'min_row': 0, 'max_row': 6, 'home_rows': range(0, 5), 'home_cols': {0, 2, 4, 6, 8}},
}


class RulesValidator:
    """Validates and corrects board state using Xiangqi rules."""

    def validate_and_correct(
        self,
        board_state: BoardState,
        piece_confidences: Optional[Dict[Tuple[int, int], float]] = None,
        alternates: Optional[Dict[Tuple[int, int], List[Tuple[str, float]]]] = None,
    ) -> BoardState:
        """
        Validate board state and auto-correct violations.

        Args:
            board_state: Current board state.
            piece_confidences: Map of (row, col) -> confidence score.
            alternates: Map of (row, col) -> [(fen_symbol, confidence), ...]
                second-opinion classes from NMS-suppressed detections. When
                an excess piece must be removed, a legal alternate class at
                the same cell is substituted instead of emptying the cell —
                the detector clearly saw a piece there, it just disagreed
                with itself about the class.

        Returns:
            Corrected BoardState.
        """
        if piece_confidences is None:
            piece_confidences = {}

        board_state = self._fix_excess_pieces(
            board_state, piece_confidences, alternates or {})
        board_state = self._fix_invalid_positions(board_state, piece_confidences)

        return board_state

    def _count_on_board(self, board_state: BoardState, symbol: str) -> int:
        return sum(
            1
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
            if board_state.board[row][col] == symbol
        )

    def _is_legal_cell(self, symbol: str, row: int, col: int) -> bool:
        if symbol in VALID_POSITIONS:
            return (row, col) in VALID_POSITIONS[symbol]
        if symbol in PAWN_CONSTRAINTS:
            c = PAWN_CONSTRAINTS[symbol]
            return c['min_row'] <= row <= c['max_row']
        return True

    def _best_alternate(
        self,
        board_state: BoardState,
        cell: Tuple[int, int],
        dropped_symbol: str,
        alternates: Dict[Tuple[int, int], List[Tuple[str, float]]],
    ) -> Optional[str]:
        """Best legal substitute class for a removed excess piece."""
        row, col = cell
        candidates = sorted(
            alternates.get(cell, []), key=lambda a: a[1], reverse=True)
        for symbol, _conf in candidates:
            if symbol == dropped_symbol:
                continue
            if self._count_on_board(board_state, symbol) >= \
                    MAX_PIECE_COUNTS.get(symbol, 99):
                continue
            if not self._is_legal_cell(symbol, row, col):
                continue
            return symbol
        return None

    def _fix_excess_pieces(
        self,
        board_state: BoardState,
        confidences: Dict[Tuple[int, int], float],
        alternates: Optional[Dict[Tuple[int, int], List[Tuple[str, float]]]] = None,
    ) -> BoardState:
        """Remove excess pieces, keeping highest confidence ones. Where an
        NMS-suppressed detection offers a legal second-opinion class at the
        same cell, substitute it instead of leaving the cell empty."""
        alternates = alternates or {}
        counts: Dict[str, List[Tuple[int, int, float]]] = {}

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece:
                    conf = confidences.get((row, col), 0.5)
                    if piece not in counts:
                        counts[piece] = []
                    counts[piece].append((row, col, conf))

        for piece_type, positions in counts.items():
            max_count = MAX_PIECE_COUNTS.get(piece_type, 99)
            if len(positions) > max_count:
                # Sort by confidence descending, keep top max_count
                positions.sort(key=lambda x: x[2], reverse=True)
                for row, col, _ in positions[max_count:]:
                    board_state.board[row][col] = None
                    substitute = self._best_alternate(
                        board_state, (row, col), piece_type, alternates)
                    if substitute:
                        board_state.board[row][col] = substitute

        # Rebuild pieces list
        board_state.pieces = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece:
                    board_state.pieces.append((row, col, piece))

        return board_state

    def _fix_invalid_positions(
        self,
        board_state: BoardState,
        confidences: Dict[Tuple[int, int], float]
    ) -> BoardState:
        """Remove pieces at invalid positions (low confidence only)."""
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece and piece in VALID_POSITIONS:
                    if (row, col) not in VALID_POSITIONS[piece]:
                        conf = confidences.get((row, col), 0.5)
                        # Only remove if confidence is low - high confidence
                        # detections might indicate unusual but real positions
                        # (or board orientation issues)
                        if conf < 0.7:
                            board_state.board[row][col] = None

        # Check pawn constraints
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece in PAWN_CONSTRAINTS:
                    constraint = PAWN_CONSTRAINTS[piece]
                    if not (constraint['min_row'] <= row <= constraint['max_row']):
                        conf = confidences.get((row, col), 0.5)
                        if conf < 0.7:
                            board_state.board[row][col] = None

        # Rebuild pieces list
        board_state.pieces = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                piece = board_state.board[row][col]
                if piece:
                    board_state.pieces.append((row, col, piece))

        return board_state
