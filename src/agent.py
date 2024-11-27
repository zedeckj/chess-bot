from abc import ABC, abstractmethod
from re import I
from typing import Optional

import chess
from numpy import dtype
from sympy import evaluate
from math import inf
import torch 
from torch import nn

import random

INF = 100000
            

class AbstractChessAgent(ABC):

    def __init__(self, look_ahead = 1):
        self.look_ahead = look_ahead
        self.cache = {}
        self.softmax = nn.Softmax()
        self.softmin = nn.Softmin()

    @abstractmethod
    def evaluate(self, board : chess.Board) -> float:
        """
        A higher score should indicate a better evaluation for White, and a lower score a better evaluation for Black
        """
        ...


    def maxi(self,  board : chess.Board, depth : int, alpha : float = -INF, beta : float = INF) -> float:
        if depth == 0:
            return self.evaluate(board)
        value = -INF
        moves = list(board.legal_moves)
        for move in moves:
            board.push(move)
            value = max(self.mini(board, depth - 1, alpha, beta), value)
            board.pop()
            if value > beta:
                break
            alpha = max(value, alpha)
        return value
    
    def mini(self, board : chess.Board, depth : int, alpha : float = -INF, beta : float = INF) -> float:
        if depth == 0:
            return self.evaluate(board)
        value = INF #
        moves = list(board.legal_moves)
        for move in moves:
            board.push(move)
            value = min(self.maxi(board, depth - 1, alpha, beta), value)
            board.pop()
            if value < alpha:
                break
            beta = min(value, beta)
        return value

    def softmax_i(self, evals : list[float]):
        probs = self.softmax(torch.tensor(evals, dtype = torch.float64)).tolist()
        print(probs)
        i = random.choices([i for i in range(len(evals))], probs, k = 1)[0]
        return i

    def softmin_i(self, evals : list[float]):
        probs = self.softmin(torch.tensor(evals, dtype = torch.float64)).tolist()
        print(probs)
        i = random.choices([i for i in range(len(evals))], probs, k = 1)[0]
        return i

    def argmax_i(self, evals : list[float]) -> int:
        """
        max_eval = max(evals)
        possible_moves = []
        for i in range(len(evals)):
            if evals[i] == max_eval:
                possible_moves.append(i)
        """
        return int(torch.argmax(torch.tensor(evals, dtype = torch.float64)).item())

    def argmin_i(self, evals : list[float]) -> int:
        return int(torch.argmin(torch.tensor(evals, dtype = torch.float64)).item())


    def print_evals(self, evals : list[float], moves : list[chess.Move]):
        for i in range(len(evals)):
            print(f"Move {moves[i]} Eval {evals[i]}")

    def get_move(self, board : chess.Board) -> Optional[chess.Move]:
        evals = []
        moves = list(board.legal_moves)
        if len(moves) == 0:
            return None
        elif board.turn == chess.WHITE:
            for move in moves:
                board.push(move)
                evals.append(self.mini(board, self.look_ahead))
                board.pop()
            self.print_evals(evals, moves)
            i = self.argmax_i(evals)
            return moves[i]
        else:
            for move in moves:
                board.push(move)
                evals.append(self.maxi(board, self.look_ahead))
            self.print_evals(evals, moves)
            i = self.argmin_i(evals)
            return moves[i]
            

class NaiveChessAgent(AbstractChessAgent):


    def _calculate_material(self, board : chess.Board):
        material = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece != None:
                if piece.piece_type == chess.PAWN:
                    value = 1
                elif piece.piece_type == chess.BISHOP:
                    value = 3
                elif piece.piece_type == chess.KNIGHT:
                    value = 3
                elif piece.piece_type == chess.ROOK:
                    value = 5
                elif piece.piece_type == chess.QUEEN:
                    value = 9
                else:
                    value = 0 # kings are always present
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value
        return material


    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -INF
            print(f"CHECKMATE FOR BLACK\n {board.fen()}")
            return INF
        else:
            return self._calculate_material(board)