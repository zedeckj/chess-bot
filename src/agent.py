from abc import ABC, abstractmethod
from re import I
from typing import Callable, Optional

import chess
from numpy import dtype
from sympy import beta, evaluate
from math import inf
import torch 
from torch import nn
import numpy as np
import random
import sys
sys.path.append("./")
from src.chess_models import ProductionAutoencoder, ProductionEvaluator


INF = 1e6
            

class AbstractChessAgent(ABC):

    def __init__(self, look_ahead):
        self.look_ahead = look_ahead
        self.cache = {}
        self.softmax_tn = nn.Softmax(dim = 0)
        self.softmin_tn = nn.Softmin(dim = 0)

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, board : chess.Board) -> float:
        """
        A higher score should indicate a better evaluation for White, and a lower score a better evaluation for Black
        """
        ...


    def _alpha_cutoff(self, evaluation : float, alpha : float, beta : float):
        """
        Should be used for mini function
        """
        return evaluation < alpha
    
    def _beta_cutfoff(self, evaluation : float, alpha : float, beta : float):
        """
        Should be used for maxi function
        """
        return evaluation > beta
    
    def _new_alpha(self, evaluation : float, alpha : float, beta : float) -> tuple[float, float]:
        """
        Should be used for maxi function
        """
        return max(evaluation,alpha), beta
    

    def _new_beta(self, evaluation : float, alpha : float, beta : float) -> tuple[float, float]:
        """
        Should be used for mini function
        """
        return alpha, min(evaluation, beta)



    def recursive_search(self,  board : chess.Board, 
                         depth : int, cutoff_func : Callable[[float,float,float], bool],
                         new_alpha_beta_func :  Callable[[float,float,float], tuple[float, float]],
                         other_search_func : Callable[[chess.Board, int, float, float], float],
                         selector : Callable[[float, float], float],
                         initial_value : float,
                         alpha : float = -INF, beta : float = INF) -> float:
        if depth == 0:
            return self.evaluate(board)
        value = initial_value
        moves = list(board.legal_moves)
        for move in moves:
            board.push(move)
            value = selector(other_search_func(board, depth - 1, alpha, beta), value)
            board.pop()
            if cutoff_func(value, alpha, beta):
                break
            alpha, beta = new_alpha_beta_func(value, alpha, beta) 
        return value
    


    def maxi(self,  board : chess.Board, depth : int, alpha : float = -INF, beta : float = INF):
        """
        Should be called to evaluate boards where the turn is white's 
        """
        assert(board.turn == chess.WHITE)
        return self.recursive_search(board, depth, self._beta_cutfoff, self._new_alpha, self.mini, max, -INF, alpha, beta)
    
    def mini(self,  board : chess.Board, depth : int, alpha : float = -INF, beta : float = INF):
        """
        Should be called to evaluate boards where the turn is blacks's 
        """
        assert(board.turn == chess.BLACK)
        return self.recursive_search(board, depth, self._alpha_cutoff, self._new_beta, self.maxi, min, INF, alpha, beta)
    

    def softmax(self, evals : list[float]):
        probs = self.softmax_tn(torch.tensor(evals, dtype = torch.float64)).tolist()
        i = random.choices([i for i in range(len(evals))], probs, k = 1)[0]
        return i

    def softmin(self, evals : list[float]):
        probs = self.softmin_tn(torch.tensor(evals, dtype = torch.float64)).tolist()
        i = random.choices([i for i in range(len(evals))], probs, k = 1)[0]
        return i

    def firmmax(self, evals : list[float]):
        """
        Like the softmax function, but cubes each term first. The effect of this is a higher probability of selecting higher
        evaluations
        """
        tensor = torch.tensor(evals, dtype = torch.float64)
        cubed = torch.pow(tensor, 3)
        probs = self.softmax_tn(cubed).tolist()
        i = random.choices([i for i in range(len(evals))], probs, k = 1)[0]
        return i

    def firmmin(self, evals : list[float]):
        """
        Like the softmin function, but squares each term first. The effect of this is a higher probability of selecting lower
        evaluations
        """
        tensor = torch.tensor(evals, dtype = torch.float64)
        cubed = torch.pow(tensor, 3)
        probs = self.softmin_tn(cubed).tolist()
        i = random.choices([i for i in range(len(evals))], probs, k = 1)[0]
        return i

    

    def move_selector(self, board : chess.Board, moves : list[chess.Move], 
                      selection_function : Callable[[list[float]], int],
                      search_function : Callable[[chess.Board, int], float]) -> chess.Move:
        """
        Evaluates each position according to the given search function. Based on these evaluations, uses the given selection_function
        to select an index of the move to play.
        """
        evals = []
        for move in moves:
            board.push(move)
            evals.append(search_function(board, self.look_ahead - 1))
            board.pop()
        i = selection_function(evals)
        return moves[i]
    


    """
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
    """




    def print_evals(self, evals : list[float], moves : list[chess.Move]):
        for i in range(len(evals)):
            print(f"Move {moves[i]} Eval {evals[i]}")
 

    def get_move(self, board : chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if len(moves) == 0:
            return None
        elif board.turn == chess.WHITE:
            return self.move_selector(board, moves, np.argmax, self.mini)
        else:
            return self.move_selector(board, moves, np.argmin, self.maxi)
            




class NaiveChessAgent(AbstractChessAgent):


    def _material(self, board : chess.Board) -> float: 
        MATERIAL_SCALE = 100
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
        return MATERIAL_SCALE * material
    
    def _mobility(self, board : chess.Board) -> float:
        MOBILITY_SCALE = 10
        original_turn = board.turn
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        board.turn = original_turn
        return MOBILITY_SCALE * (white_mobility - black_mobility)


    def _double_pawns(self, board : chess.Board) -> float:
        board.


    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -INF
            return INF
        else:
            return self._calculate_material(board)
        
    def get_name(self) -> str:
        return "Naive Agent"
        

class RandomAgent(AbstractChessAgent):

    def evaluate(self, board: chess.Board) -> float:
        return random.random()
        

    def get_name(self) -> str:
        return "Random Agent"

class NeuralAgent(AbstractChessAgent):
    def __init__(self, look_ahead : int):
        super().__init__(look_ahead)
        self.evaluator = ProductionEvaluator()
        self.autoencoder = ProductionAutoencoder()

    def evaluate(self, board: chess.Board) -> float:
        return self.evaluator(self.autoencoder.encodeFromBoard(board)).item()

        
    def get_name(self) -> str:
        return f"Neural Agent"

