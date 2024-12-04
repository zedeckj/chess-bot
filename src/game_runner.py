import torch
from torch import nn
import chess
import sys
sys.path.append("./")
from src.agent import AbstractChessAgent, NeuralAgent, RandomAgent, NaiveChessAgent
import sys


class GameRunner(nn.Module):

    """
    This class enables a model to be used in games with comparison models
    """

    def __init__(self, 
                 player1 : AbstractChessAgent, 
                 player2 : AbstractChessAgent):
        self.player1 = player1
        self.player2 = player2

    def run(self, player1_white : bool, initial_board : chess.Board = chess.Board()):
        if player1_white:
            white_player = self.player1
            black_player = self.player2
        else:
            white_player = self.player2
            black_player = self.player1
        board = initial_board
        outcome = None
        while True:
            move = white_player.get_move(board)
            #if not move:
            if (outcome := board.outcome()) or not move:
                break
            board.push(move)
            self.display_board(board, player1_white)
            move = black_player.get_move(board)
            #if not move:
            if (outcome := board.outcome()) or not move:
                break
            board.push(move)    
            self.display_board(board, player1_white)     

        return outcome
        

    def display_board(self, board : chess.Board, player1_white : bool):
        for _ in range(13):
            print("\033[F\033[K", end="") 
        if player1_white:
            info = f"{self.player1.get_name().upper()} vs {self.player2.get_name().casefold()}"
        else:
            info = f"{self.player1.get_name().casefold()} vs {self.player2.get_name().upper()}"
        print(f"{info}\n{board}\n\n")



if __name__ == "__main__":
    game_runner = GameRunner(NaiveChessAgent(look_ahead=2), RandomAgent(look_ahead=1))
    print(game_runner.run(True))