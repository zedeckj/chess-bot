from src.chess_models import Autoencoder128, Autoencoder256
from src.display_utils import BoardDisplayer
import chess
import random

def random_board():
    board = chess.Board()
    for i in range(random.randint(2,100)):
        moves = list(board.legal_moves)
        board.push(random.choice(moves))
    return board

def ppstr_list(l : list):
    return f"[{l[0]}" + f", ".join([str(v) for v in l[1:5]])+ "...]"

def show_random(use_256 : bool):
    autoencoder = Autoencoder256() if use_256 else Autoencoder128()
    print(f"autoencoder{256 if use_256 else 128} results")
    board = random_board()
    encoded = autoencoder.encodeFromBoard(board)
    decoded_board = autoencoder.decodeToBoard(encoded)
    print(f"INPUT FEN: {board.fen(en_passant= 'fen')}")
    BoardDisplayer.show(board)
    print(f"ENCODED TENSOR: \n{ppstr_list(encoded.tolist()[0])}\n")
    print(f"DECODED FEN: {decoded_board.fen(en_passant= 'fen')}")
    BoardDisplayer.show(decoded_board)

def main():
    for i in range(3):
        show_random(True)

if __name__ == "__main__":
    main()