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

def show_random():
    autoencoder256 = Autoencoder256()
    autoencoder128 = Autoencoder128()
    board = random_board()
    encoded128 = autoencoder128.encodeFromBoard(board)
    encoded256 = autoencoder256.encodeFromBoard(board)
    decoded_board128 = autoencoder128.decodeToBoard(encoded128)
    decoded_board256 = autoencoder256.decodeToBoard(encoded256)
    print(f"INPUT FEN: {board.fen(en_passant= 'fen')}")
    BoardDisplayer.show(board)
    print(f"ENCODED TENSOR 128: \n{ppstr_list(encoded128.tolist()[0])}\n")
    print(f"ENCODED TENSOR 256: \n{ppstr_list(encoded256.tolist()[0])}\n")
    print(f"DECODED 128 FEN: {decoded_board128.fen(en_passant= 'fen')}")
    BoardDisplayer.show(decoded_board128)
    print(f"DECODED 256 FEN: {decoded_board256.fen(en_passant= 'fen')}")
    BoardDisplayer.show(decoded_board256)

def main():
    show_random()

if __name__ == "__main__":
    main()