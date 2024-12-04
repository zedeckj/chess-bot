import chess
import sys
sys.path.append("./")

from src.board_final import BoardGenerator


class BoardDisplayer:

    @staticmethod
    def show(board : chess.Board):
        piece_dic = {
            (chess.KING, chess.WHITE): "♔",
            (chess.QUEEN, chess.WHITE): "♕",
            (chess.ROOK, chess.WHITE): "♖",
            (chess.BISHOP, chess.WHITE): "♗",
            (chess.KNIGHT, chess.WHITE): "♘",
            (chess.PAWN, chess.WHITE): "♙",

            (chess.KING, chess.BLACK): "♚",
            (chess.QUEEN, chess.BLACK): "♛",
            (chess.ROOK, chess.BLACK): "♜",
            (chess.BISHOP, chess.BLACK): "♝",
            (chess.KNIGHT, chess.BLACK): "♞",
            (chess.PAWN, chess.BLACK): "♟",

        }
        light_background = True
        for i in range(8):
            for j in range(8):
                ansii = f"\x1b[48;5;{229 if light_background else 178}m"
                square = 8 * (7 - i) + j
                piece = board.piece_at(square)
                if piece == None:
                    print(f"{ansii} ", end = " ")
                else:
                    display = piece_dic[(piece.piece_type, piece.color)]
                    print(f"{ansii}{display}", end = " ")
                light_background = not light_background
            light_background = not light_background
            print("\x1b[0m")
        print("")

    @staticmethod
    def remove():
        print("\x1b[9A", end = "\r")

if __name__ == "__main__":
    import time
    for board in BoardGenerator(100):
        BoardDisplayer.show(board)
        time.sleep(0.1)
        BoardDisplayer.remove()