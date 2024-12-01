from ctypes.wintypes import SIZE
import random
from typing import Callable, Optional
import chess
import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional
import unittest
import itertools
import torch.share
import sys
sys.path.append("./")

from src.test_torch import TorchUtils

class TensorBoardUtilV4():
    """
    Utils for creating and using a non-normalized, non compressed, float32 tensor of 582 values representing a Chess board. 
    """

    SIZE = 904
    BINARY_RANGE = 902 # Number of values that are represented as 1 or 0 
    PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    PIECE_COMPONENT_SIZE = 13
    TURN_OFFSET = 64 * PIECE_COMPONENT_SIZE
    CASTLING_OFFSET = TURN_OFFSET + 1
    EN_PASSANT_OFFSET = CASTLING_OFFSET + 4
    CLOCK_OFFSET = EN_PASSANT_OFFSET + 65

    

    @staticmethod
    def indexOfPiece(piece : Optional[chess.Piece]):
        """
        Gets the index of a type of piece inside a piece tensor, which uses the following representation.
        [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
        isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
        isBlankSpace]
        """
        TYPES = len(TensorBoardUtilV4.PIECES)
        if piece == None:
            return TYPES * 2
        offset = 0 if piece.color == chess.WHITE else TYPES
        i = 0
        for compare_type in TensorBoardUtilV4.PIECES:
            if compare_type == piece.piece_type:
                return offset + i
            i += 1
        raise Exception("Invalid Piece Given")

    @staticmethod
    def _pieceToBoolList(piece : Optional[chess.Piece]) -> list[bool]:
        """
        Returns a len 13 list of booleans corresponding to information about the given piece, in the format of 
        [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
        isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
        isBlankSpace]
        """
        
        out_list = [False for _ in range(TensorBoardUtilV4.PIECE_COMPONENT_SIZE)]
        if piece == None:
            out_list[-1] = True
        else:
            offset = 6 if piece.color == chess.BLACK else 0
            for i, check in enumerate(TensorBoardUtilV4.PIECES):
                if check == piece.piece_type:
                    out_list[offset+i] = True
        return out_list
        
    @staticmethod
    def _enpassantToBoolList(en_passant_square : Optional[chess.Square]):
        if en_passant_square == None:
            return [True] + [False for _ in range(64)]
        return [False] + [i == en_passant_square for i in range(64)]
    
    
    @staticmethod 
    def _castlingRightsToBoolList(board : chess.Board) -> list[bool]:
        """
        Returns a list of boolean values corresponding to castling rights in a given board, in the format of
        [whiteKingside, whiteQueenside, blackKingside, blackQueenside]
        """
        out_list = [False for _ in range(4)]
        out_list[0] = board.has_kingside_castling_rights(chess.WHITE)
        out_list[1] = board.has_queenside_castling_rights(chess.WHITE)
        out_list[2] = board.has_kingside_castling_rights(chess.BLACK)
        out_list[3] = board.has_queenside_castling_rights(chess.BLACK)
        return out_list
    
    

    @staticmethod
    
    def tensorToPieceTensors(tensor : torch.Tensor) -> torch.Tensor:
        """
        Based on the given board tensor(s) of shape [N, TensorBoardUtilV4.SIZE], returns a tensor of shape [N, 64, 13], which represents 
        the type of piece at each square. Each piece tensor represents the following

        [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
        isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
        isBlankSpace]
        """
        out_tensor = tensor[..., 0 : TensorBoardUtilV4.TURN_OFFSET].reshape((tensor.shape[0], 64, TensorBoardUtilV4.PIECE_COMPONENT_SIZE))
        return out_tensor
    
    
    @staticmethod
    def tensorToCastlingRights(tensor : torch.Tensor) -> torch.Tensor:
        #assert(len(tensor.shape) == 2 and tensor.shape[1] == TensorBoardUtilV4.SIZE)
        out_tensor = tensor[...,TensorBoardUtilV4.CASTLING_OFFSET: TensorBoardUtilV4.CASTLING_OFFSET + 4]
        """
        assert(len(out_tensor.shape) == 3
               and out_tensor.shape[0] == tensor.shape[0]
               and out_tensor.shape[1] == 1 
                and out_tensor.shape[2] == 4)
        """
        return out_tensor

    @staticmethod
    def tensorToEnPassant(tensor: torch.Tensor) -> torch.Tensor:
        #assert(len(tensor.shape) == 2 and tensor.shape[1] == TensorBoardUtilV4.SIZE)
        out_tensor = tensor[...,TensorBoardUtilV4.EN_PASSANT_OFFSET: TensorBoardUtilV4.EN_PASSANT_OFFSET + 65]
        """
        assert(len(out_tensor.shape) == 3
               and out_tensor.shape[0] == tensor.shape[0]
               and out_tensor.shape[1] == 1 
                and out_tensor.shape[2] == 65)
        """
        return out_tensor

    @staticmethod
    def tensorToTimers(tensor: torch.Tensor) -> torch.Tensor:
        #assert(len(tensor.shape) == 2 and tensor.shape[1] == TensorBoardUtilV4.SIZE)
        out_tensor = tensor[...,TensorBoardUtilV4.CLOCK_OFFSET: TensorBoardUtilV4.CLOCK_OFFSET + 2]
        """
        assert(len(out_tensor.shape) == 3
               and out_tensor.shape[0] == tensor.shape[0]
               and out_tensor.shape[1] == 1 
                and out_tensor.shape[2] == 2)
        """
        return out_tensor

    @staticmethod
    def tensorToTurn(tensor: torch.Tensor) -> torch.Tensor:
        #assert(len(tensor.shape) == 2 and tensor.shape[1] == TensorBoardUtilV4.SIZE)
        out_tensor = tensor[...,TensorBoardUtilV4.TURN_OFFSET: TensorBoardUtilV4.TURN_OFFSET + 1]
        """
        assert(len(out_tensor.shape) == 3
               and out_tensor.shape[0] == tensor.shape[0]
               and out_tensor.shape[1] == 1 
                and out_tensor.shape[2] == 1)
        """
        return out_tensor


    @staticmethod
    def discretizeTensor(tensor: torch.Tensor) -> torch.Tensor:
        #assert(tensor.shape[-1] == TensorBoardUtilV4.SIZE)
        #if len(tensor.shape) == 3:
        #    tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], tensor.shape[2]))
        
        pieces_tensor = TensorBoardUtilV4.tensorToPieceTensors(tensor)
        turn_tensor = TensorBoardUtilV4.tensorToTurn(tensor)
        castling_tensor = TensorBoardUtilV4.tensorToCastlingRights(tensor)
        en_passant_tensor = TensorBoardUtilV4.tensorToEnPassant(tensor)
        clock_tensor = TensorBoardUtilV4.tensorToTimers(tensor)

        discrete_pieces = torch.reshape(TorchUtils.argmax_onehot(pieces_tensor), (pieces_tensor.shape[0], TensorBoardUtilV4.PIECE_COMPONENT_SIZE * 64))
        discrete_turn = torch.round(torch.sigmoid(turn_tensor))
        discrete_castling = torch.round(torch.sigmoid(castling_tensor))
        discrete_en_passant = TorchUtils.argmax_onehot(en_passant_tensor)
        discrete_clock = torch.round(clock_tensor)
        return torch.cat([discrete_pieces, discrete_turn, discrete_castling, discrete_en_passant, discrete_clock], dim = 1)

    @staticmethod
    def simulateContinous(tensor : torch.Tensor):
        ones = torch.ones((tensor.shape[0],TensorBoardUtilV4.BINARY_RANGE))
        rands = torch.rand((tensor.shape[0],TensorBoardUtilV4.BINARY_RANGE)) / 3 # values 0 - .3333
        bitrange = tensor[...,:TensorBoardUtilV4.BINARY_RANGE]


        to_substract = bitrange * rands # only substract from ones
        to_add = torch.logical_not(bitrange).float() * rands # only add to zeros

        bitrange = bitrange + to_add
        bitrange = bitrange - to_substract


        contrange = tensor[...,TensorBoardUtilV4.BINARY_RANGE:]
        contrange = contrange + 0.25
        contrange = contrange - torch.rand(2) / 2

        return torch.cat([bitrange, contrange], dim = 1)

    @staticmethod
    def fromBoard(board : chess.Board) -> torch.Tensor:
        total_bool_list = []
        for square in range(64):
            total_bool_list.extend(TensorBoardUtilV4._pieceToBoolList(board.piece_at(square)))
        total_bool_list.append(board.turn == chess.WHITE)
        total_bool_list.extend(TensorBoardUtilV4._castlingRightsToBoolList(board))
        total_bool_list.extend(TensorBoardUtilV4._enpassantToBoolList(board.ep_square))
        out_list = [float(b) for b in total_bool_list] + [board.halfmove_clock, board.fullmove_number]
        tensor_out = torch.tensor([out_list], dtype = torch.float32)
        assert(list(tensor_out.shape) == [1,TensorBoardUtilV4.SIZE])
        return tensor_out
    
    @staticmethod 
    def _castlingFENtoBoolList(castling_fen : str) -> list[bool]:
        out_list = [False for _ in range(4)]
        out_list[0] = "K" in castling_fen
        out_list[1] = "Q" in castling_fen
        out_list[2] = "k" in castling_fen
        out_list[3] = "q" in castling_fen
        return out_list


    @staticmethod
    def _square_to_index(square : str) -> Optional[int]:
        if square == "-":
            return None
        square = square.casefold()
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        return (int(square[1]) - 1) * 8 + files.index(square[0])

    @staticmethod
    def fromFEN(fen : str) -> torch.Tensor:
        pieces_str, turn, castling, en_passant, half_move, full_move = fen.split(" ")
        rows = pieces_str.split("/")
        pieces : list[Optional[chess.Piece]] = []
        rows.reverse()
        for row in rows:
            for c in row:
                l = c.casefold()
                type = None
                if l == "p":
                    type = chess.PAWN
                elif l == "n":
                    type = chess.KNIGHT
                elif l == "b":
                    type = chess.BISHOP
                elif l == "r":
                    type = chess.ROOK
                elif l == "q":
                    type = chess.QUEEN
                elif l == "k":
                    type = chess.KING
                else:
                    pieces.extend(None for _ in range(int(c)))
                    continue
                color = chess.BLACK if c == l else chess.WHITE
                pieces.append(chess.Piece(type, color))

        total_bool_list = []
        for piece in pieces:
            total_bool_list.extend(TensorBoardUtilV4._pieceToBoolList(piece))
        total_bool_list.append(turn == "w")
        total_bool_list.extend(TensorBoardUtilV4._castlingFENtoBoolList(castling))
        total_bool_list.extend(TensorBoardUtilV4._enpassantToBoolList(TensorBoardUtilV4._square_to_index(en_passant)))
        out_list = [float(b) for b in total_bool_list] + [int(half_move), int(full_move)]
        tensor_out = torch.tensor([out_list], dtype = torch.float32)
        return tensor_out

    @staticmethod
    def castlingTensorFromFEN(castling_fen : str) -> torch.Tensor:
        bool_list = TensorBoardUtilV4._castlingFENtoBoolList(castling_fen)
        return torch.tensor([[1 if b else 0 for b in bool_list]])
    
    @staticmethod
    def getCastlingFEN(board : chess.Board):
        return board.fen().split(" ")[2]

    @staticmethod
    def toBoard(tensor : torch.Tensor) -> chess.Board:
        assert(len(tensor.shape) == 2 and tensor.shape[1] == TensorBoardUtilV4.SIZE and tensor.shape[0] == 1)
        assert(torch.equal(tensor[...,:TensorBoardUtilV4.BINARY_RANGE].bool().float(),tensor[...,:TensorBoardUtilV4.BINARY_RANGE]))
        tensor = tensor.squeeze()
        out_board = chess.Board.empty()
        
        for square in range(64):
            i = square * TensorBoardUtilV4.PIECE_COMPONENT_SIZE
            # Each square is represented as 
            #  [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
            # isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
            # isBlankSpace]
            piece_type = None
            for j, color in [(0, chess.WHITE), (6, chess.BLACK)]:
                for k, piece_type in zip(list(range(6)), TensorBoardUtilV4.PIECES):
                    if tensor[i + j + k] == 1:
                        out_board.set_piece_at(square, chess.Piece(piece_type, color))
                        break
        out_board.turn = chess.WHITE if tensor[TensorBoardUtilV4.TURN_OFFSET] == 1 else chess.BLACK
        castling_fen = ""
        # [whiteKingside, whiteQueenside, blackKingside, blackQueenside]
        if tensor[TensorBoardUtilV4.CASTLING_OFFSET] == 1:
            castling_fen += "K"
        if tensor[TensorBoardUtilV4.CASTLING_OFFSET + 1] == 1:
            castling_fen += "Q"
        if tensor[TensorBoardUtilV4.CASTLING_OFFSET + 2] == 1:
            castling_fen += "k"
        if tensor[TensorBoardUtilV4.CASTLING_OFFSET + 3] == 1:
            castling_fen += "q"
        out_board.set_castling_fen(castling_fen)
        if not tensor[TensorBoardUtilV4.EN_PASSANT_OFFSET].bool().item():
            for i in range(64):
                if tensor[TensorBoardUtilV4.EN_PASSANT_OFFSET + i + 1] == 1:
                    out_board.ep_square = i
        out_board.halfmove_clock = int(tensor[TensorBoardUtilV4.CLOCK_OFFSET].item())
        out_board.fullmove_number = int(tensor[TensorBoardUtilV4.CLOCK_OFFSET + 1].item())
        return out_board




class BoardGenerator():
    

    def __init__(self, boards : int):
        self.board_count = boards

 

    def __iter__(self):
        past = []
        current_board = chess.Board()
        for _ in range(self.board_count):
            yield current_board
            past.append(current_board.fen(en_passant = "fen"))
            used = True
            while used:
                legal_moves = list(current_board.legal_moves)
                if len(legal_moves) == 0:
                    current_board = chess.Board()
                else:
                    current_board.push(random.choice(legal_moves))
                used = current_board.fen(en_passant="fen") in past

                

class TestTensorBoard(unittest.TestCase):

    
    def test_indexFromPiece(self):
        """
        Should follow the given:
        [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
        isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
        isBlankSpace]
        """
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.WHITE)), 0)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KNIGHT, chess.WHITE)), 1)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.BISHOP, chess.WHITE)), 2)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.ROOK, chess.WHITE)), 3)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.WHITE)), 4)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KING, chess.WHITE)), 5)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.BLACK)), 6)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KNIGHT, chess.BLACK)), 7)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.BISHOP, chess.BLACK)), 8)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.ROOK, chess.BLACK)), 9)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.BLACK)), 10)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KING, chess.BLACK)), 11)
        self.assertEqual(TensorBoardUtilV4.indexOfPiece(None), 12)
        


    def test_from_fen(self):
        for board in BoardGenerator(1000):
            tensor1 = TensorBoardUtilV4.fromBoard(board)
            fen = board.fen(en_passant = "fen")
            tensor2 = TensorBoardUtilV4.fromFEN(fen)
            with self.subTest(fen):
                self.assertEqual(tensor1.shape, tensor2.shape)
                self.assertEqual(tensor1.tolist(), tensor2.tolist())

    def test_symmetry_from_games(self):
            for board in BoardGenerator(1000):
                fen1 = board.fen()
                tensor1 = TensorBoardUtilV4.fromBoard(board)
                board2 = TensorBoardUtilV4.toBoard(tensor1)
                fen2 = board2.fen()
                tensor2 = TensorBoardUtilV4.fromBoard(board2)
                with self.subTest(fen1):
                    self.assertEqual(tensor1.tolist(), tensor2.tolist())
                    self.assertEqual(fen1, fen2)

    def test_en_passant_symmetry(self):
        board = chess.Board()
        for square in [None] + list(range(64)):
            board.ep_square = square
            fen1 = board.fen()
            tensor1 = TensorBoardUtilV4.fromBoard(board)
            board2 = TensorBoardUtilV4.toBoard(tensor1)
            tensor2 = TensorBoardUtilV4.fromBoard(board2)
            fen2 = board2.fen()
            with self.subTest(fen2):
                self.assertEqual(tensor1.tolist(), tensor2.tolist())
                self.assertEqual(fen1, fen2)
                self.assertEqual(board.ep_square, board2.ep_square)
                self.assertEqual(board2.ep_square, square)

    def test_castling_symmetry(self):
        full_cs = "KQkq"
        castle_strings =  [''.join(tup) for tup in (list(itertools.combinations(full_cs, 1)) 
                    + list(itertools.combinations(full_cs, 2)) 
                    + list(itertools.combinations(full_cs, 3)) 
                    + list(itertools.combinations(full_cs, 4)))]
        for c in castle_strings:
            board1 = chess.Board()
            board1.set_castling_fen(c)
            tensor1 = TensorBoardUtilV4.fromBoard(board1)
            board2 = TensorBoardUtilV4.toBoard(tensor1)
            tensor2 = TensorBoardUtilV4.fromBoard(board2)
            fen1 = board1.fen()
            fen2 = board2.fen()
            with self.subTest(fen2):
                self.assertEqual(tensor1.tolist(), tensor2.tolist())
                self.assertEqual(fen1, fen2)
                self.assertEqual(board1.castling_rights, board2.castling_rights)
                if "K" in c:
                     self.assertTrue(board2.has_kingside_castling_rights(chess.WHITE))
                if "Q" in c:
                     self.assertTrue(board2.has_queenside_castling_rights(chess.WHITE))
                if "k" in c:
                     self.assertTrue(board2.has_kingside_castling_rights(chess.BLACK))
                if "q" in c:
                     self.assertTrue(board2.has_queenside_castling_rights(chess.BLACK))

    def test_simulate_continous(self):
        for board in BoardGenerator(100):
            tensor1 = TensorBoardUtilV4.fromBoard(board)
            tensor2 = TensorBoardUtilV4.simulateContinous(tensor1)
            tensor3 = torch.round(tensor2)
            with self.subTest(board.fen):
                self.assertEqual(tensor1.tolist(), tensor3.tolist())
                self.assertNotEqual(tensor1.tolist(), tensor2.tolist())

    def test_argmax_equality(self):
        for board in BoardGenerator(100):
            for _ in range(100):
                fen = board.fen()
                tensor1 = TensorBoardUtilV4.fromBoard(board)
                tensor2 = TensorBoardUtilV4.simulateContinous(tensor1)
                tensor3 = TensorBoardUtilV4.discretizeTensor(tensor1)
                with self.subTest(fen):
                    self.assertEqual(tensor2.shape[0], tensor1.shape[0])
                    self.assertEqual(tensor1.tolist(), tensor3.tolist())



 
    def test_discretize_castling(self):
        full_cs = "KQkq"
        castle_strings =  [''.join(tup) for tup in (list(itertools.combinations(full_cs, 1)) 
                    + list(itertools.combinations(full_cs, 2)) 
                    + list(itertools.combinations(full_cs, 3)) 
                    + list(itertools.combinations(full_cs, 4)))]
        
        for cfen in castle_strings:
            for board1 in BoardGenerator(10):
                tensor1 = TensorBoardUtilV4.fromBoard(board1)
                pieces_tensor = TensorBoardUtilV4.tensorToPieceTensors(tensor1)
                pieces_tensor = torch.reshape(TorchUtils.argmax_onehot(pieces_tensor), (pieces_tensor.shape[0], TensorBoardUtilV4.PIECE_COMPONENT_SIZE * 64))
                turn_tensor = TensorBoardUtilV4.tensorToTurn(tensor1)
                en_passant_tensor = TensorBoardUtilV4.tensorToEnPassant(tensor1)
                clock_tensor = TensorBoardUtilV4.tensorToTimers(tensor1)
    
                board1_castle_fen = TensorBoardUtilV4.getCastlingFEN(board1)
                if cfen not in board1_castle_fen:
                    continue
                
                castling_tensor = TensorBoardUtilV4.castlingTensorFromFEN(cfen)
                tensor2 = torch.cat([pieces_tensor, turn_tensor, castling_tensor, en_passant_tensor, clock_tensor], dim = 1)
                board2 = TensorBoardUtilV4.toBoard(tensor2)

                fen1 = board1.fen()
                fen2 = board2.fen()
                with self.subTest(f"\n{fen1}\n{fen2}"):
                    self.assertEqual(cfen, TensorBoardUtilV4.getCastlingFEN(board2))
                    if "K" in cfen:
                        self.assertTrue(board2.has_kingside_castling_rights(chess.WHITE))
                    if "Q" in cfen:
                        self.assertTrue(board2.has_queenside_castling_rights(chess.WHITE))
                    if "k" in cfen:
                        self.assertTrue(board2.has_kingside_castling_rights(chess.BLACK))
                    if "q" in cfen:
                        self.assertTrue(board2.has_queenside_castling_rights(chess.BLACK))
            

            

    
    def test_invariants(self):
        return
        for move_num in range(30):
            board = chess.Board()
            for _ in range(100):
                fen = board.fen()
                tensor = TensorBoardUtilV4.fromBoard(board)

                with self.subTest(f"Bitrange"):
                    self.assertEqual(tensor.shape[0], TensorBoardUtilV4.SIZE)
                    bitrange = tensor[0:TensorBoardUtilV4.BINARY_RANGE]
                    gt1 = torch.lt(bitrange, torch.full([TensorBoardUtilV4.BINARY_RANGE], 2)).float().sum().item()
                    self.assertEqual(gt1, TensorBoardUtilV4.BINARY_RANGE)
                    moves = list(board.legal_moves)



                board.push(moves[move_num % len(moves)])

if __name__ == "__main__":
    unittest.main()