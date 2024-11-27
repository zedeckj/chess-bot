from abc import ABC, abstractmethod
from dataclasses import dataclass
import multiprocessing
from tkinter.filedialog import Open
from enum import Enum
from typing import Optional, Union
import pyzstd
import io
import torch
import pickle
import chess
from chess import pgn
import tqdm
import threading
import os
import random
from board_final import TensorBoardUtilV4
import mmap
from chess_models import ProductionAutoencoder
SMALL_TENSORS = "board_tensorsV3/2013-01-board-tensors"



LARGER_DATASET = "datasets/lichess_db_standard_rated_2014-07.pgn.zst"
LARGER_TENSORS = "board_tensorsV3/2014-07-board-tensors-first-half"
LARGER_TENSORS2 = "board_tensorsV3/2014-07-board-tensors-second-half"
DATASET3 = "datasets/lichess_db_standard_rated_2013-12.pgn.zst"
TENSORS3 = "board_tensorsV3/2013-12-board-tensors"


SMALL_DATASET = "datasets/lichess_db_standard_rated_2013-01.pgn.zst"
SMALL_DATASET2 = "datasets/lichess_db_standard_rated_2013-12.pgn.zst"
DATASET_2016_03 = "datasets/lichess_db_standard_rated_2016-03.pgn.zst"
DATASET_2015_07 = "datasets/lichess_db_standard_rated_2015-07.pgn.zst"
MASTERS_DATASET = "datasets/masters"
SMALL_DIR = "2013-01"
DIR_2016_03 = "2016-03"
DIR_2013_12 = "2013-12"

MASTERS1 = "masters1"
import time


class AbstractReader(ABC):


    TENSORS_DIR = "board_tensorsV3"
    TENSORS_MOVES_DIR = "move_tensors"
    def __init__(self, in_path : str, out_dir : str):
        self.in_path = in_path
        self.out_dir = out_dir

    @abstractmethod
    def helper(self, procs : int, this_proc : int):
        ...

    def maybe_read_game(self, f : io.TextIOWrapper, current_i : int, modulo : Optional[int] = None, return_index : Optional[int] = None) -> Union[Optional[pgn.Game], bool]:
        if modulo != None and return_index != None:
            if (current_i % modulo) == return_index:
                return pgn.read_game(f)
            pgn.skip_game(f)
            return True

        else:
            out = pgn.read_game(f)
            if out == None:
                return False
            return out
        
    def run(self) -> None:
        t = time.time()
        PROCS = 20
        proc_list = []
        with pyzstd.open(self.in_path, "r") as zst:
            text = zst.read().decode("utf-8")
        with open(f"{AbstractReader.TENSORS_DIR}/tmp", "w") as buffer:
            buffer.write(text)
        del text
        if self.out_dir not in os.listdir(f"{BoardReader.TENSORS_DIR}"):
            os.mkdir(f"{AbstractReader.TENSORS_DIR}/{self.out_dir}")
        assert(len(os.listdir(f"{AbstractReader.TENSORS_DIR}/{self.out_dir}")) == 0)
        for i in range(PROCS):
            proc_list.append(multiprocessing.Process(target = self.helper, args = (PROCS, i, self.out_dir)))
        for p in proc_list:
            p.start()
        for p in proc_list:
            p.join()  
        os.remove(f"{AbstractReader.TENSORS_DIR}/tmp")


    


class BoardReader(AbstractReader):
    """
    Given an input file path and an output directory name, this class converts a .pgn.zst file into 
    pickled TensorBoardUtilV4 tensors files. 
    """


    def helper(self, procs : int, this_proc : int):
        starting_tensor = TensorBoardUtilV4.fromBoard(chess.Board())
        tensors_list : list[torch.Tensor] = []
        with open(f"{AbstractReader.TENSORS_DIR}/tmp", "r") as f:
            i = 0
            saved = 0
            while True:
                if this_proc == 0:
                    print(i, end = "\r")
                last = self.maybe_read_game(f, i, procs, this_proc)
                if type(last) is pgn.Game:
                    tensors_list.append(starting_tensor)
                    board = chess.Board()
                    for move in last.mainline_moves():
                        board.push(move)
                        tensors_list.append(TensorBoardUtilV4.fromBoard(board))
                if len(tensors_list) > 500000 or not last:
                    pickle.dump(torch.cat(tensors_list), open(f"{AbstractReader.TENSORS_DIR}/{self.out_dir}/{this_proc}_{saved}.tnsrs", "wb"))
                    print(f"saved! {this_proc}_{saved}.tnsrs")
                    saved += 1
                    tensors_list = []
                    if not last:
                        break
                i += 1

    def refactor(self):
        path = f"{AbstractReader.TENSORS_DIR}/{self.out_dir}"
        for file in os.listdir(path):
            print(f"Refactoring {file}")
            with open(f"{path}/{file}", "rb") as f:
                tensor = pickle.load(f)
                tensor = torch.squeeze(tensor)
            pickle.dump(tensor,open(f"{path}/{file}", "wb"))

    def verify(self):
        path = f"{AbstractReader.TENSORS_DIR}/{self.out_dir}"
        for file in os.listdir(path):
            print(f"Loading {file}")
            failed = False
            with open(f"{path}/{file}", "rb") as f:
                try:
                    pickle.load(f)
                except:
                    failed = True
            if failed:
                print(f"Loading {file} failed, deleting")
                os.remove(f"{path}/{file}")

class GameOutcome(Enum):

    WHITE_WON = 0
    BLACK_WON = 1
    OTHER = 2

class MoveVisitor(pgn.BaseVisitor[Optional[list[list[chess.Board]]]]):

    """
    When visiting a pgn.Game, this class returns a possible list of list of boards. This value not being None
    indicates that a player won the game, and that each list of boards in the list of lists is ordered such that
    the better position for White is first.
    """

    def __init__(self):
        self.outcome : GameOutcome = GameOutcome.OTHER
        self.white_move_boards : list[list[chess.Board]] = []
        self.black_move_boards : list[list[chess.Board]] = []

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        next_board = board.copy()
        random_board = board.copy()
        next_board.push(move)
        other_moves = [m for m in list(board.legal_moves) if m.uci() != move.uci()]
        if len(other_moves) == 0:
            return
        random_board.push(random.choice(other_moves))
        if board.turn == chess.WHITE:
            self.white_move_boards.append([next_board, random_board])
        else:
            self.black_move_boards.append([random_board, next_board])

    def visit_result(self, result: str) -> None:
        assert (
            result == "1-0" or
            result == "0-1" or
            result == "1/2-1/2" or
            result == "*"
        )
        if result == "1-0":
            self.outcome = GameOutcome.WHITE_WON
        elif result == "0-1":
            self.outcome = GameOutcome.BLACK_WON
        else:
            self.outcome = GameOutcome.OTHER

    def result(self) -> Optional[list[list[chess.Board]]]:
        if self.outcome == GameOutcome.WHITE_WON:
            return self.white_move_boards
        elif self.outcome == GameOutcome.BLACK_WON:
            return self.black_move_boards
            

def load_tensor(path : str) -> torch.Tensor:
    with open(path, "rb") as f:
        tensors : torch.Tensor = pickle.load(f)
    return tensors.requires_grad_(False)


class MoveSelectionReader(AbstractReader):
    """
    Given an input file path and an output directory name, this class converts a .pgn.zst file into a selected move format
    """

    def __init__(self, in_path : str, out_dir : str):
        super().__init__(in_path, out_dir)
        self.autoencoder = ProductionAutoencoder()

    def helper(self, procs : int, this_proc : int):
        
        tensors_list : list[torch.Tensor] = []
        with open(f"{AbstractReader.TENSORS_MOVES_DIR}/tmp", "r") as f:
            i = 0
            saved = 0
            while True:
                if this_proc == 0:
                    print(i, end = "\r")
                last = self.maybe_read_game(f, i, procs, this_proc)
                if type(last) is pgn.Game:
                    visitor = MoveVisitor()
                    maybe_boards = last.accept(visitor)
                    if maybe_boards:
                        for board_list in maybe_boards:
                            """
                            if this_proc == 0:
                                print("WHITE BETTER FIRST")
                                print(board_list[0], "\n\n")
                                print(board_list[1])
                            """
                            tensor0 = self.autoencoder.encodeFromBoard(board_list[0])
                            tensor1 = self.autoencoder.encodeFromBoard(board_list[1])
                            tensors_list.append(torch.stack([tensor0, tensor1]))
                            # The better board for white is always first
                if len(tensors_list) > 50000 or not last:
                    pickle.dump(torch.cat(tensors_list), open(f"{AbstractReader.TENSORS_MOVES_DIR}/{self.out_dir}/{this_proc}_{saved}.tnsrs", "wb"))
                    print(f"saved! {this_proc}_{saved}.tnsrs")
                    saved += 1
                    tensors_list = []
                    if not last:
                        break
                i += 1

    def run_uncompressed(self) -> None:
        t = time.time()
        PROCS = 20
        proc_list = []
        files = [f"{self.in_path}/{f}" for f in os.listdir(self.in_path)]
        with open(f"{AbstractReader.TENSORS_MOVES_DIR}/tmp", "w") as buffer:
            for fname in files:
                with open(fname, "r") as f:
                    buffer.write(f.read() + "\n")
        if self.out_dir not in os.listdir(f"{BoardReader.TENSORS_DIR}"):
            os.mkdir(f"{AbstractReader.TENSORS_MOVES_DIR}/{self.out_dir}")
        assert(len(os.listdir(f"{AbstractReader.TENSORS_MOVES_DIR}/{self.out_dir}")) == 0)
        for i in range(PROCS):
            proc_list.append(multiprocessing.Process(target = self.helper, args = (PROCS, i)))
        for p in proc_list:
            p.start()
        for p in proc_list:
            p.join()  
        os.remove(f"{AbstractReader.TENSORS_MOVES_DIR}/tmp")






def pipeline():
    t = time.time()
    reader = MoveSelectionReader(MASTERS_DATASET, MASTERS1)
    reader.run_uncompressed()
    print(f"Full pipeline took {time.time() - t}")


def load_train_test(tensor_file_name, batch_size, subset_percent : Optional[float] = None): 
    tensor = load_tensor(tensor_file_name)
    if len(tensor.shape) == 3:
        tensor = torch.squeeze(tensor)
    print(tensor.shape)
    batch_count = tensor.shape[0] // batch_size
    tensor = tensor[:batch_count * batch_size]
    tensor = tensor.reshape(batch_count, batch_size, tensor.shape[1])
    train_size = int(tensor.shape[0] * 0.95)
    test_size = int(tensor.shape[0]) - train_size
    split = torch.split(tensor, [train_size, test_size])
    if subset_percent:
        new_train_size = int(train_size * subset_percent)
        new_test_size = int(test_size * subset_percent)
        split = (split[0][0 : new_train_size], split[1][0 : new_test_size])
    print(f"Split into training data {split[0].shape} and testing data {split[1].shape}")
    return split[0], split[1]

if __name__ == "__main__":
    pipeline()