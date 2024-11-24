from dataclasses import dataclass
from tkinter.filedialog import Open
from typing import Optional
import pyzstd
import io
import torch
import pickle
from chess import pgn
import tqdm
import threading

from board_final import TensorBoardUtilV4
SMALL_DATASET = "datasets/lichess_db_standard_rated_2013-01.pgn.zst"
SMALL_TENSORS = "board_tensorsV3/2013-01-board-tensors"
LARGER_DATASET = "datasets/lichess_db_standard_rated_2014-07.pgn.zst"
LARGER_TENSORS = "board_tensorsV3/2014-07-board-tensors-first-half"
LARGER_TENSORS2 = "board_tensorsV3/2014-07-board-tensors-second-half"
DATASET3 = "datasets/lichess_db_standard_rated_2013-12.pgn.zst"
TENSORS3 = "board_tensorsV3/2013-12-board-tensors"




def read_pgn_zst(path : str) -> list[pgn.Game]:
    """
    Reats a .pgn.zst file and returns a list of python-chess Game objects
    """
    games = []
    print(f"reading games from {path}...")
    with pyzstd.open(path, "r") as zst:
        bytes = zst.read()
        f = io.StringIO(bytes.decode("utf-8"))
        last = pgn.read_game(f)
        while last != None:
            games.append(last)
            last = pgn.read_game(f)
    print("done!")
    return games
        


def games_to_board_file(games : list[pgn.Game], path : str):
    tensors_list = []
    for i in tqdm.tqdm(range(len(games)), desc = "Making tensors"):
        game = games[i]
        board = game.board()
        tensors_list.append(TensorBoardUtilV4.fromBoard(board))
        for move in game.mainline_moves():
            board.push(move)
            tensors_list.append(TensorBoardUtilV4.fromBoard(board))
     

        
def games_to_board_tensors(games : list[pgn.Game], tensors_list : list[torch.Tensor]):
    for i in tqdm.tqdm(range(len(games)), desc = "Making tensors"):
        game = games[i]
        board = game.board()
        tensors_list.append(TensorBoardUtilV4.fromBoard(board))
        for move in game.mainline_moves():
            board.push(move)
            tensors_list.append(TensorBoardUtilV4.fromBoard(board))


def threaded_games_to_tensors(games : list[pgn.Game]) -> torch.Tensor:
    split_games = []
    THREADS = 20
    increment = len(games) // THREADS
    for i in range(THREADS):
        split_games.append(games[i * increment : min((i+1) * increment, len(games))])
    threads = []
    split_tensors = []
    for i in range(THREADS):
        out_list = []
        split_tensors.append(out_list)
        threads.append(threading.Thread(target = games_to_board_tensors, args = (split_games[i], out_list)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    tensors = []
    for l in split_tensors:
        tensors.append(torch.stack(l))
    return torch.cat(tensors)


def load_tensor(path : str) -> torch.Tensor:
    with open(path, "rb") as f:
        tensors : torch.Tensor = pickle.load(f)
    return tensors.requires_grad_(False)

def pipeline():
    games = read_pgn_zst(DATASET3)
    tensors = threaded_games_to_tensors(games)
    pickle.dump(tensors, open(TENSORS3, "wb"))



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