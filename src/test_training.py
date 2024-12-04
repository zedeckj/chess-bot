import unittest
import sys
import os
import chess


sys.path.append("./")
from src.board_final import TensorBoardUtilV4
from src.autoencoder import BoardAutoencoder, BoardLoss
from src.trainer import SelfSupervisedTrainer

class TestBoardLoss(unittest.TestCase):

    def setUp(self) -> None:
        PATH = "board_tensorsV3/2016-03"
        files = [f"{PATH}/{file}" for file in os.listdir(PATH) if ".tnsrs" in file]
        files.sort()
        testing_files = files[:1]
        training_files = files[1:2]
        trainer = SelfSupervisedTrainer(
            training_files,
            testing_files,
            BoardAutoencoder(),
            "autoencoder",
            [TensorBoardUtilV4.SIZE],
            loss_fn_constructor = BoardLoss,
        )
        self.loss_fn = BoardLoss(trainer.load_training(0))

    def testPieceCount(self):
        real = chess.Board("5bn1/r3p3/2k1p2r/pp4pp/2PB4/PP1P1PP1/3K2RP/RNQ2B1N w - b6 0 31")
        pred = chess.Board("rkkkkrrr/pppbkkrr/ppnpbprp/pppNpPpp/PPPnPbPP/PPNPBKKP/PPPKKKKR/RKKRRRRR w - b6 1 32")
        real_tensor = TensorBoardUtilV4.fromBoard(real).to("mps")
        pred_tensor = TensorBoardUtilV4.fromBoard(pred).to("mps") * 1000
        print(self.loss_fn.piece_invariants_loss_fn(real_tensor * 100))
        print(self.loss_fn.piece_invariants_loss_fn(pred_tensor))
        print(self.loss_fn.piece_loss_fn(pred_tensor, real_tensor))
        print(self.loss_fn.piece_loss_fn(real_tensor * 100, real_tensor))

if __name__ == "__main__":
    unittest.main()
