import torch
import random
from board_final import TensorBoardUtilV4

"""
def precision_recall(binary_output : torch.Tensor, binary_target : torch.Tensor):
    # true positives are both predicted as true and actually true
    true_positives = torch.logical_and(binary_output, binary_target).float().sum().item()

    # false positives are predicated as true and actually false
    false_positives = torch.logical_and(binary_output, torch.logical_not(binary_target)).float().sum().item()

    # false negatives are predicted as false and are actually true
    false_negatives = torch.logical_and(torch.logical_not(binary_output), binary_target).float().sum().item()
    if (true_positives + false_negatives) == 0:
        recall = 1
    else:
        recall = true_positives / (true_positives + false_negatives)

    if (true_positives + false_positives) == 0:
        precision = 1
    else:
        precision = true_positives / (true_positives + false_positives)

    return precision, recall
"""


def precision_recall(discrete_output : torch.Tensor, discrete_target : torch.Tensor):

    """
        [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
        isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
        isBlankSpace]
    """
    assert(len(discrete_output.shape) == 2 and discrete_output.shape[1] == TensorBoardUtilV4.SIZE)
    device = discrete_output.get_device()
    total_correct = torch.eq(discrete_output, discrete_target).float().sum().item()
    #accuarcy = total_correct / discrete_target.shape[-1]

    board_count = discrete_output.shape[0]

    piece_output = TensorBoardUtilV4.tensorToPieceTensors(discrete_output) # [board_count, 64, 13]
    piece_target = TensorBoardUtilV4.tensorToPieceTensors(discrete_target)

    

    
    # recall = true positives / true positives + false negatives
    # pre = true positives / true positives + false positives
    true_positives = torch.zeros(13, device=device)
    false_positives = torch.zeros(13, device=device)
    false_negatives = torch.zeros(13, device=device)

    for i in range(64):
        square_output = piece_output[...,i, :]
        square_target = piece_target[...,i, :]

        true_positives += torch.logical_and(square_output, square_target).float().sum(dim = 0)
        false_positives += torch.logical_and(square_output, torch.logical_not(square_target)).float().sum(dim = 0)
        false_negatives += torch.logical_and(torch.logical_not(square_output), square_target).float().sum(dim = 0)

    recall_tensor = true_positives / (true_positives + false_negatives)
    precision_tensor = true_positives / (true_positives + false_positives)
    return recall_tensor, precision_tensor
    

import unittest

class TestTensorBoard(unittest.TestCase):

    def test_accuracy_recall_basic(self):
        ones = torch.ones(100, device= "mps")
        zeros = torch.zeros(100, device= "mps")
        half200 = torch.cat([ones, zeros])
        ones200 = torch.ones(200, device= "mps")
        zero200 = torch.zeros(200, device= "mps")
        pre1, rec1 = precision_recall(ones200, half200)
        pre2, rec2 = precision_recall(zero200, half200)
        self.assertEqual(pre1, 0.5)
        self.assertEqual(rec1, 1)
        self.assertEqual(pre2, 1)
        self.assertEqual(rec2, 0)

if __name__ == "__main__":
    unittest.main()