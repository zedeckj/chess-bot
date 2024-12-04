import chess
import torch
import random
import sys
sys.path.append("./")
from src.board_final import TensorBoardUtilV4

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

def en_passant_accuarcy(discrete_output: torch.Tensor, discrete_target: torch.Tensor) -> tuple[float,float]:
    """
    Returns a tuple of the accuarcy of accuarcy of non en-passants, and true en-passants
    % of None Correct, % of non-None correct
    """
    en_passant_output = TensorBoardUtilV4.tensorToEnPassant(discrete_output)
    en_passant_target = TensorBoardUtilV4.tensorToEnPassant(discrete_output)
    # Shape is [N, 65]
    en_passant_output = torch.argmax(en_passant_output, dim = 1)
    en_passant_target = torch.argmax(en_passant_target, dim = 1)



    none_count = en_passant_output.shape[0]
    ep_count = 0

    none_correct = 0
    ep_correct = 0
    if en_passant_target == 0:
        none_count += 1
        if en_passant_output == 0:
            none_correct += 1

    else: 
        ep_count += 1
        if en_passant_target == en_passant_output:
            ep_correct += 1
    
    return none_correct/none_count, ep_correct/ep_count

def turn_accuracy(discrete_output : torch.Tensor, discrete_target : torch.Tensor):
    output_turns = TensorBoardUtilV4.tensorToTurn(discrete_output)
    target_turns = TensorBoardUtilV4.tensorToTurn(discrete_target)
    unreduced = torch.eq(output_turns.bool(), target_turns.bool()).float()
    return torch.sum(unreduced) / output_turns.shape[0]

def ep_accuarcy(discrete_output : torch.Tensor, discrete_target : torch.Tensor):
    output_ep = TensorBoardUtilV4.tensorToEnPassant(discrete_output)
    target_ep = TensorBoardUtilV4.tensorToEnPassant(discrete_target)
    print(output_ep.shape)
    return torch.sum(torch.eq(output_ep.bool(),target_ep.bool()))/(target_ep.shape[0] * 65)

def clock_accuarcy(discrete_output : torch.Tensor, discrete_target : torch.Tensor):
    output_clocks = TensorBoardUtilV4.tensorToTimers(discrete_output)
    target_clocks = TensorBoardUtilV4.tensorToTimers(discrete_target)
    return torch.mean(torch.lt(torch.abs(output_clocks - target_clocks), 2).float())

def metrics_tensor(discrete_output : torch.Tensor, discrete_target : torch.Tensor):

    """
        [isWhitePawn, isWhiteKnight, isWhiteBishop, isWhiteRook, isWhiteQueen, isWhiteKing, 
        isBlackPawn, isBlackKnight, isBlackBishop, isBlackRook, isBlackQueen, isBlackKing, 
        isBlankSpace, whiteKingsideCastling, whiteQueensideCastling, blackKingsideCastling, 
        blackQueensideCastling]
    """
    assert(len(discrete_output.shape) == 2 and discrete_output.shape[1] == TensorBoardUtilV4.SIZE)
    device = discrete_output.get_device()
    #accuarcy = total_correct / discrete_target.shape[-1]


    piece_output = TensorBoardUtilV4.tensorToPieceTensors(discrete_output) # [board_count, 64, 13]
    piece_target = TensorBoardUtilV4.tensorToPieceTensors(discrete_target)

    

    
    # recall = true positives / true positives + false negatives
    # pre = true positives / true positives + false positives
    """
    true_positives = torch.zeros(13, device=device)
    false_positives = torch.zeros(13, device=device)
    false_negatives = torch.zeros(13, device=device)
    """

    
    true_positives = torch.logical_and(piece_output, piece_target).float().sum(dim = 1).sum(dim = 0)
    false_positives = torch.logical_and(piece_output, torch.logical_not(piece_target)).float().sum(dim = 1).sum(dim = 0)
    false_negatives = torch.logical_and(torch.logical_not(piece_output), piece_target).float().sum(dim = 1).sum(dim = 0)

    """
    for i in range(64):
        square_output = piece_output[...,i, :]
        square_target = piece_target[...,i, :]
        print(square_output.shape)
        log_and = torch.logical_and(square_output, square_target)
        true_positives += torch.logical_and(square_output, square_target,).float().sum(dim = 0)
        false_positives += torch.logical_and(square_output, torch.logical_not(square_target)).float().sum(dim = 0)
        false_negatives += torch.logical_and(torch.logical_not(square_output), square_target).float().sum(dim = 0)"""

    castling_output = TensorBoardUtilV4.tensorToCastlingRights(discrete_output)
    castling_target = TensorBoardUtilV4.tensorToCastlingRights(discrete_target)




    
    true_positives_castling = torch.logical_and(castling_output, castling_target).float().sum(dim = 0)
    false_positives_castling = torch.logical_and(castling_output, torch.logical_not(castling_target)).float().sum(dim = 0)
    false_negatives_castling = torch.logical_and(torch.logical_not(castling_output), castling_target).float().sum(dim = 0)
    print(true_positives_castling, false_positives_castling, false_negatives_castling)
    true_positives = torch.cat([true_positives, true_positives_castling])
    false_positives = torch.cat([false_positives, false_positives_castling])
    false_negatives = torch.cat([false_negatives, false_negatives_castling])

    recall_tensor = true_positives / (true_positives + false_negatives)
    precision_tensor = true_positives / (true_positives + false_positives)
    return recall_tensor, precision_tensor

def display_precision_recall(real_output : torch.Tensor, discrete_target : torch.Tensor):
    discrete_output = TensorBoardUtilV4.discretizeTensor(real_output)
    recall_tensor, precision_tensor = metrics_tensor(discrete_output, discrete_target)
    index_to_name = ["White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King", 
                     "Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King",
                     "Black Space", "White Kingside Castling", "White Queenside Castling", "Black Kingside Castling",
                     "Black Queenside Castling",]
    for i in range(17):
        print(f"{index_to_name[i]} Precision: {precision_tensor[i].item()*100:2f}%")
        print(f"{index_to_name[i]} Recall: {recall_tensor[i].item() * 100:2f}%")
    ep_acc = ep_accuarcy(discrete_output, discrete_target)
    print(f"En-Passant Accuarcy: {ep_acc.item()*100:2f}%")
    turn_acc = turn_accuracy(discrete_output, discrete_target)
    print(f"Turn Accuarcy: {turn_acc.item()*100:2f}%")
    clock_acc = clock_accuarcy(discrete_output, discrete_target)
    print(f"Clock Accuarcy: {clock_acc.item()*100:2f}%")
    #rint(f"En-Passant Accuarcy: {ep_acc*100:2f}%")


