import unittest
import torch
from torch.nn import functional

class TorchUtils:

    @staticmethod
    def argmax_onehot(tensor : torch.Tensor):
        # Find the indices of the maximum values along the last dimension
        indices = tensor.argmax(dim=-1)
        
        # Create a tensor of zeros with the same shape as the input tensor
        output = torch.zeros_like(tensor)
        
        # Use scatter_ to place 1s at the indices of the maximum values
        output.scatter_(-1, indices.unsqueeze(-1), 1)
        return output


class TestTorch(unittest.TestCase):

    def test_argmax_onehot(self):
        r = torch.rand((2,4))
        a = TorchUtils.argmax_onehot(r)
        max_ls = []
        for i in range(r.shape[0]):
            #for j in range(r.shape[1]):
                #for k in range(r.shape[2]):
                    max_l = -1
                    max_v = -2
                    for l in range(r.shape[1]):
                        if max_v < r[i][l]:
                            max_v = r[i][l]
                            max_l = l 
                    max_ls.append(max_l)


    def testSlicing(self):
        tensor = torch.rand((3,3,3))
        print(tensor)
        print(tensor[...,0,:])
if __name__ == "__main__":
    unittest.main()