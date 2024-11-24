import unittest
from autoencoder import BoardAutoencoderV2

class TestAutoencoder(unittest.TestCase):

    def setUp(self) -> None:
        self.model = BoardAutoencoderV2()
