import unittest
from autoencoder import BoardAutoencoder

class TestAutoencoder(unittest.TestCase):

    def setUp(self) -> None:
        self.model = BoardAutoencoder()
