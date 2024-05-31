import unittest

from HARK.distribution import Bernoulli
import HARK.model as model
from HARK.model import Control

# TODO: let the shock constructor reference this parameter.
LivPrb = 0.98

test_block_A_data = {
    "name": "test block A",
    "shocks": {
        "live": Bernoulli(p=LivPrb),
    },
    "dynamics": {
        "y": lambda p: p,
        "m": lambda Rfree, a, y: Rfree * a + y,
        "c": Control(["m"]),
        "p": lambda PermGroFac, p: PermGroFac * p,
        "a": lambda m, c: m - c,
    },
    "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
}


class test_DBlock(unittest.TestCase):
    def setUp(self):
        self.test_block_A = model.DBlock(**test_block_A_data)

    def test_init(self):
        self.assertEqual(self.test_block_A.name, "test block A")
