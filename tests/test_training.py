import argparse
from src.training.trainer import Trainer
from src.utils.config import load_configs
import unittest


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.config = load_configs("configs/training.yaml")
        self.config["epochs"] = 1
        self.device = "mps"

    def test_trainer_initialization(self):
        trainer = Trainer(config=self.config, device_name=self.device)
        self.assertIsNotNone(trainer)

    def test_trainer_train(self):
        trainer = Trainer(config=self.config, device_name=self.device)
        trainer.train()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
