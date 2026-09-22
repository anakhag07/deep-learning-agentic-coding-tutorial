import unittest

from experiment import run


class ExperimentTest(unittest.TestCase):
    def test_nonlinearity_solves_circles(self):
        results = run()
        self.assertGreater(results["mlp"]["test_accuracy"], 0.9)
        self.assertGreater(results["mlp"]["test_accuracy"] - results["linear"]["test_accuracy"], 0.15)
        self.assertLess(results["mlp"]["final_loss"], results["mlp"]["initial_loss"])


if __name__ == "__main__":
    unittest.main()
