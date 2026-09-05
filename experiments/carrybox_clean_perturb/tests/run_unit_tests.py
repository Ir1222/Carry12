"""Run all focused CPU tests with PyTorch/NumPy; no Isaac Gym or pytest needed."""

import importlib
from pathlib import Path


def discover_tests():
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        for name, value in sorted(vars(module).items()):
            if name.startswith("test_") and callable(value):
                yield value


if __name__ == "__main__":
    tests = list(discover_tests())
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} evaluator CPU tests")
