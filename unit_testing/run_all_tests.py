import unittest
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Discover and run all tests in the "tests" folder
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(start_dir="unit_testing", pattern="test*.py")

test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)