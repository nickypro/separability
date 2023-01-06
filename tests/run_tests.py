""" Runs all tests in the tests folder. """

import argparse
from test_attn_crossover_delete import test_calculate_attn_crossover_and_delete
from test_collection import test_ff_collections, test_attn_collections
from test_delete_attn_pre_out_layer import test_delete_attn_pre_out_layer
from test_delete_ff_keys import test_ff_key_counting, test_delete_ff_keys
from test_delete_ff_and_evaluate import test_delete_ff_and_evaluate
from test_evaluate_all import test_evaluate_all
from test_prune_and_evaluate import test_prune_and_evaluate

tests = [
    test_evaluate_all,
    test_calculate_attn_crossover_and_delete,
    test_ff_collections,
    test_attn_collections,
    test_delete_attn_pre_out_layer,
    test_ff_key_counting,
    test_delete_ff_keys,
    test_delete_ff_and_evaluate,
    test_prune_and_evaluate,
]

def test_all(verbose: bool = False):
    num_tests_run = 0
    num_tests_passed = 0

    tests_failed = []
    for test in tests:
        try:
            num_tests_run += 1
            output = test(verbose)

            if not output is True:
                raise AssertionError("Test did not return True")

            num_tests_passed += 1
            print(f"Test Passed: {test.__name__}")

        # pylint: disable=W0703
        except Exception as err:
            tests_failed.append(test.__name__)
            print(f"Test Failed: {test.__name__}")
            print(err)

    print()
    print("# Tests Summary:")
    if len(tests_failed) > 0:
        print(f"Failed Tests: {tests_failed}")
    print(f"Passed {num_tests_passed}/{num_tests_run} tests")
    return num_tests_passed == num_tests_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", "-v", action="store_true", default=False,
        help="Print extra information" )
    args = parser.parse_args()

    test_all(args.verbose)
