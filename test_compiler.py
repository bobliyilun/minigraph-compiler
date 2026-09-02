import unittest

from compiler import eliminate_dead_code, optimize, run


PROGRAM = [
    {"op": "const", "out": "x", "value": 2},
    {"op": "const", "out": "y", "value": 3},
    {"op": "add", "out": "sum", "args": ["x", "y"]},
    {"op": "mul", "out": "unused", "args": ["x", "y"]},
    {"op": "return", "args": ["sum"]},
]


class CompilerTests(unittest.TestCase):
    def test_optimization_preserves_result(self):
        optimized = optimize(PROGRAM)
        self.assertEqual(run(PROGRAM), run(optimized))
        self.assertEqual(optimized, [
            {"op": "const", "out": "sum", "value": 5.0},
            {"op": "return", "args": ["sum"]},
        ])

    def test_dead_code_keeps_dependencies(self):
        reduced = eliminate_dead_code(PROGRAM)
        self.assertEqual([node.get("out") for node in reduced], ["x", "y", "sum", None])

    def test_subtraction_and_division_fold(self):
        program = [
            {"op": "const", "out": "x", "value": 9},
            {"op": "const", "out": "y", "value": 3},
            {"op": "sub", "out": "difference", "args": ["x", "y"]},
            {"op": "div", "out": "quotient", "args": ["difference", "y"]},
            {"op": "return", "args": ["quotient"]},
        ]
        optimized = optimize(program)
        self.assertEqual(run(program), 2.0)
        self.assertEqual(run(optimized), 2.0)
        self.assertEqual(optimized, [
            {"op": "const", "out": "quotient", "value": 2.0},
            {"op": "return", "args": ["quotient"]},
        ])

    def test_rejects_use_before_definition(self):
        with self.assertRaisesRegex(ValueError, "use before definition: x"):
            run([{"op": "return", "args": ["x"]}])

    def test_rejects_duplicate_outputs(self):
        with self.assertRaisesRegex(ValueError, "duplicate output: x"):
            optimize([
                {"op": "const", "out": "x", "value": 1},
                {"op": "const", "out": "x", "value": 2},
                {"op": "return", "args": ["x"]},
            ])


if __name__ == "__main__":
    unittest.main()
