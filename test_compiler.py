import unittest

from compiler import common_subexpression_elimination, eliminate_dead_code, infer_metadata, optimize, run


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

    def test_comparison_and_select_fold(self):
        program = [
            {"op": "const", "out": "x", "value": 2},
            {"op": "const", "out": "y", "value": 3},
            {"op": "lt", "out": "condition", "args": ["x", "y"]},
            {"op": "const", "out": "chosen", "value": 10},
            {"op": "const", "out": "rejected", "value": 20},
            {"op": "select", "out": "result", "args": ["condition", "chosen", "rejected"]},
            {"op": "return", "args": ["result"]},
        ]
        optimized = optimize(program)
        self.assertEqual(run(program), 10.0)
        self.assertEqual(optimized, [
            {"op": "const", "out": "result", "value": 10.0},
            {"op": "return", "args": ["result"]},
        ])

    def test_comparison_result_stays_boolean_after_folding(self):
        program = [
            {"op": "const", "out": "x", "value": 2},
            {"op": "const", "out": "y", "value": 3},
            {"op": "eq", "out": "result", "args": ["x", "y"]},
            {"op": "return", "args": ["result"]},
        ]
        self.assertIs(run(program), False)
        self.assertIs(run(optimize(program)), False)

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

    def test_infers_scalar_shape_and_dtype(self):
        program = [
            {"op": "const", "out": "x", "value": 2},
            {"op": "const", "out": "y", "value": 3},
            {"op": "lt", "out": "condition", "args": ["x", "y"]},
            {"op": "return", "args": ["condition"]},
        ]
        self.assertEqual(infer_metadata(program), {
            "x": ((), "float"),
            "y": ((), "float"),
            "condition": ((), "bool"),
        })

    def test_rejects_incompatible_metadata(self):
        with self.assertRaisesRegex(ValueError, "constant dtype does not match value"):
            optimize([
                {"op": "const", "out": "x", "value": 1, "dtype": "bool"},
                {"op": "const", "out": "y", "value": 2},
                {"op": "add", "out": "sum", "args": ["x", "y"]},
                {"op": "return", "args": ["sum"]},
            ])

    def test_runs_tensor_constant_and_infers_metadata(self):
        program = [
            {"op": "const", "out": "matrix", "value": [[1, 2], [3, 4]]},
            {"op": "return", "args": ["matrix"]},
        ]
        self.assertEqual(run(program), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(infer_metadata(program)["matrix"], ((2, 2), "float"))

    def test_rejects_ragged_tensor_constant(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            run([
                {"op": "const", "out": "matrix", "value": [[1, 2], [3]]},
                {"op": "return", "args": ["matrix"]},
            ])

    def test_broadcasts_tensor_arithmetic_and_folds(self):
        program = [
            {"op": "const", "out": "matrix", "value": [[1, 2, 3], [4, 5, 6]]},
            {"op": "const", "out": "column", "value": [[10], [20]]},
            {"op": "add", "out": "result", "args": ["matrix", "column"]},
            {"op": "return", "args": ["result"]},
        ]
        expected = [[11.0, 12.0, 13.0], [24.0, 25.0, 26.0]]
        self.assertEqual(run(program), expected)
        self.assertEqual(infer_metadata(program)["result"], ((2, 3), "float"))
        self.assertEqual(run(optimize(program)), expected)

        row_program = [
            {"op": "const", "out": "matrix", "value": [[1, 2, 3], [4, 5, 6]]},
            {"op": "const", "out": "row", "value": [10, 20, 30]},
            {"op": "add", "out": "result", "args": ["matrix", "row"]},
            {"op": "return", "args": ["result"]},
        ]
        self.assertEqual(run(row_program), [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])

    def test_broadcasts_scalar_comparison(self):
        program = [
            {"op": "const", "out": "values", "value": [1, 2]},
            {"op": "const", "out": "limit", "value": 2},
            {"op": "lt", "out": "result", "args": ["values", "limit"]},
            {"op": "return", "args": ["result"]},
        ]
        self.assertEqual(run(program), [True, False])
        self.assertEqual(infer_metadata(program)["result"], ((2,), "bool"))

    def test_rejects_incompatible_broadcast_shapes(self):
        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            run([
                {"op": "const", "out": "left", "value": [1, 2]},
                {"op": "const", "out": "right", "value": [1, 2, 3]},
                {"op": "add", "out": "result", "args": ["left", "right"]},
                {"op": "return", "args": ["result"]},
            ])

    def test_eliminates_common_subexpressions_and_rewrites_uses(self):
        program = [
            {"op": "const", "out": "x", "value": 2},
            {"op": "const", "out": "y", "value": 3},
            {"op": "add", "out": "first", "args": ["x", "y"]},
            {"op": "add", "out": "second", "args": ["x", "y"]},
            {"op": "mul", "out": "result", "args": ["first", "second"]},
            {"op": "return", "args": ["result"]},
        ]
        reduced = common_subexpression_elimination(program)
        self.assertEqual([node.get("out") for node in reduced], ["x", "y", "first", "result", None])
        self.assertEqual(reduced[-2]["args"], ["first", "first"])
        self.assertEqual(run(program), run(reduced))


if __name__ == "__main__":
    unittest.main()
