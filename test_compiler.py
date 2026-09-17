import unittest

from compiler import algebraic_simplify, common_subexpression_elimination, constant_propagate, eliminate_dead_code, fuse_multiply_add, graphviz_export, infer_metadata, liveness_report, lower_to_stack_machine, memory_slot_reuse_analysis, optimize, optimize_with_trace, parse_textual_ir, print_textual_ir, run, run_stack_machine, topological_sort


PROGRAM = [
    {"op": "const", "out": "x", "value": 2},
    {"op": "const", "out": "y", "value": 3},
    {"op": "add", "out": "sum", "args": ["x", "y"]},
    {"op": "mul", "out": "unused", "args": ["x", "y"]},
    {"op": "return", "args": ["sum"]},
]


class CompilerTests(unittest.TestCase):
    def test_graphviz_export_emits_dependency_edges_and_return_node(self):
        dot = graphviz_export(PROGRAM)
        self.assertEqual(dot, """digraph program {
  rankdir=LR;
  \"x\" [label=\"x = const\"];
  \"y\" [label=\"y = const\"];
  \"sum\" [label=\"sum = add\"];
  \"x\" -> \"sum\";
  \"y\" -> \"sum\";
  \"unused\" [label=\"unused = mul\"];
  \"x\" -> \"unused\";
  \"y\" -> \"unused\";
  \"return_4\" [label=\"return\"];
  \"sum\" -> \"return_4\";
}""")

    def test_textual_ir_round_trips_constants_and_metadata(self):
        program = [
            {"op": "const", "out": "values", "value": [1, 2], "dtype": "float", "shape": [2]},
            {"op": "const", "out": "limit", "value": 2},
            {"op": "lt", "out": "result", "args": ["values", "limit"]},
            {"op": "return", "args": ["result"]},
        ]
        printed = print_textual_ir(program)
        self.assertEqual(parse_textual_ir(printed), program)
        self.assertEqual(run(parse_textual_ir(printed)), [True, False])

    def test_textual_ir_rejects_invalid_line_with_location(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            parse_textual_ir("x = const 1\nthis is not IR")

    def test_optimization_preserves_result(self):
        optimized = optimize(PROGRAM)
        self.assertEqual(run(PROGRAM), run(optimized))
        self.assertEqual(optimized, [
            {"op": "const", "out": "sum", "value": 5.0},
            {"op": "return", "args": ["sum"]},
        ])

    def test_fixed_point_trace_reports_each_pass(self):
        optimized, trace = optimize_with_trace(PROGRAM)
        self.assertEqual(optimized, optimize(PROGRAM))
        self.assertEqual([step["pass"] for step in trace[:6]], [
            "common_subexpression_elimination",
            "fuse_multiply_add",
            "algebraic_simplify",
            "constant_propagate",
            "constant_fold",
            "eliminate_dead_code",
        ])
        self.assertTrue(any(step["changed"] for step in trace))
        self.assertTrue(all(not step["changed"] for step in trace[-5:]))
        self.assertEqual(trace[-1]["nodes_after"], len(optimized))

    def test_dead_code_keeps_dependencies(self):
        reduced = eliminate_dead_code(PROGRAM)
        self.assertEqual([node.get("out") for node in reduced], ["x", "y", "sum", None])

    def test_liveness_report_tracks_values_before_and_after_each_instruction(self):
        report = liveness_report(PROGRAM)
        self.assertEqual(report, [
            {"live_in": [], "live_out": ["x"]},
            {"live_in": ["x"], "live_out": ["x", "y"]},
            {"live_in": ["x", "y"], "live_out": ["sum", "x", "y"]},
            {"live_in": ["sum", "x", "y"], "live_out": ["sum"]},
            {"live_in": ["sum"], "live_out": []},
        ])

    def test_memory_slot_analysis_reuses_slots_after_last_use(self):
        program = [
            {"op": "const", "out": "left", "value": 2},
            {"op": "const", "out": "right", "value": 3},
            {"op": "add", "out": "total", "args": ["left", "right"]},
            {"op": "const", "out": "bias", "value": 4},
            {"op": "add", "out": "result", "args": ["total", "bias"]},
            {"op": "return", "args": ["result"]},
        ]
        self.assertEqual(memory_slot_reuse_analysis(program), {
            "left": 0, "right": 1, "total": 0, "bias": 1, "result": 0,
        })

    def test_stack_machine_lowering_preserves_program_result(self):
        instructions = lower_to_stack_machine(PROGRAM)
        self.assertEqual(instructions[:7], [
            {"op": "push_const", "value": 2}, {"op": "store", "name": "x"},
            {"op": "push_const", "value": 3}, {"op": "store", "name": "y"},
            {"op": "load", "name": "x"}, {"op": "load", "name": "y"}, {"op": "add"},
        ])
        self.assertEqual(run_stack_machine(instructions), run(PROGRAM))

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

    def test_topological_sort_orders_forward_dependencies_stably(self):
        unordered = [
            {"op": "add", "out": "sum", "args": ["x", "y"]},
            {"op": "return", "args": ["sum"]},
            {"op": "const", "out": "y", "value": 3},
            {"op": "const", "out": "x", "value": 2},
        ]
        ordered = topological_sort(unordered)
        self.assertEqual([node.get("out") for node in ordered], ["x", "y", "sum", None])
        self.assertEqual(run(ordered), 5.0)

    def test_topological_sort_reports_ssa_cycle(self):
        cyclic = [
            {"op": "alias", "out": "x", "args": ["y"]},
            {"op": "alias", "out": "y", "args": ["x"]},
            {"op": "return", "args": ["x"]},
        ]
        with self.assertRaisesRegex(ValueError, "cycle detected: x -> y -> x"):
            topological_sort(cyclic)

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

    def test_fuses_single_use_multiply_add(self):
        program = [
            {"op": "const", "out": "x", "value": 2},
            {"op": "const", "out": "y", "value": 3},
            {"op": "const", "out": "bias", "value": 4},
            {"op": "mul", "out": "product", "args": ["x", "y"]},
            {"op": "add", "out": "result", "args": ["product", "bias"]},
            {"op": "return", "args": ["result"]},
        ]
        fused = fuse_multiply_add(program)
        self.assertEqual(fused[-2], {"op": "fma", "out": "result", "args": ["x", "y", "bias"]})
        self.assertEqual(run(fused), run(program))

    def test_simplifies_arithmetic_identities_and_rewrites_uses(self):
        program = [
            {"op": "const", "out": "x", "value": 4},
            {"op": "const", "out": "zero", "value": 0},
            {"op": "const", "out": "one", "value": 1},
            {"op": "add", "out": "plus_zero", "args": ["x", "zero"]},
            {"op": "sub", "out": "minus_zero", "args": ["plus_zero", "zero"]},
            {"op": "mul", "out": "times_one", "args": ["one", "minus_zero"]},
            {"op": "div", "out": "result", "args": ["times_one", "one"]},
            {"op": "return", "args": ["result"]},
        ]
        simplified = algebraic_simplify(program)
        self.assertEqual(simplified[-1]["args"], ["x"])
        self.assertEqual(run(program), run(simplified))
        self.assertEqual(optimize(program), [
            {"op": "const", "out": "x", "value": 4},
            {"op": "return", "args": ["x"]},
        ])

    def test_propagates_constants_through_aliases(self):
        program = [
            {"op": "const", "out": "x", "value": 2},
            {"op": "alias", "out": "x_copy", "args": ["x"]},
            {"op": "alias", "out": "x_again", "args": ["x_copy"]},
            {"op": "const", "out": "y", "value": 3},
            {"op": "add", "out": "result", "args": ["x_again", "y"]},
            {"op": "return", "args": ["result"]},
        ]
        propagated = constant_propagate(program)
        self.assertEqual(propagated[-2]["args"], ["x", "y"])
        self.assertEqual(run(program), run(propagated))
        self.assertEqual(optimize(program), [
            {"op": "const", "out": "result", "value": 5.0},
            {"op": "return", "args": ["result"]},
        ])


if __name__ == "__main__":
    unittest.main()
