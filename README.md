# minigraph-compiler

A tiny, executable computation-graph compiler for studying optimization passes. Programs use a JSON-compatible SSA-like intermediate representation; the current optimizer performs common-subexpression elimination, constant folding, and dead-code elimination, then verifies semantics with an interpreter.

This is an educational compiler lab, not a production graph runtime.

## Run

```bash
python3 compiler.py example.json
python3 -m unittest -v
```

See [ROADMAP.md](ROADMAP.md) for planned IR, analysis, and lowering passes.

Every scalar value has inferred metadata: `shape` is `[]`, and `dtype` is
`"float"` or `"bool"`. Constants may state matching `dtype` and `shape`
metadata explicitly; incompatible operator inputs are rejected.

Constants may also be non-empty rectangular JSON arrays. Their shape and dtype
are inferred recursively (for example, `[[1, 2], [3, 4]]` is a `float`
tensor with shape `[2, 2]`). Arithmetic and comparisons are elementwise and
follow NumPy-style broadcasting: dimensions must match or one side must be
`1`; scalars broadcast to every element.

`alias` creates an SSA name for one existing value. The optimizer removes alias
chains, allowing constants to reach later folding passes.

`liveness_report(program)` returns the SSA values live immediately before and
after each instruction, which can guide later storage-reuse passes.

`topological_sort(program)` orders a valid dependency graph even when its
instructions are supplied out of order. Cycles are rejected with the involved
SSA names, such as `cycle detected: x -> y -> x`.

`optimize_with_trace(program)` runs the optimizer to a fixed point and returns
the optimized program plus one record per pass. Each record includes the pass
name, iteration, whether it changed the IR, and node counts before and after.

`parse_textual_ir(text)` and `print_textual_ir(program)` convert programs to a
one-instruction-per-line syntax such as `sum = add x y` and `return sum`.
Constants use JSON values; optional metadata is written as
`dtype="float" shape=[2]`.

`graphviz_export(program)` returns a Graphviz DOT dependency graph with one
node per instruction and edges from each argument to its consumer.

The optimizer also fuses a single-use `mul` immediately followed by `add` into
an `fma` instruction, preserving the program result while exposing a
fused-multiply-add lowering opportunity.

`memory_slot_reuse_analysis(program)` assigns buffer slots to live SSA values,
reusing a slot after its prior value reaches its last use.
