# minigraph-compiler

A tiny, executable computation-graph compiler for studying optimization passes. Programs use a JSON-compatible SSA-like intermediate representation; the current optimizer performs constant folding and dead-code elimination, then verifies semantics with an interpreter.

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
tensor with shape `[2, 2]`). Elementwise tensor operations and broadcasting
are planned next.
