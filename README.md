# minigraph-compiler

A tiny, executable computation-graph compiler for studying optimization passes. Programs use a JSON-compatible SSA-like intermediate representation; the current optimizer performs constant folding and dead-code elimination, then verifies semantics with an interpreter.

This is an educational compiler lab, not a production graph runtime.

## Run

```bash
python3 compiler.py example.json
python3 -m unittest -v
```

See [ROADMAP.md](ROADMAP.md) for planned IR, analysis, and lowering passes.

