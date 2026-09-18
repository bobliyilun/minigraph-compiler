# Optimization example

This report was generated with the current `optimize_with_trace` implementation
using the program below. Both interpreters returned `6.0`.

## Before optimization

```text
x = const 2
y = const 3
zero = const 0
product = mul x y
result = add product zero
return result
```

## After optimization

```text
result = const 6.0
return result
```

The graph fell from 6 nodes to 2: fixed-point iteration one fused `mul` plus
`add` into `fma`, constant-folded the resulting arithmetic, and removed the
now-dead inputs. Iteration two made no changes.

## Pass statistics

| Iteration | Pass | Changed | Nodes before | Nodes after |
| --- | --- | --- | ---: | ---: |
| 1 | common_subexpression_elimination | no | 6 | 6 |
| 1 | fuse_multiply_add | yes | 6 | 5 |
| 1 | algebraic_simplify | no | 5 | 5 |
| 1 | constant_propagate | no | 5 | 5 |
| 1 | constant_fold | yes | 5 | 5 |
| 1 | eliminate_dead_code | yes | 5 | 2 |
| 2 | common_subexpression_elimination | no | 2 | 2 |
| 2 | fuse_multiply_add | no | 2 | 2 |
| 2 | algebraic_simplify | no | 2 | 2 |
| 2 | constant_propagate | no | 2 | 2 |
| 2 | constant_fold | no | 2 | 2 |
| 2 | eliminate_dead_code | no | 2 | 2 |
