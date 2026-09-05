"""A tiny computation-graph optimizer and interpreter."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

Program = List[dict]
Value = Union[bool, float, list]
Metadata = Tuple[Tuple[int, ...], str]


def _const_metadata(node: dict) -> Metadata:
    value = node["value"]
    shape, dtype = _value_metadata(value)
    if node.get("dtype", dtype) != dtype:
        raise ValueError(f"constant dtype does not match value: {node['out']}")
    declared_shape = tuple(node.get("shape", shape))
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in declared_shape):
        raise ValueError(f"invalid shape: {node['out']}")
    if declared_shape != shape:
        raise ValueError(f"constant shape does not match value: {node['out']}")
    return shape, dtype


def _value_metadata(value: object) -> Metadata:
    if isinstance(value, bool):
        return (), "bool"
    if isinstance(value, (int, float)):
        return (), "float"
    if not isinstance(value, list) or not value:
        raise ValueError("constant must be a scalar or non-empty rectangular tensor")
    child_metadata = [_value_metadata(child) for child in value]
    if len(set(child_metadata)) != 1:
        raise ValueError("tensor constant must be rectangular with one dtype")
    child_shape, dtype = child_metadata[0]
    return (len(value),) + child_shape, dtype


def _constant_value(value: object) -> Value:
    if isinstance(value, list):
        return [_constant_value(child) for child in value]
    return value if isinstance(value, bool) else float(value)


def _broadcast_shape(left: Tuple[int, ...], right: Tuple[int, ...]) -> Tuple[int, ...]:
    result = []
    for left_size, right_size in zip(reversed(left), reversed(right)):
        if left_size != right_size and left_size != 1 and right_size != 1:
            raise ValueError("inputs cannot be broadcast together")
        result.append(max(left_size, right_size))
    longer = left if len(left) > len(right) else right
    return longer[:abs(len(left) - len(right))] + tuple(reversed(result))


def _elementwise(left: Value, right: Value, operation):
    left_shape, _ = _value_metadata(left)
    right_shape, _ = _value_metadata(right)
    shape = _broadcast_shape(left_shape, right_shape)

    def value_at(value, value_shape, coordinates):
        for size, coordinate in zip(value_shape, coordinates[-len(value_shape):]):
            value = value[0 if size == 1 else coordinate]
        return value

    def build(coordinates=()):
        if len(coordinates) == len(shape):
            return operation(value_at(left, left_shape, coordinates), value_at(right, right_shape, coordinates))
        return [build(coordinates + (index,)) for index in range(shape[len(coordinates)])]

    return build()


def _apply_binary(op: str, left: Value, right: Value) -> Value:
    operations = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "eq": lambda a, b: a == b,
        "lt": lambda a, b: a < b,
        "gt": lambda a, b: a > b,
    }
    return _elementwise(left, right, operations[op])


def infer_metadata(program: Iterable[dict]) -> Dict[str, Metadata]:
    """Return each SSA value's ``(shape, dtype)`` metadata."""
    metadata: Dict[str, Metadata] = {}
    for node in program:
        op = node["op"]
        if op == "const":
            metadata[node["out"]] = _const_metadata(node)
        elif op in {"add", "sub", "mul", "div"}:
            left, right = (metadata[name] for name in node["args"])
            if left[1] != right[1] or left[1] != "float":
                raise ValueError(f"arithmetic requires float inputs: {node['out']}")
            metadata[node["out"]] = _broadcast_shape(left[0], right[0]), "float"
        elif op in {"eq", "lt", "gt"}:
            left, right = (metadata[name] for name in node["args"])
            if left[1] != right[1]:
                raise ValueError(f"comparison requires matching inputs: {node['out']}")
            metadata[node["out"]] = _broadcast_shape(left[0], right[0]), "bool"
        elif op == "select":
            condition, when_true, when_false = (metadata[name] for name in node["args"])
            if condition != ((), "bool") or when_true != when_false:
                raise ValueError(f"select requires a scalar boolean and matching choices: {node['out']}")
            metadata[node["out"]] = when_true
    return metadata


def validate(program: Iterable[dict]) -> None:
    program = list(program)
    defined = set()
    for node in program:
        for name in node.get("args", []):
            if name not in defined:
                raise ValueError(f"use before definition: {name}")
        output = node.get("out")
        if output is not None:
            if output in defined:
                raise ValueError(f"duplicate output: {output}")
            defined.add(output)
    infer_metadata(program)


def run(program: Iterable[dict]) -> Value:
    program = list(program)
    validate(program)
    values: Dict[str, Value] = {}
    for node in program:
        op = node["op"]
        if op == "const":
            values[node["out"]] = _constant_value(node["value"])
        elif op in {"add", "sub", "mul", "div", "eq", "lt", "gt"}:
            left, right = (values[name] for name in node["args"])
            values[node["out"]] = _apply_binary(op, left, right)
        elif op == "select":
            condition, when_true, when_false = (values[name] for name in node["args"])
            values[node["out"]] = when_true if condition else when_false
        elif op == "return":
            return values[node["args"][0]]
        else:
            raise ValueError(f"unknown op: {op}")
    raise ValueError("program has no return")


def constant_fold(program: Iterable[dict]) -> Program:
    constants: Dict[str, Value] = {}
    output: Program = []
    for original in program:
        node = dict(original)
        if node["op"] == "const":
            constants[node["out"]] = _constant_value(node["value"])
        elif node["op"] in {"add", "sub", "mul", "div", "eq", "lt", "gt"} and all(name in constants for name in node["args"]):
            left, right = (constants[name] for name in node["args"])
            value = _apply_binary(node["op"], left, right)
            node = {"op": "const", "out": node["out"], "value": value}
            constants[node["out"]] = value
        elif node["op"] == "select" and all(name in constants for name in node["args"]):
            condition, when_true, when_false = (constants[name] for name in node["args"])
            value = when_true if condition else when_false
            node = {"op": "const", "out": node["out"], "value": value}
            constants[node["out"]] = value
        output.append(node)
    return output


def eliminate_dead_code(program: Iterable[dict]) -> Program:
    nodes = list(program)
    live = set()
    kept = []
    for node in reversed(nodes):
        if node["op"] == "return":
            live.update(node["args"])
            kept.append(node)
        elif node.get("out") in live:
            live.discard(node["out"])
            live.update(node.get("args", []))
            kept.append(node)
    return list(reversed(kept))


def optimize(program: Iterable[dict]) -> Program:
    program = list(program)
    validate(program)
    return eliminate_dead_code(constant_fold(program))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    args = parser.parse_args()
    program = json.loads(args.program.read_text(encoding="utf-8"))
    optimized = optimize(program)
    print(json.dumps({"result": run(optimized), "program": optimized}, indent=2))


if __name__ == "__main__":
    main()
