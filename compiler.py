"""A tiny computation-graph optimizer and interpreter."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Union

Program = List[dict]
Value = Union[bool, float]


def validate(program: Iterable[dict]) -> None:
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


def run(program: Iterable[dict]) -> Value:
    program = list(program)
    validate(program)
    values: Dict[str, Value] = {}
    for node in program:
        op = node["op"]
        if op == "const":
            values[node["out"]] = node["value"] if isinstance(node["value"], bool) else float(node["value"])
        elif op in {"add", "sub", "mul", "div", "eq", "lt", "gt"}:
            left, right = (values[name] for name in node["args"])
            if op == "add":
                values[node["out"]] = left + right
            elif op == "sub":
                values[node["out"]] = left - right
            elif op == "mul":
                values[node["out"]] = left * right
            elif op == "eq":
                values[node["out"]] = left == right
            elif op == "lt":
                values[node["out"]] = left < right
            elif op == "gt":
                values[node["out"]] = left > right
            else:
                values[node["out"]] = left / right
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
            constants[node["out"]] = node["value"] if isinstance(node["value"], bool) else float(node["value"])
        elif node["op"] in {"add", "sub", "mul", "div", "eq", "lt", "gt"} and all(name in constants for name in node["args"]):
            left, right = (constants[name] for name in node["args"])
            if node["op"] == "add":
                value = left + right
            elif node["op"] == "sub":
                value = left - right
            elif node["op"] == "mul":
                value = left * right
            elif node["op"] == "eq":
                value = left == right
            elif node["op"] == "lt":
                value = left < right
            elif node["op"] == "gt":
                value = left > right
            else:
                value = left / right
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
