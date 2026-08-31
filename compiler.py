"""A tiny computation-graph optimizer and interpreter."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

Program = List[dict]


def run(program: Iterable[dict]) -> float:
    values: Dict[str, float] = {}
    for node in program:
        op = node["op"]
        if op == "const":
            values[node["out"]] = float(node["value"])
        elif op in {"add", "mul"}:
            left, right = (values[name] for name in node["args"])
            values[node["out"]] = left + right if op == "add" else left * right
        elif op == "return":
            return values[node["args"][0]]
        else:
            raise ValueError(f"unknown op: {op}")
    raise ValueError("program has no return")


def constant_fold(program: Iterable[dict]) -> Program:
    constants: Dict[str, float] = {}
    output: Program = []
    for original in program:
        node = dict(original)
        if node["op"] == "const":
            constants[node["out"]] = float(node["value"])
        elif node["op"] in {"add", "mul"} and all(name in constants for name in node["args"]):
            left, right = (constants[name] for name in node["args"])
            value = left + right if node["op"] == "add" else left * right
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

