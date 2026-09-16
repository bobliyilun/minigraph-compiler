"""A tiny computation-graph optimizer and interpreter."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

Program = List[dict]
Value = Union[bool, float, list]
Metadata = Tuple[Tuple[int, ...], str]


def parse_textual_ir(text: str) -> Program:
    """Parse one SSA instruction per line (``x = add a b`` or ``return x``)."""
    program = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            if line.startswith("return "):
                node = {"op": "return", "args": [line.removeprefix("return ").strip()]}
            else:
                output, expression = (part.strip() for part in line.split("=", 1))
                op, arguments = expression.split(maxsplit=1)
                if op == "const":
                    decoder = json.JSONDecoder()
                    value, index = decoder.raw_decode(arguments.lstrip())
                    node = {"op": "const", "out": output, "value": value}
                    metadata = arguments.lstrip()[index:].strip()
                    while metadata:
                        key, value_text = metadata.split("=", 1)
                        key = key.strip()
                        value, index = decoder.raw_decode(value_text.lstrip())
                        if key not in {"dtype", "shape"}:
                            raise ValueError(f"unknown constant metadata: {key}")
                        node[key] = value
                        metadata = value_text.lstrip()[index:].strip()
                else:
                    node = {"op": op, "out": output, "args": arguments.split()}
            program.append(node)
        except (ValueError, IndexError) as error:
            raise ValueError(f"invalid textual IR on line {line_number}: {raw_line}") from error
    validate(program)
    return program


def print_textual_ir(program: Iterable[dict]) -> str:
    """Print a program in the format accepted by :func:`parse_textual_ir`."""
    lines = []
    for node in program:
        if node["op"] == "return":
            lines.append(f"return {node['args'][0]}")
        elif node["op"] == "const":
            metadata = "".join(f" {key}={json.dumps(node[key])}" for key in ("dtype", "shape") if key in node)
            lines.append(f"{node['out']} = const {json.dumps(node['value'])}{metadata}")
        else:
            lines.append(f"{node['out']} = {node['op']} {' '.join(node['args'])}")
    return "\n".join(lines)


def graphviz_export(program: Iterable[dict]) -> str:
    """Export a validated program as a Graphviz DOT dependency graph."""
    nodes = list(program)
    validate(nodes)
    lines = ["digraph program {", "  rankdir=LR;"]
    for index, node in enumerate(nodes):
        target = node.get("out", f"return_{index}")
        label = "return" if node["op"] == "return" else f"{target} = {node['op']}"
        lines.append(f"  {json.dumps(target)} [label={json.dumps(label)}];")
        lines.extend(f"  {json.dumps(argument)} -> {json.dumps(target)};" for argument in node.get("args", []))
    return "\n".join(lines + ["}"])


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
        elif op == "fma":
            left, right, addend = (metadata[name] for name in node["args"])
            if {left[1], right[1], addend[1]} != {"float"}:
                raise ValueError(f"arithmetic requires float inputs: {node['out']}")
            metadata[node["out"]] = _broadcast_shape(_broadcast_shape(left[0], right[0]), addend[0]), "float"
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
        elif op == "alias":
            metadata[node["out"]] = metadata[node["args"][0]]
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


def topological_sort(program: Iterable[dict]) -> Program:
    """Return a stable dependency order, rejecting cycles with their SSA path."""
    nodes = list(program)
    producers = {}
    for index, node in enumerate(nodes):
        output = node.get("out")
        if output is not None:
            if output in producers:
                raise ValueError(f"duplicate output: {output}")
            producers[output] = index

    dependencies = []
    for node in nodes:
        node_dependencies = []
        for name in node.get("args", []):
            if name not in producers:
                raise ValueError(f"use before definition: {name}")
            node_dependencies.append(producers[name])
        dependencies.append(node_dependencies)

    ordered = []
    visiting = []
    visited = set()

    def visit(index: int) -> None:
        if index in visited:
            return
        if index in visiting:
            start = visiting.index(index)
            cycle = visiting[start:] + [index]
            names = [nodes[item].get("out", "return") for item in cycle]
            raise ValueError(f"cycle detected: {' -> '.join(names)}")
        visiting.append(index)
        for dependency in dependencies[index]:
            visit(dependency)
        visiting.pop()
        visited.add(index)
        ordered.append(nodes[index])

    for index in range(len(nodes)):
        visit(index)
    validate(ordered)
    return ordered


def liveness_report(program: Iterable[dict]) -> List[dict]:
    """Return values live immediately before and after each instruction."""
    nodes = list(program)
    validate(nodes)
    live = set()
    report = []
    for node in reversed(nodes):
        live_after = sorted(live)
        output = node.get("out")
        if output is not None:
            live.discard(output)
        live.update(node.get("args", []))
        report.append({"live_in": sorted(live), "live_out": live_after})
    return list(reversed(report))


def memory_slot_reuse_analysis(program: Iterable[dict]) -> Dict[str, int]:
    """Assign reusable buffer slots to SSA values that remain live after a node."""
    nodes = list(program)
    report = liveness_report(nodes)
    slots: Dict[str, int] = {}
    free_slots = set()
    next_slot = 0
    for node, state in zip(nodes, report):
        live_out = set(state["live_out"])
        for name in set(state["live_in"]) - live_out:
            free_slots.add(slots[name])
        output = node.get("out")
        if output is not None and output in live_out:
            if free_slots:
                slots[output] = min(free_slots)
                free_slots.remove(slots[output])
            else:
                slots[output] = next_slot
                next_slot += 1
    return slots


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
        elif op == "fma":
            left, right, addend = (values[name] for name in node["args"])
            values[node["out"]] = _apply_binary("add", _apply_binary("mul", left, right), addend)
        elif op == "select":
            condition, when_true, when_false = (values[name] for name in node["args"])
            values[node["out"]] = when_true if condition else when_false
        elif op == "alias":
            values[node["out"]] = values[node["args"][0]]
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
        elif node["op"] == "fma" and all(name in constants for name in node["args"]):
            left, right, addend = (constants[name] for name in node["args"])
            value = _apply_binary("add", _apply_binary("mul", left, right), addend)
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


def constant_propagate(program: Iterable[dict]) -> Program:
    """Remove aliases, letting their constant sources reach later operations."""
    aliases: Dict[str, str] = {}
    output: Program = []
    for original in program:
        node = dict(original)
        if "args" in node:
            node["args"] = [aliases.get(name, name) for name in node["args"]]
        if node["op"] == "alias":
            aliases[node["out"]] = node["args"][0]
        else:
            output.append(node)
    return output


def common_subexpression_elimination(program: Iterable[dict]) -> Program:
    """Reuse the first output of each identical pure expression."""
    expressions: Dict[tuple, str] = {}
    aliases: Dict[str, str] = {}
    output: Program = []
    for original in program:
        node = dict(original)
        if "args" in node:
            node["args"] = [aliases.get(name, name) for name in node["args"]]
        if node["op"] == "return":
            output.append(node)
            continue
        key = node["op"], tuple(node.get("args", [])), repr(node.get("value"))
        if key in expressions:
            aliases[node["out"]] = expressions[key]
        else:
            expressions[key] = node["out"]
            output.append(node)
    return output


def fuse_multiply_add(program: Iterable[dict]) -> Program:
    """Fuse a single-use ``mul`` immediately consumed by an ``add`` into ``fma``."""
    nodes = list(program)
    uses = {}
    for node in nodes:
        for name in node.get("args", []):
            uses[name] = uses.get(name, 0) + 1
    output = []
    index = 0
    while index < len(nodes):
        node = nodes[index]
        following = nodes[index + 1] if index + 1 < len(nodes) else None
        product = node.get("out")
        if node["op"] == "mul" and uses.get(product) == 1 and following and following["op"] == "add" and product in following["args"]:
            addend = following["args"][1] if following["args"][0] == product else following["args"][0]
            output.append({"op": "fma", "out": following["out"], "args": [*node["args"], addend]})
            index += 2
        else:
            output.append(dict(node))
            index += 1
    return output


def algebraic_simplify(program: Iterable[dict]) -> Program:
    """Eliminate arithmetic identities whose constant operand is scalar."""
    constants: Dict[str, Value] = {}
    aliases: Dict[str, str] = {}
    output: Program = []
    for original in program:
        node = dict(original)
        if "args" in node:
            node["args"] = [aliases.get(name, name) for name in node["args"]]
        if node["op"] == "const":
            constants[node["out"]] = _constant_value(node["value"])
            output.append(node)
            continue
        if node["op"] in {"add", "sub", "mul", "div"}:
            left, right = node["args"]
            left_value, right_value = constants.get(left), constants.get(right)
            replacement = None
            if node["op"] == "add" and right_value == 0 or node["op"] == "sub" and right_value == 0:
                replacement = left
            elif node["op"] == "add" and left_value == 0 or node["op"] == "mul" and left_value == 1:
                replacement = right
            elif node["op"] == "mul" and right_value == 1 or node["op"] == "div" and right_value == 1:
                replacement = left
            if replacement is not None:
                aliases[node["out"]] = replacement
                continue
        output.append(node)
    return output


PASSES = (
    ("common_subexpression_elimination", common_subexpression_elimination),
    ("fuse_multiply_add", fuse_multiply_add),
    ("algebraic_simplify", algebraic_simplify),
    ("constant_propagate", constant_propagate),
    ("constant_fold", constant_fold),
    ("eliminate_dead_code", eliminate_dead_code),
)


def optimize_with_trace(program: Iterable[dict]) -> Tuple[Program, List[dict]]:
    """Optimize to a fixed point and report each pass's effect."""
    optimized = list(program)
    validate(optimized)
    trace = []
    iteration = 0
    while True:
        iteration += 1
        changed = False
        for name, optimization_pass in PASSES:
            updated = optimization_pass(optimized)
            pass_changed = updated != optimized
            trace.append({
                "iteration": iteration,
                "pass": name,
                "changed": pass_changed,
                "nodes_before": len(optimized),
                "nodes_after": len(updated),
            })
            changed |= pass_changed
            optimized = updated
        if not changed:
            return optimized, trace


def optimize(program: Iterable[dict]) -> Program:
    return optimize_with_trace(program)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    args = parser.parse_args()
    program = json.loads(args.program.read_text(encoding="utf-8"))
    optimized = optimize(program)
    print(json.dumps({"result": run(optimized), "program": optimized}, indent=2))


if __name__ == "__main__":
    main()
