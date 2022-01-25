import argparse
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import onnion_runtime as rt
import onnx
from onnx import numpy_helper

# 前提条件
# - サブグラフ内で利用するinitializerがちゃんとそのサブグラフ内にあること


class NameTable:
    def __init__(self, prefix: str) -> None:
        self.tbl: Dict[str, str] = dict()
        self.prefix = prefix

    def __getitem__(self, key: str) -> str:
        if not (key in self.tbl):
            self.tbl[key] = f"{self.prefix}{len(self.tbl)}"
        return self.tbl[key]

    def debug_info(self) -> str:
        info = "\n".join([f"    {k} -> {self.tbl[k]}" for k in self.tbl])
        return info


def embed_ndarray(arr: npt.NDArray[Any]) -> str:
    return f"np.array({arr.tolist()}, dtype=np.{arr.dtype}).reshape({arr.shape})"


def collect_subgraphs(graph: onnx.GraphProto) -> List[onnx.GraphProto]:
    subgraphs: List[onnx.GraphProto] = []
    for n in graph.node:
        subgraphs = subgraphs + [attr.g for attr in n.attribute if attr.HasField("g")]
    return subgraphs


def gen_init(initializer: Dict[str, npt.NDArray[Any]], sub_graphs: List[onnx.GraphProto], graph_name_table: NameTable) -> str:
    init_code = []
    init_code.append("self.initializer = dict()")
    for k in initializer:
        arr = initializer[k]
        init_code.append(f'self.initializer["{k}"] = {embed_ndarray(arr)}')
    init_code.append("")
    init_code.append("self.sub_graphs = dict()")
    for g in sub_graphs:
        sub_graph_name = graph_name_table[f"{id(g)}"]
        init_code.append(f'self.sub_graphs["{sub_graph_name}"] = {sub_graph_name}()')
    return "\n        ".join(init_code)


def gen_init_with_npy(
    initializer: Dict[str, npt.NDArray[Any]],
    sub_graphs: List[onnx.GraphProto],
    graph_name_table: NameTable,
    graph_name: str,
    export_tensor_size: int,
) -> str:
    npy_file = f"{graph_name}_tensors.npy"
    init_code = []
    init_code.append("self.initializer = dict()")
    init_code.append(f"with open('{npy_file}', 'rb') as f:")
    with open(f"{graph_name}_tensors.npy", "wb") as f:
        for k in initializer:
            arr = initializer[k]
            if arr.size < export_tensor_size:
                init_code.append(f'    self.initializer["{k}"] = {embed_ndarray(arr)}')
            else:
                np.save(f, arr)  # type: ignore
                init_code.append(f'    self.initializer["{k}"] = np.load(f)')
    init_code.append("")
    init_code.append("self.sub_graphs = dict()")
    for g in sub_graphs:
        sub_graph_name = graph_name_table[f"{id(g)}"]
        init_code.append(f'self.sub_graphs["{sub_graph_name}"] = {sub_graph_name}()')
    return "\n        ".join(init_code)


def gen_run_body(
    opset_version: int,
    graph: onnx.GraphProto,
    initializer: Dict[str, npt.NDArray[Any]],
    graph_name_table: NameTable,
    value_name_table: NameTable,
    force: bool,
) -> str:
    graph_body = []
    unsupported_operators = []
    for n in graph.node:
        outputs = []
        for o in n.output:
            if o == "":
                # Optional output
                # See: https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
                outputs.append("_")
            else:
                outputs.append(value_name_table[o])
        outputs.append("*_")
        output_names = ", ".join(outputs)

        inputs = []
        for i in n.input:
            if i == "":
                # Optional Input
                # See: https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
                # onnion-rt require None for optional inputs.
                inputs.append("None")
            elif i in initializer:
                inputs.append(f'self.initializer["{i}"]')
            else:
                inputs.append(value_name_table[i])
        input_names = ", ".join(inputs)
        op_args = [f"{opset_version}"]
        for a in n.attribute:
            if a.HasField("f"):
                op_args.append(f"{a.name}={a.f}")
            elif a.HasField("i"):
                op_args.append(f"{a.name}={a.i}")
            elif a.HasField("s"):
                op_args.append(f"{a.name}={a.s}")
            elif a.HasField("t"):
                v = numpy_helper.to_array(a.t)
                op_args.append(f"{a.name}={embed_ndarray(v)}")
            elif a.HasField("g"):
                sub_graph_name = graph_name_table[f"{id(a.g)}"]
                op_args.append(f'{a.name}=self.sub_graphs["{sub_graph_name}"]')
            elif len(a.floats) != 0:
                op_args.append(f"{a.name}={a.floats}")
            elif len(a.ints) != 0:
                op_args.append(f"{a.name}={a.ints}")
            elif len(a.strings) != 0:
                op_args.append(f"{a.name}={a.strings}")
            elif len(a.tensors) != 0:
                vs = [numpy_helper.to_array(t) for t in a.tensors]
                op_args.append(f"{a.name}={[embed_ndarray(v) for v in vs]}")
            elif len(a.graphs) != 0:
                sub_graph_names = [graph_name_table[f"{id(g)}"] for g in a.graphs]
                sub_graphs = [f'self.sub_graphs["{n}"]' for n in sub_graph_names]
                op_args.append(f"{a.name}={sub_graphs}")
            else:
                if force:
                    print(f"Warning: Ignore unknown attribute '{a.name}' in the node:\n{n}", file=sys.stderr)
                else:
                    sys.exit(f"Error: Not support the attributes in the node:\n{n}")
        args = ", ".join(op_args)

        if not rt.is_supported(n.op_type):
            unsupported_operators.append(n)

        w = rt.warning(n.op_type, opset_version)
        if w is not None:
            print(f"Warning: {w}", file=sys.stderr)

        graph_body.append(f"{output_names} = rt.{n.op_type}({args}).run({input_names}) # {n.name}")

    if len(unsupported_operators) != 0:
        if force:
            print("Warning: there are unsupported operators", file=sys.stderr)
        else:
            for n in unsupported_operators:
                print(n, file=sys.stderr)
            sys.exit("Error: there are unsupported operators")

    return "\n        ".join(graph_body)


def graph2pyclass(
    opset_version: int, graph: onnx.GraphProto, graph_name_table: NameTable, export_tensor_size: Optional[int], force: bool
) -> str:
    graph_name = graph_name_table[f"{id(graph)}"]

    sub_graphs: List[onnx.GraphProto] = collect_subgraphs(graph)
    sub_graph_code = "\n".join(
        [graph2pyclass(opset_version, g, graph_name_table, export_tensor_size, force) for g in sub_graphs]
    )

    value_name_table = NameTable("val")
    input_names = [value_name_table[v.name] for v in graph.input]
    output_names = [value_name_table[v.name] for v in graph.output]
    initializer = dict([(v.name, numpy_helper.to_array(v)) for v in graph.initializer])

    if export_tensor_size is None:
        init_code = gen_init(initializer, sub_graphs, graph_name_table)
    elif all([initializer[k].size < export_tensor_size for k in initializer]):
        init_code = gen_init(initializer, sub_graphs, graph_name_table)
    else:
        init_code = gen_init_with_npy(initializer, sub_graphs, graph_name_table, graph_name, export_tensor_size)

    graph_body = gen_run_body(opset_version, graph, initializer, graph_name_table, value_name_table, force)
    value_name_table_info = value_name_table.debug_info()

    code = f"""
{sub_graph_code}

class {graph_name}:
    '''
    Value name table: origin_name -> identifier
{value_name_table_info}
    '''
    def __init__(self):
        {init_code}

    def run(self, {', '.join(input_names)}):
        {graph_body}
        return [{', '.join(output_names)}]"""
    return code


def generate_code(model: onnx.ModelProto, export_tensor_size: Optional[int], force: bool) -> str:
    opset_version = model.opset_import[0].version
    graph_name_table = NameTable("Graph")
    main_graph = graph2pyclass(opset_version, model.graph, graph_name_table, export_tensor_size, force)
    code = f"""# Generated by onnion
import numpy as np
import onnion_runtime as rt

{main_graph}

def init_graph():
    return Graph0()
"""
    return code


def onnion(input_path: str, output_path: str, export_tensor_size: Optional[int], force: bool) -> None:
    model = onnx.load(input_path)

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        sys.exit(f"Error: {e}")

    code = generate_code(model, export_tensor_size, force)

    with open(output_path, mode="w") as f:
        f.write(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input ONNX path")
    parser.add_argument(
        "-o", "--output", dest="output_path", required=False, default="model.py", help="Output python code path"
    )
    parser.add_argument(
        "--export-tensor-size",
        dest="export_tensor_size",
        required=False,
        type=int,
        help="Export tensors to an npy file if the size of tensors is larger than the option value",
    )
    parser.add_argument(
        "--force",
        dest="force",
        required=False,
        action="store_true",
        help="Generate code even if there are unsupported operators",
    )
    args = parser.parse_args()
    onnion(args.input_path, output_path=args.output_path, export_tensor_size=args.export_tensor_size, force=args.force)


if __name__ == "__main__":
    main()
