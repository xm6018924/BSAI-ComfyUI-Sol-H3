"""Derive a separately identified dynamic tile graph; never edit source ONNX.

Only external input/output dimension declarations change. Every operator,
constant, initializer and external-data descriptor remains byte-identical.
This is NOT execution qualification; a new engine and native comparisons are
required, including T1 position IDs and small spatial tiles.
"""
import argparse
import hashlib
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from trt_vae_build import MODEL_SHA, WEIGHTS_SHA, digest_file, write_new_json  # noqa: E402


def derive(model):
    import copy
    import onnx
    candidate = copy.deepcopy(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1 or model.graph.value_info:
        raise ValueError("Unexpected decoder I/O or cached internal shape declarations")
    inp, out = candidate.graph.input[0], candidate.graph.output[0]
    if inp.name != "latent_tile" or out.name != "pixel_tile":
        raise ValueError("Unexpected tile names")
    if inp.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16 or out.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
        raise ValueError("Expected FP16 I/O")
    if [d.dim_value for d in inp.type.tensor_type.shape.dim][1:] != [24,7,16,16]:
        raise ValueError("Unexpected source decoder declared geometry")
    if len(out.type.tensor_type.shape.dim) != 5:
        raise ValueError("Unexpected decoder output rank")
    for value, dims in ((inp, [1,24,"tile_t","tile_h","tile_w"]),
                        (out, [1,3,"raw_frames","pixel_h","pixel_w"])):
        for dim, size in zip(value.type.tensor_type.shape.dim, dims):
            dim.Clear()
            if isinstance(size, int):
                dim.dim_value = size
            else:
                dim.dim_param = size
    # Restoring ONLY I/O proves all other protobuf fields unchanged, including
    # shape arithmetic and RoPE Range operators and every learned parameter.
    restored = copy.deepcopy(candidate)
    restored.graph.input[0].CopyFrom(model.graph.input[0])
    restored.graph.output[0].CopyFrom(model.graph.output[0])
    if restored.SerializeToString() != model.SerializeToString():
        raise ValueError("Derivation modified more than external shape annotations")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if output.exists() or args.report.exists() or output.parent != source.parent:
        raise ValueError("New graph alongside original external data required")
    if digest_file(source) != MODEL_SHA:
        raise ValueError("Only pinned audited source decoder may be derived")
    data = source.with_suffix(".onnx.data")
    if digest_file(data) != WEIGHTS_SHA:
        raise ValueError("Decoder external weights changed")
    import onnx
    model = onnx.load(source, load_external_data=False)
    candidate = derive(model)
    with output.open("xb") as stream:
        stream.write(candidate.SerializeToString())
    onnx.checker.check_model(str(output))
    write_new_json(args.report, {"status": "annotation_only_flex_graph_not_execution_qualified",
                   "source_sha256": MODEL_SHA, "weights_sha256": WEIGHTS_SHA,
                   "derived_sha256": digest_file(output), "derived_path": str(output),
                   "nodes": len(model.graph.node), "initializers": len(model.graph.initializer),
                   "operators_sha256": hashlib.sha256(b"".join(n.SerializeToString() for n in model.graph.node)).hexdigest(),
                   "source_operators_and_weights_unchanged": True,
                   "limits": "Only I/O declarations changed; actual profile support requires new TRT compilation and native output comparison. No padding, new weights or universal-dimension claim."})
    print(digest_file(output))


if __name__ == "__main__":
    main()
