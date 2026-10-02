"""Strict named-I/O TRT tile execution. Imported without TensorRT or CUDA work.

The owner must verify the engine bundle and device before constructing a runner,
and must close it before another GPU task. Initial profile is the pinned T7 tile;
other profiles require their own qualified exports, not shape reinterpretation.
"""

import threading

from .trt_vae_build import (
    INPUT_SHAPE, OUTPUT_SHAPE, ENCODER_INPUT_SHAPE, ENCODER_OUTPUT_SHAPE,
    ENCODER_T1_INPUT_SHAPE, ENCODER_T1_OUTPUT_SHAPE,
    flex_output_shape, DECODER_FLEX_MIN_SHAPE,
)


def io_contract(kind="decoder", profile="default"):
    if profile == "flex" and kind == "decoder":
        return "latent_tile", "pixel_tile", INPUT_SHAPE, OUTPUT_SHAPE
    if profile == "t1" and kind == "encoder":
        return "pixel_tile", "moments_tile", ENCODER_T1_INPUT_SHAPE, ENCODER_T1_OUTPUT_SHAPE
    if profile != "default":
        raise ValueError("Unknown or incompatible engine profile")
    if kind == "decoder":
        return "latent_tile", "pixel_tile", INPUT_SHAPE, OUTPUT_SHAPE
    if kind == "encoder":
        return "pixel_tile", "moments_tile", ENCODER_INPUT_SHAPE, ENCODER_OUTPUT_SHAPE
    raise ValueError("Unknown engine kind")


def resolve_io(engine, context, trt, input_shape, *, kind="decoder", profile="default"):
    input_name, output_name, expected_input, expected_output = io_contract(kind, profile)
    if profile == "flex":
        expected_input, expected_output = tuple(input_shape), flex_output_shape(input_shape)
        if tuple(map(tuple, engine.get_tensor_profile_shape(input_name, 0))) != (DECODER_FLEX_MIN_SHAPE, INPUT_SHAPE, INPUT_SHAPE):
            raise ValueError("Flex engine profile does not match the bounded graph contract")
    if tuple(input_shape) != expected_input:
        raise ValueError("Tile requires a different, matching engine profile")
    if engine.num_io_tensors != 2:
        raise ValueError("Expected exactly one engine input and output")
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    if names != {input_name, output_name}:
        raise ValueError("Unexpected engine tensor names")
    for name, mode in ((input_name, trt.TensorIOMode.INPUT), (output_name, trt.TensorIOMode.OUTPUT)):
        if (engine.get_tensor_mode(name) != mode or engine.get_tensor_dtype(name) != trt.float16
                or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE):
            raise ValueError("Unexpected engine tensor mode, dtype or memory location")
    if not context.set_input_shape(input_name, expected_input):
        raise RuntimeError("Engine rejected input shape")
    actual = tuple(context.get_tensor_shape(output_name))
    if actual != expected_output:
        raise RuntimeError(f"Resolved output shape mismatch: {actual}")
    return actual


def enqueue_checked(context, input_pointer, output_pointer, stream_pointer, *, kind="decoder"):
    input_name, output_name, _, _ = io_contract(kind)
    for pointer in (input_pointer, output_pointer, stream_pointer):
        if type(pointer) is not int or pointer <= 0:
            raise ValueError("Expected nonzero device/stream addresses")
    if not context.set_tensor_address(input_name, input_pointer):
        raise RuntimeError("Failed to bind engine input")
    if not context.set_tensor_address(output_name, output_pointer):
        raise RuntimeError("Failed to bind engine output")
    if not context.execute_async_v3(stream_handle=stream_pointer):
        raise RuntimeError("TensorRT execution submission failed")


class TileEngineRunner:
    def __init__(self, engine, context, trt, *, device, kind="decoder", profile="default"):
        import torch

        self.device = torch.device(device)
        self.kind = kind
        self.profile = profile
        _, _, self.input_shape, _ = io_contract(kind, profile)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Explicit CUDA device index required")
        self.engine, self.context, self.trt = engine, context, trt
        self.lock = threading.Lock()
        self.failed, self.closed, self.calls = False, False, 0
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        resolve_io(engine, context, trt, self.input_shape, kind=self.kind, profile=self.profile)

    def __call__(self, source):
        import torch

        if self.closed or self.failed:
            raise RuntimeError("Engine runner is closed or previously failed")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Engine context is already in use")
        try:
            if not isinstance(source, torch.Tensor):
                raise ValueError("Engine tile must be a tensor")
            if self.profile == "flex":
                flex_output_shape(tuple(source.shape))
            elif tuple(source.shape) != self.input_shape:
                raise ValueError("Unsupported engine input tile shape")
            if not source.is_floating_point():
                raise ValueError("Engine input tile must be floating point")
            if source.device.type == "cuda" and source.device != self.device:
                raise ValueError("Source tensor is on a different GPU")
            with torch.inference_mode(), torch.cuda.device(self.device):
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.stream):
                    value = source.to(device=self.device, dtype=torch.float16).contiguous()
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Nonfinite engine input")
                    shape = resolve_io(self.engine, self.context, self.trt, value.shape, kind=self.kind, profile=self.profile)
                    # Allocate only AFTER runtime output dimensions are verified.
                    output = torch.empty(shape, dtype=torch.float16, device=self.device)
                    enqueue_checked(self.context, value.data_ptr(), output.data_ptr(), int(self.stream.cuda_stream), kind=self.kind)
                self.stream.synchronize()
                if not bool(torch.isfinite(output).all()):
                    raise RuntimeError("Engine returned NaN or Inf")
            self.calls += 1
            return output
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def close(self):
        if self.closed:
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close an engine while execution is active")
        try:
            self.closed = True
            try:
                self.stream.synchronize()
            finally:
                self.context = self.engine = self.stream = None
        finally:
            self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
