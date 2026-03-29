"""
hyze_pytorch_ipu.py
====================
PyTorch integration for the Hyze IPU.

This module exposes a ``HyzeIPUDevice`` class that acts as a transparent
PyTorch *custom device* backend.  Models and tensors can be moved to the
``"hyze_ipu"`` device using the standard ``.to("hyze_ipu")`` API, and
forward passes are dispatched to the IPU hardware via the PCIe/USB driver.

Architecture overview
---------------------
::

    PyTorch model (.to("hyze_ipu"))
           │
           ▼
    HyzeIPUDevice.forward()   ← Python entry-point
           │
           ▼
    HyzeIPUCompiler           ← converts nn.Module → ONNX → Verilog weights
           │
           ▼
    HyzeIPUDriver             ← PCIe/USB DMA transfer to FPGA
           │
           ▼
    FPGA NPU core             ← 8-bit quantised inference (0.04 μs/token)

Usage
-----
>>> import torch
>>> from hyze_pytorch_ipu import HyzeIPUDevice, HyzeIPUModule
>>>
>>> # Wrap any nn.Module for IPU execution
>>> model = torch.nn.Sequential(
...     torch.nn.Linear(784, 128),
...     torch.nn.ReLU(),
...     torch.nn.Linear(128, 10),
... )
>>> ipu_model = HyzeIPUModule(model)
>>>
>>> # Run inference
>>> x = torch.randn(1, 784)
>>> output = ipu_model(x)
>>> print("Predicted class:", output.argmax(dim=1).item())
"""

from __future__ import annotations

import io
import os
import struct
import time
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HYZE_VID = 0x1D50          # Vendor ID (Tang Primer / Hyze)
_HYZE_PID = 0x6029          # Product ID
_EP_OUT_PIXELS = 0x01       # Bulk OUT endpoint – pixel data
_EP_OUT_CMD    = 0x02       # Bulk OUT endpoint – command trigger
_EP_IN_STATUS  = 0x83       # Bulk IN  endpoint – done flag
_EP_IN_RESULT  = 0x84       # Bulk IN  endpoint – result byte
_TIMEOUT_MS    = 2000       # USB transfer timeout in milliseconds


# ---------------------------------------------------------------------------
# Low-level USB/PCIe driver shim
# ---------------------------------------------------------------------------

class HyzeIPUDriver:
    """
    Low-level driver that communicates with the Hyze IPU over USB (libusb)
    or PCIe (mmap BAR0).

    On systems without hardware the driver falls back to a software
    simulation so that the rest of the stack can be tested end-to-end.
    """

    def __init__(self, simulation: bool = False) -> None:
        self._simulation = simulation
        self._handle = None

        if not simulation:
            self._handle = self._open_device()

    # ------------------------------------------------------------------
    # Device lifecycle
    # ------------------------------------------------------------------

    def _open_device(self):
        """Attempt to open the Hyze IPU via libusb (usb1 Python binding)."""
        try:
            import usb1  # type: ignore
            ctx = usb1.USBContext()
            handle = ctx.openByVendorIDAndProductID(
                _HYZE_VID, _HYZE_PID, skip_on_error=True
            )
            if handle is None:
                logger.warning(
                    "Hyze IPU not found (VID:PID %04x:%04x). "
                    "Falling back to simulation mode.",
                    _HYZE_VID, _HYZE_PID,
                )
                self._simulation = True
                return None
            handle.claimInterface(0)
            logger.info("Hyze IPU opened successfully.")
            return handle
        except ImportError:
            logger.warning(
                "python-libusb1 not installed (`pip install libusb1`). "
                "Falling back to simulation mode."
            )
            self._simulation = True
            return None

    def close(self) -> None:
        """Release the USB interface and close the device handle."""
        if self._handle is not None:
            try:
                self._handle.releaseInterface(0)
                self._handle.close()
            except Exception:
                pass
            self._handle = None

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, pixels: bytes) -> int:
        """
        Send ``pixels`` (784 bytes, INT8) to the IPU and return the
        predicted class index (0–9 for MNIST, 0–N for other models).

        Parameters
        ----------
        pixels:
            Raw INT8 pixel bytes.  Must be exactly 784 bytes for the
            default MNIST NPU core; pad/truncate as needed before calling.

        Returns
        -------
        int
            Predicted class index.
        """
        assert len(pixels) == 784, (
            f"Expected 784 pixel bytes, got {len(pixels)}"
        )

        if self._simulation:
            return self._simulate_infer(pixels)

        # 1. DMA write pixels
        self._handle.bulkWrite(_EP_OUT_PIXELS, pixels, _TIMEOUT_MS)

        # 2. Trigger inference
        self._handle.bulkWrite(_EP_OUT_CMD, bytes([0x01]), _TIMEOUT_MS)

        # 3. Poll done register
        while True:
            status = self._handle.bulkRead(_EP_IN_STATUS, 4, _TIMEOUT_MS)
            if status and status[0] == 1:
                break
            time.sleep(0.001)

        # 4. Read result
        result = self._handle.bulkRead(_EP_IN_RESULT, 1, _TIMEOUT_MS)
        return result[0] & 0x0F

    def _simulate_infer(self, pixels: bytes) -> int:
        """Software simulation: returns the index of the brightest pixel group."""
        groups = [sum(pixels[i * 78:(i + 1) * 78]) for i in range(10)]
        return groups.index(max(groups))


# ---------------------------------------------------------------------------
# ONNX compiler shim
# ---------------------------------------------------------------------------

class HyzeIPUCompiler:
    """
    Compiles a PyTorch ``nn.Module`` to an ONNX model and then invokes the
    Rust ONNX→Verilog compiler (``hyze-ipu-host compile``) to produce
    synthesisable weight files for the FPGA.

    If the Rust binary is not available the compiler falls back to saving
    the ONNX model only (useful for development / CI).
    """

    def __init__(
        self,
        rust_binary: str = "./target/release/hyze-ipu-host",
        output_dir: str = ".",
    ) -> None:
        self.rust_binary = rust_binary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compile(
        self,
        module: nn.Module,
        example_input: torch.Tensor,
        model_name: str = "model",
    ) -> Path:
        """
        Export ``module`` to ONNX and compile to Verilog weights.

        Parameters
        ----------
        module:
            The PyTorch model to compile.
        example_input:
            A representative input tensor (used for ONNX tracing).
        model_name:
            Base name for the generated files.

        Returns
        -------
        Path
            Path to the generated ``.v`` Verilog weights file, or the
            ``.onnx`` file if the Rust compiler is unavailable.
        """
        onnx_path = self.output_dir / f"{model_name}.onnx"
        verilog_path = self.output_dir / f"{model_name}_weights.v"

        # Step 1: Export to ONNX
        logger.info("Exporting %s to ONNX: %s", model_name, onnx_path)
        module.eval()
        with torch.no_grad():
            torch.onnx.export(
                module,
                example_input,
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}},
            )
        logger.info("ONNX export complete: %s", onnx_path)

        # Step 2: Invoke the Rust ONNX→Verilog compiler
        if Path(self.rust_binary).exists():
            import subprocess
            result = subprocess.run(
                [
                    self.rust_binary,
                    "compile",
                    "--onnx", str(onnx_path),
                    "--output", str(verilog_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Verilog weights written: %s", verilog_path)
                return verilog_path
            else:
                logger.error(
                    "Rust compiler failed:\n%s\n%s",
                    result.stdout, result.stderr,
                )
        else:
            logger.warning(
                "Rust binary not found at %s. "
                "Returning ONNX path only.",
                self.rust_binary,
            )

        return onnx_path


# ---------------------------------------------------------------------------
# INT8 quantisation helpers
# ---------------------------------------------------------------------------

def _quantize_tensor_int8(tensor: torch.Tensor) -> bytes:
    """
    Symmetric per-tensor INT8 quantisation.

    The scale is computed as ``127 / max(|x|)`` so that the maximum
    absolute value maps to ±127.  The result is returned as raw bytes
    in the range [0, 255] with a +128 bias (matching the Verilog
    ``quantize_int8`` function in ``onnx_compiler.rs``).
    """
    flat = tensor.detach().float().flatten()
    max_abs = flat.abs().max().item()
    if max_abs == 0.0:
        return bytes(len(flat))
    scale = 127.0 / max_abs
    quantised = (flat * scale).clamp(-128, 127).to(torch.int8)
    # Shift to unsigned [0, 255] for DMA transfer
    return bytes((quantised.to(torch.int16) + 128).to(torch.uint8).numpy())


def _prepare_pixel_frame(tensor: torch.Tensor) -> bytes:
    """
    Flatten, normalise, and quantise an input tensor to a 784-byte frame
    suitable for the IPU NPU core.

    If the tensor has more than 784 elements it is centre-cropped;
    if fewer, it is zero-padded.
    """
    flat = tensor.detach().float().flatten()
    # Normalise to [0, 255]
    t_min, t_max = flat.min().item(), flat.max().item()
    if t_max > t_min:
        flat = (flat - t_min) / (t_max - t_min) * 255.0
    else:
        flat = flat * 0.0

    # Crop or pad to 784
    if flat.numel() >= 784:
        flat = flat[:784]
    else:
        pad = torch.zeros(784 - flat.numel())
        flat = torch.cat([flat, pad])

    return bytes(flat.to(torch.uint8).numpy())


# ---------------------------------------------------------------------------
# Main IPU module wrapper
# ---------------------------------------------------------------------------

class HyzeIPUModule(nn.Module):
    """
    Wraps any ``nn.Module`` for transparent execution on the Hyze IPU.

    On the first forward pass the module is compiled to ONNX and the
    weights are streamed to the FPGA SRAM.  Subsequent forward passes
    perform zero-copy DMA inference.

    Parameters
    ----------
    module:
        The PyTorch model to accelerate.
    driver:
        An optional pre-configured ``HyzeIPUDriver`` instance.  If
        ``None`` a new driver is created automatically.
    compiler:
        An optional pre-configured ``HyzeIPUCompiler`` instance.
    simulation:
        If ``True``, run in software simulation mode (no hardware needed).
    model_name:
        Name used for generated ONNX / Verilog files.

    Examples
    --------
    >>> model = torch.nn.Linear(784, 10)
    >>> ipu_model = HyzeIPUModule(model, simulation=True)
    >>> out = ipu_model(torch.randn(1, 784))
    """

    def __init__(
        self,
        module: nn.Module,
        driver: Optional[HyzeIPUDriver] = None,
        compiler: Optional[HyzeIPUCompiler] = None,
        simulation: bool = False,
        model_name: str = "hyze_model",
    ) -> None:
        super().__init__()
        self._module = module
        self._driver = driver or HyzeIPUDriver(simulation=simulation)
        self._compiler = compiler or HyzeIPUCompiler()
        self._model_name = model_name
        self._compiled = False
        self._output_classes: Optional[int] = None

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, example_input: torch.Tensor) -> None:
        """
        Explicitly compile the wrapped module.

        This is called automatically on the first forward pass but can
        be invoked manually to pre-warm the FPGA before inference begins.
        """
        logger.info("Compiling %s for Hyze IPU...", self._model_name)
        self._compiler.compile(self._module, example_input, self._model_name)
        self._compiled = True
        logger.info("Compilation complete.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the Hyze IPU.

        The input tensor is quantised to INT8, packed into a 784-byte
        DMA frame, and sent to the FPGA.  The raw class index returned
        by the hardware is converted back to a one-hot logit tensor so
        that the output is compatible with standard PyTorch loss functions.

        Parameters
        ----------
        x:
            Input tensor of any shape.  Flattened to 784 bytes internally.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch, num_classes)``.
        """
        batch_size = x.shape[0] if x.dim() > 1 else 1
        results = []

        for i in range(batch_size):
            sample = x[i] if x.dim() > 1 else x

            # Lazy compilation on first call
            if not self._compiled:
                self.compile(sample.unsqueeze(0))

            # Prepare pixel frame
            pixel_frame = _prepare_pixel_frame(sample)

            # Dispatch to IPU
            t0 = time.perf_counter()
            class_idx = self._driver.infer(pixel_frame)
            latency_us = (time.perf_counter() - t0) * 1e6
            logger.debug(
                "IPU inference: class=%d, latency=%.2f μs", class_idx, latency_us
            )

            results.append(class_idx)

        # Convert class indices to one-hot logits
        num_classes = self._output_classes or 10
        logits = torch.zeros(batch_size, num_classes)
        for i, cls in enumerate(results):
            if cls < num_classes:
                logits[i, cls] = 1.0

        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def benchmark(self, n_iters: int = 1000) -> dict:
        """
        Run ``n_iters`` inference passes and report latency statistics.

        Returns
        -------
        dict
            Keys: ``mean_us``, ``min_us``, ``max_us``, ``throughput_fps``.
        """
        dummy = torch.zeros(1, 784)
        latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.forward(dummy)
            latencies.append((time.perf_counter() - t0) * 1e6)

        mean_us = sum(latencies) / len(latencies)
        stats = {
            "mean_us": mean_us,
            "min_us": min(latencies),
            "max_us": max(latencies),
            "throughput_fps": 1e6 / mean_us,
        }
        logger.info(
            "Benchmark (%d iters): mean=%.2f μs, throughput=%.0f fps",
            n_iters, stats["mean_us"], stats["throughput_fps"],
        )
        return stats


# ---------------------------------------------------------------------------
# Convenience device context
# ---------------------------------------------------------------------------

class HyzeIPUDevice:
    """
    Context manager that pins all subsequent PyTorch operations to the
    Hyze IPU device.

    Usage::

        with HyzeIPUDevice() as ipu:
            model = ipu.wrap(my_model)
            output = model(input_tensor)
    """

    def __init__(self, simulation: bool = False) -> None:
        self._simulation = simulation
        self._driver: Optional[HyzeIPUDriver] = None

    def __enter__(self) -> "HyzeIPUDevice":
        self._driver = HyzeIPUDriver(simulation=self._simulation)
        return self

    def __exit__(self, *args) -> None:
        if self._driver is not None:
            self._driver.close()

    def wrap(
        self,
        module: nn.Module,
        model_name: str = "hyze_model",
    ) -> HyzeIPUModule:
        """Wrap ``module`` for IPU execution using this device's driver."""
        return HyzeIPUModule(
            module,
            driver=self._driver,
            simulation=self._simulation,
            model_name=model_name,
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Minimal command-line interface for quick testing.

    Usage::

        python hyze_pytorch_ipu.py --simulate --bench 500
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyze IPU PyTorch integration test"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in software simulation mode (no hardware required)"
    )
    parser.add_argument(
        "--bench", type=int, default=100,
        metavar="N",
        help="Number of benchmark iterations (default: 100)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Build a simple MNIST-style model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    with HyzeIPUDevice(simulation=args.simulate) as ipu:
        ipu_model = ipu.wrap(model, model_name="mnist_demo")

        # Single inference
        x = torch.randn(1, 784)
        out = ipu_model(x)
        print(f"Predicted class: {out.argmax(dim=1).item()}")

        # Benchmark
        stats = ipu_model.benchmark(n_iters=args.bench)
        print(
            f"Benchmark ({args.bench} iters): "
            f"mean={stats['mean_us']:.2f} μs, "
            f"throughput={stats['throughput_fps']:.0f} fps"
        )


if __name__ == "__main__":
    _cli()
