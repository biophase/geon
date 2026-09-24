"""Run from the checkout with PYTHONPATH=src; prints synchronized GPU frame times."""

import argparse
import json
from time import perf_counter

import numpy as np
import vtk

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer


def benchmark(count, frames):
    rng = np.random.default_rng(42)
    data = PointCloudData(rng.random((count, 3), dtype=np.float32))
    data.add_field("foreground", rng.integers(0, 256, (count, 3), dtype=np.uint8), FieldType.COLOR)
    data.add_field("background", rng.integers(0, 256, (count, 3), dtype=np.uint8), FieldType.COLOR)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(1000, 800)
    window.AddRenderer(renderer)
    layer = PointCloudLayer(data)
    layer.attach(renderer)
    layer.update()
    renderer.ResetCamera()

    def render():
        window.Render()
        window.WaitForCompletion()
        assert layer._main_actor.GetMapper() is layer._mapper_fine

    def upload_times():
        vbos = layer._mapper_fine.GetVBOs()
        return {name: vbos.GetVBO(name).GetUploadTime().GetMTime()
                for name in ("vertexMC", "scalarColor", "geonBackgroundRGB")
                if vbos.GetVBO(name) is not None}

    result = {"points": count, "frames": frames, "vtk": vtk.vtkVersion.GetVTKVersion()}
    try:
        for mode in ("ordinary", "fixed_blend", "moving_blend"):
            if mode == "fixed_blend":
                layer.set_background_field_name("background")
            for _ in range(10):
                render()
            before = upload_times()
            times = []
            for i in range(frames):
                start = perf_counter()
                if mode == "moving_blend":
                    layer.set_foreground_mix((i % 101) / 100)
                render()
                times.append((perf_counter() - start) * 1000)
            after = upload_times()
            result[mode] = {"median_ms": round(float(np.median(times)), 3),
                            "p95_ms": round(float(np.percentile(times, 95)), 3),
                            "vbo_uploads_unchanged": before == after}
            assert before == after, "Unexpected point/color upload during steady-state rendering"
        result["gpu"] = [line.strip() for line in window.ReportCapabilities().splitlines()
                         if "OpenGL renderer" in line or "OpenGL vendor" in line]
    finally:
        layer.detach()
        window.Finalize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, nargs="+", default=[1_000_000, 3_000_000])
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    for count in args.points:
        print(json.dumps(benchmark(count, args.frames)), flush=True)
