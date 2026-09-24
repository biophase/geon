# Point-cloud blend benchmark

Measured on 2026-09-24 with VTK 9.6.0 and NVIDIA RTX A5000, using
`benchmark_pointcloud_blend.py`: seeded random clouds, full resolution,
1000 × 800 offscreen window, 2-pixel points, multisampling disabled,
10 warm-up frames and 60 measured frames per mode. Each frame waits for GPU
completion. These measurements exclude Qt event processing and display/vsync.

| Points | Mode | Median ms | P95 ms |
| ---: | --- | ---: | ---: |
| 1,000,000 | Ordinary | 0.615 | 0.859 |
| 1,000,000 | Fixed blend | 0.650 | 0.882 |
| 1,000,000 | Moving blend | 0.655 | 0.836 |
| 3,000,000 | Ordinary | 1.410 | 1.731 |
| 3,000,000 | Fixed blend | 1.417 | 1.659 |
| 3,000,000 | Moving blend | 1.386 | 1.572 |

Vertex-buffer upload timestamps were unchanged throughout every measured run,
including positions, foreground colors, and background colors. Timing differences
at this scale include measurement noise; this is not an application FPS guarantee.

Run with the checkout on `PYTHONPATH`:

```text
python scripts/benchmark_pointcloud_blend.py --points 1000000 3000000 --frames 60
```

The local `geon_dev` environment has an editable-package redirect that takes
precedence over `PYTHONPATH`. For these measurements and the test run, that finder
was removed from `sys.meta_path` in the invocation only, ensuring imports came
from this checkout without changing the environment installation.
