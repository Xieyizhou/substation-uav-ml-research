# Third-party sources and dependency notices

The root [MIT license](LICENSE) applies to project-authored code. It does not
replace licenses for dependencies, upstream model weights, datasets, simulator
assets or the software used to build and run this project. This document is a
source-preview inventory, not a license grant for external artifacts.

## Project origin and included material

- This project derives from
  [Xieyizhou/uav-path-planning-demo](https://github.com/Xieyizhou/uav-path-planning-demo).
  The predecessor's MIT notice is retained; [PROVENANCE.md](PROVENANCE.md)
  records the initial source snapshot.
- `archive/research_scripts/` preserves this project's historical scripts and
  checksum manifest. Moving a file there does not change its license or the
  identities of historical evidence.
- `docs/assets/research_preview_desktop.png` is a capture of this project's
  desktop interface from the 2026-09-25 acceptance run. It is not a photograph
  from an external substation dataset. The other documentation images show
  project UI or generated route plots.
- The tracked `simulation/models/x500_research/model.sdf` extends the external
  `model://x500` vehicle through an SDF include. The PX4 vehicle model and
  simulator binaries are supplied by the user's separate PX4/Gazebo installation;
  they are not relicensed by the project MIT notice.

## Separately installed software

The requirements files declare installation dependencies, not vendored copies.
The links below point to upstream license sources. Exact dependency versions
and complete bundled notices should be taken from the installed distribution
when packaging or redistributing an environment.

| Component | Use | Upstream license source |
| --- | --- | --- |
| MAVSDK-Python | Flight interface; core uses 3.17.2 | [BSD-3-Clause](https://github.com/mavlink/MAVSDK-Python/blob/main/LICENSE.txt) |
| PX4 Autopilot | External SITL runtime and vehicle assets | [BSD-3-Clause and included third-party notices](https://github.com/PX4/PX4-Autopilot/blob/main/LICENSE) |
| Gazebo Sim | External simulation runtime | [Apache-2.0](https://github.com/gazebosim/gz-sim/blob/main/LICENSE) |
| Gazebo rendering / Ogre Next | Optional native research probes | [gz-rendering Apache-2.0](https://github.com/gazebosim/gz-rendering/blob/main/LICENSE), [Ogre Next MIT and third-party notices](https://github.com/OGRECave/ogre-next/blob/master/COPYING) |
| Ultralytics | Optional YOLO training and inference | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| PyTorch | Optional training runtime | [License and bundled third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| NumPy / SciPy / pandas | Numeric and data analysis | [NumPy BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [SciPy BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt), [pandas BSD-3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| Matplotlib | Plots and route previews | [Matplotlib license](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| PyYAML | Configuration parsing | [MIT](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| Pillow | Image processing | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| OpenCV Python wheels | Image and recording utilities | [Wheel and bundled-library notices](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt) |
| psutil | Process accounting | [BSD-3-Clause](https://github.com/giampaolo/psutil/blob/master/LICENSE) |
| ONNX | Optional model export | [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE) |
| ONNX Runtime / ONNX Script / ONNX Slim | Optional inference and export tools | [Runtime MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE), [Script MIT](https://github.com/microsoft/onnxscript/blob/main/LICENSE), [Slim MIT](https://github.com/inisis/OnnxSlim/blob/main/LICENSE) |
| gRPC / Protocol Buffers | Flight and simulator message transport | [gRPC Apache-2.0](https://github.com/grpc/grpc/blob/master/LICENSE), [Protobuf BSD-3-Clause](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE) |

The optional Ultralytics integration is **not an MIT-only software stack**.
Its upstream AGPL terms continue to apply. The project does not distribute an
Ultralytics enterprise license or grant additional rights to upstream pretrained
weights. A combined redistribution or hosted deployment must satisfy the terms
of the components it actually includes; this table does not establish an
exemption based on optional installation or separate directories.

The native macOS target uses Apple SDKs and Swift system frameworks. It has no
third-party Swift package dependency in the current `Package.swift`. Python,
PX4/Gazebo and ML packages are not bundled into the native Demo application.

## Datasets, weights and historical evidence

Raw external datasets, downloaded previews, trained weights, checkpoints and
large evidence archives are excluded from the source distribution. A source
registry entry is not permission to redistribute its images.

- [Source registry](config/perception/real_domain_sources.json): upstream URLs,
  declared licenses, review status and pinned download metadata.
- [Real-domain intake policy](docs/REAL_DOMAIN_V3_SOURCES.md): the approved
  Gomes/Zenodo source declares CC BY 4.0; other listed sources remain restricted
  by their evaluation-only or quarantine status and provenance review.
- [Official Gomes dataset record](https://doi.org/10.5281/zenodo.7884270): use its
  attribution and license information if you obtain and redistribute the data.
- Frozen local receipts retain original paths as provenance. Those references
  are not downloadable assets and do not extend the project license to them.

Preserve each upstream notice when you redistribute its material. Record the
source and license before adding any new third-party code, image or model to Git.
