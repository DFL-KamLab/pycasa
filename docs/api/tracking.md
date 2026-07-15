# API: Tracking

Tracking links frame-level detections over time into trajectories so downstream motility and visualization steps can operate on motion continuity rather than isolated per-frame points.

pycasa implements three interchangeable tracking backends, each with its own trade-offs:

| Backend | Algorithm | Best for |
|---|---|---|
| `sort` | Kalman filter + Hungarian IoU assignment ([Bewley et al. 2016](https://arxiv.org/abs/1602.00763)) | Fast, simple, robust default. |
| `deepsort` | SORT + appearance embeddings ([Wojke et al. 2017](https://github.com/nwojke/deep_sort)) | Same as SORT when appearance is uninformative; supports custom ReID features. |
| `jpdaf` | Joint Probabilistic Data Association Filter ([Urbano et al. 2017](https://doi.org/10.1109/TMI.2016.2630720)) | Dense, occluded sperm fields with overlapping trajectories. |

All three write to the standard `casa["tracks"][backend][source]` schema, so downstream motility and visualization steps work identically regardless of which one you choose. Calling a new backend clears `casa["tracks"]` and a yellow warning is emitted to flag the overwrite.

!!! note "This page is generated from the code"
    Signatures, parameters, and descriptions below are rendered directly from the
    `pycasa` source docstrings, so they always match the installed version.

---

::: pycasa.casa.tracking.tracking_wrapper._SessionTrackingNamespace.sort

---

::: pycasa.casa.tracking.tracking_wrapper._SessionTrackingNamespace.deepsort

---

::: pycasa.casa.tracking.tracking_wrapper._SessionTrackingNamespace.jpdaf

---

## Citations

**SORT** in pycasa is adapted from the original work by Alex Bewley:

> Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). **Simple Online and Realtime Tracking.** *IEEE International Conference on Image Processing (ICIP)*. [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)

Source code: [https://github.com/abewley/sort](https://github.com/abewley/sort) — licensed under GPL-3.0.

**Modification note:** The original implementation does not handle frames with zero detections, which causes a crash during the IoU matrix computation step. pycasa adds a two-line early-return guard in `_iou_batch` that returns a correctly-shaped zero matrix when either input is empty. If the upstream repository is updated to include this fix, the intent is to remove the local copy and have users pull from the original repo directly.

**DeepSORT** uses the original implementation by Nicolai Wojke, auto-cloned on first use:

> Wojke, N., Bewley, A., & Paulus, D. (2017). **Simple Online and Realtime Tracking with a Deep Association Metric.** *IEEE International Conference on Image Processing (ICIP)*. [arXiv:1703.07402](https://arxiv.org/abs/1703.07402)

Source code: [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort).

**JPDAF** implements the algorithm and parameter values from:

> Urbano, L.F., Masson, P., VerMilyea, M., & Kam, M. (2017). **Automatic Tracking and Motility Analysis of Human Sperm in Time-Lapse Images.** *IEEE Transactions on Medical Imaging*, 36(3), 792–801. [DOI:10.1109/TMI.2016.2630720](https://doi.org/10.1109/TMI.2016.2630720)
