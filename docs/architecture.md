# Architecture

This document describes the design of the CV foundations pipeline: how the modules under `src/` fit together, why each model was chosen, and how this stage is meant to hand off to `02-vision-embeddings-agent`.

## Pipeline overview

```
                         Raw image
                             |
                             v
                 preprocessing/transforms.py
                             |
              +--------------+--------------+
              |                             |
              v                             v
   inference/classifier.py       inference/detector.py
      (ResNet, ViT)                  (YOLOv11/v12)
              |                             |
              |                             | boxes as prompt
              |                             v
              |                  inference/segmenter.py
              |                 (SAM2/SAM3, promptable)
              |                             |
              +--------------+--------------+
                             |
                             v
                  utils/visualization.py
              (overlay labels / boxes / masks)
```

A request enters as a raw image, is normalized/augmented by `preprocessing/transforms.py`, and then flows into one or more inference modules. The detector's output (bounding boxes) can act as a *prompt* into the segmenter, rather than the segmenter running independently — this is the key difference from a classic segmentation-head design. `utils/visualization.py` renders whatever combination of classifications, boxes, and masks was produced.

## Module responsibilities

| Module | Responsibility | Model |
|---|---|---|
| `inference/classifier.py` | Whole-image / crop classification | ResNet, ViT |
| `inference/detector.py` | Bounding-box object detection | YOLOv11/v12 |
| `inference/segmenter.py` | Promptable, zero-shot segmentation | SAM2/SAM3 |
| `preprocessing/transforms.py` | Resize, normalize, augment — model-agnostic | — |
| `utils/visualization.py` | Overlay boxes/masks/labels for inspection | — |

## Why these models

- **YOLOv11/v12** over YOLOv8: the repo intentionally avoids pinning to an aging version; detector.py should target whatever the current YOLO release is rather than hardcoding v8-specific APIs.
- **SAM2/SAM3** over a classic segmentation head (e.g. U-Net, Mask R-CNN): promptable, zero-shot segmentation is now the default way to get masks without task-specific training, and it's a better demonstration of "modern tooling" than a purpose-trained head.

## Interfaces / contracts

This is the part most likely to surprise someone extending the code, since it's not derivable from empty stub files:

- **classifier**: `image -> (label, confidence)`. Standard, no prompt required.
- **detector**: `image -> [(box, label, confidence), ...]`. Standard, no prompt required.
- **segmenter**: `(image, prompt) -> [mask, ...]`, where `prompt` is one of a point, a box, or (with SAM3) a text label — **not** just an image. Callers must supply a prompt; the detector is one natural source of box prompts, but a caller can also supply points/text directly without running the detector first.

## Extension point for Stage 2

`02-vision-embeddings-agent` is expected to sit downstream of this pipeline, consuming its outputs (labels, boxes, masks) and turning them into embeddings for semantic search/reasoning — rather than operating on raw pixels directly. Keeping the inference modules' outputs structured (not just visualizations) is what makes that handoff possible.
