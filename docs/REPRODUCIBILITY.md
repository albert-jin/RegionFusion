# Reproduction notes

## Release scope

This snapshot contains the dense-fusion student, VOC/COCO training and evaluation entry points, complete dataset lists and image-level labels, attribute descriptors, and the C-RADIO/OpenCLIP sources needed by the student. The model code always constructs the dense-fusion branch. Omitting `--teacher-dir` switches the segmentation targets to online ExCEL pseudo labels; it does not reconstruct the historical B0 model.

The original supplied package requires externally generated teacher masks. It does not contain the exact SAM + semantic-seed + reliability-gating pipeline that produced T1/T2/F3 teacher caches. Consequently this release supports student training with supplied teacher targets and evaluation with supplied checkpoints; it is not an end-to-end regeneration of the recorded experiments from raw images alone. The unrelated CorrCLIP/CropFormer pipeline in the source bundle was not substituted for that missing teacher pipeline.

## External assets

| Asset | Required location | Purpose |
| --- | --- | --- |
| CLIP ViT-B/16 | `shared/ViT-B-16.pt` | Frozen image/text encoder |
| C-RADIOv4 SO400M | `shared/region_teacher/C_RADIOv4_SO400M.safetensors` | Frozen dense visual branch |
| VOC / COCO images and validation masks | See root README | Training images and evaluation labels |
| Offline teacher masks | `--teacher-dir` | Segmentation supervision, one PNG per training ID |
| Trained student checkpoint | `--checkpoint` | Final evaluation |

The expected C-RADIO weight SHA-256 is:

```text
23e0c117de49d4ce909150fe6658d470829e6639647c7a5b035ce82e0d5b763c
```

The loader intentionally requires this specific file. The CLIP loader uses the upstream download URL and checksum embedded in `RegionFusion/clip/clip.py`. No trained student checkpoint or teacher-cache download URL was provided with the source bundle.

## Recorded teacher settings

These settings describe the teacher targets used in the supplied experimental records; the corresponding exact teacher builder is not part of this snapshot.

| Component | Setting |
| --- | --- |
| Seed connected components | 8-connected, at least 100 pixels; at most 3 components per image/class |
| Region bank | At most 1,024 descriptors per class; exclude the query image's prototypes |
| Retrieval | Top 25; class votes, similarity used to break ties |
| SAM proposals | ViT-H, 32 points per side, predicted IoU ≥ 0.88, stability ≥ 0.95 |
| View consistency | Same foreground class on original and horizontally flipped views |
| Minimum view vote fraction | 0.60 |
| Minimum view cosine similarity | 0.50 |
| Same-class seed support in eroded region core | ≥ 0.50 |
| Conflicting foreground seed fraction | ≤ 0.15 |

Accepted regions may fill seed background or same-class pixels while preserving conflicting foreground seed labels. Rejected regions retain the original seed. T1 used retrieval-only targets; T2 and F3 used gated targets.

## Evaluation protocols

VOC final evaluation uses all 1,449 validation images unless `--limit` is supplied. At scale 1.0 it uses unflipped logits; scales 0.7, 1.2 and 1.5 average original/flip logits. Logits are averaged before the raw argmax and CRF. This preserves the supplied evaluation protocol.

COCO averages flips at all four scales, interpolates logits to 0.2× the original height/width for aggregation, then restores the original size. COCO masks must already use contiguous 0–80 indices; raw COCO category IDs are not accepted as segmentation labels.

Both paths use DenseCRF with 10 iterations, positional `(xy_std=1, weight=3)` and bilateral `(xy_std=67, rgb_std=3, weight=4)` terms. Image-level class tags do not mask final predictions. Ignore label is 255.

The original COCO manifest uses the entire 40,137-image validation list for `val_internal.txt` as well. Periodic internal validation can therefore be expensive. The configuration helper verifies all required paths and creates fresh absolute deadlines; run it again before a later evaluation.

## Results and environment

`results.json` retains the supplied VOC fixed-128 records and their separately labeled `full1449_cached_reference` values. The root README uses only the latter. It contains no new evaluation from this packaging task.

Historical B0/T1/T2 used timm 0.6.12; F3 used timm 1.0.22 and an additional frozen encoder. Only seed 0 is reported. The source snapshot ships the F3 architecture, not runnable copies of all historical ablations.

The recorded runtime was Linux/Python 3.11.12/PyTorch 2.9.1/CUDA 12.8. The full pip snapshot includes OS-specific and notebook packages. `requirements.txt` separates the student dependencies; a clean Linux GPU installation still needs to be verified. Neither training nor GPU inference was rerun when preparing this release.
