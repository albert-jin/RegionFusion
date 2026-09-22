# Source packaging notes

The release was prepared from the supplied `RegionFusion_full_data_code` snapshot and `RegionFusion_Paper_Figures_v2/figures_4k` exports. The original directories were not modified.

## Included

- Student model, losses, decoder, training loop, evaluation, dataset loaders and utilities.
- Complete VOC/COCO split lists and image-level label dictionaries, attribute text and tokenizer assets.
- C-RADIO and bundled OpenCLIP runtime source/configuration, including existing license/header notices.
- Resource setup helpers, dependency reference, source checksums and five 4K figures.

## Packaging fixes

- Moved the six required entry-point/helper files from `auto_res/scripts` and `auto_res/coco_scripts` to `experiments/voc` and `experiments/coco`, keeping their directory depth for relative imports.
- Redirected generated run records to repository-root `outputs/`.
- Removed the COCO internal checkpoint review dependency/flag and unused VOC experimental configuration switches. Ordinary training, checkpoint recording and wall-clock limits remain.
- Replaced the resource preparer with source verification and external-asset linking; it no longer creates a nested Git repository.
- Made COCO setup create its output directory on a fresh clone.
- Pointed the model's default VOC descriptor argument to the included JSON file, allowing first-use evaluation without a pre-existing attribute cache.
- Required an explicit VOC dataset location instead of a machine-specific default.
- Added the existing CLIP BPE vocabulary to bundled OpenCLIP, whose tokenizer expects the same vocabulary filename locally.
- Regenerated source checksums for the published layout and added exclusions for generated artifacts.

## Excluded

- Internal `work/coco_review_control.py`, stale runtime configuration under `outputs`, automatic-search/development orchestration, and old checksum manifests for the previous directory layout.
- The unused duplicate `affutils copy.py` and machine-specific `utils/reload.py` checkpoint helper.
- Unrelated ModuSeg CorrCLIP/CropFormer demo, training and teacher pipelines; the C-RADIO/OpenCLIP runtime used by the released student is retained.
- Dataset imagery, pixel-label datasets, teacher caches, pretrained and trained weights, virtual environments, logs, presentation decks and raw visualization tensors.

The shipped source checksum manifest covers Python source, runtime configuration, split metadata and tokenizer assets. It is intended to validate this snapshot before local modifications.
