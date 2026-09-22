# Figure gallery

The five PNG files are byte-for-byte copies of the supplied `figures_4k` exports, each at 3840×2160. They are embedded in the root README without recomputing or redrawing predictions.

| File | Content |
| --- | --- |
| `01_framework.png` | Offline region supervision and dense student architecture, with real VOC example insets |
| `04_voc_full_comparison.png` | Recorded full VOC validation results, 1,449 images |
| `17a_segmentation_comparison.png` | Aeroplane and sheep examples, B0/T1/T2/F3 with CRF |
| `18_gate_ablation_crops.png` | Complete teacher-gate bundle comparison, T1 versus T2 |
| `22_dense_feature_queries.png` | Spatial-query cosine similarities and RADIO PCA |

The framework diagram combines an AI-assisted schematic with real model-output insets. The prediction, response and feature panels come from the supplied model visualizations. PCA colors are nonsemantic. Cosine similarity is a feature relationship, not a probability of correct segmentation.

Qualitative sample IDs are `2007_000033`, `2007_000129`, `2007_000175` and `2007_000187` from VOC validation. These illustrative images do not constitute a new dataset-wide evaluation. Pale ground-truth pixels denote ignored labels. Crop/query choices use ground truth for display only, independently of model prediction errors; ground truth is not fed into inference.

F3 uses an additional external encoder and a different timm runtime from B0/T1/T2. Numeric values for the comparison plot are retained in [results.json](../docs/results.json).
