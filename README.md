# 🧠 Brain Tumor Detection & Segmentation — BraTS 2020

An end-to-end medical imaging pipeline that takes raw multi-modal brain MRI scans (BraTS 2020 dataset), **classifies** whether a tumor is present, **segments** the tumor region, **quantifies** its size/severity, **explains** the model's focus area (Grad-CAM), and **auto-generates** a structured clinical report — built as an AI research/decision-support prototype, not a diagnostic device.

---

## 1. Objective

Manually reviewing brain MRI volumes slice-by-slice to detect and measure a tumor is time-consuming and requires specialist expertise. The objective of this project is to build a research pipeline that automates the full radiological workflow on a single MRI slice:

1. **Detect** — classify a slice as containing a tumor or not (CNN).
2. **Segment** — outline the exact tumor region (U-Net), not just flag its presence.
3. **Quantify** — estimate the tumor's area, volume, shape, and equivalent diameter.
4. **Triage** — assign a 5-tier severity/risk level based on tumor size.
5. **Explain** — visualize which regions of the scan drove the model's prediction (Grad-CAM), for model transparency.
6. **Report** — generate a structured, human-readable clinical summary with a recommendation, ready to be reviewed by a radiologist/neurologist.

**Dataset:** [BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) (Multimodal Brain Tumor Segmentation Challenge, MICCAI) — 369 patients, each with 4 co-registered MRI modalities plus an expert-annotated tumor segmentation mask.

---

## 2. Variables

### 2.1 Raw inputs (per patient, NIfTI volumes)

| Variable | Meaning |
|---|---|
| `flair` | FLAIR MRI modality — primary modality used in this pipeline |
| `t1` | T1-weighted MRI modality |
| `t1ce` | T1-weighted post-contrast (gadolinium-enhanced) MRI modality |
| `t2` | T2-weighted MRI modality |
| `seg` | Expert ground-truth segmentation volume (label mask) |

### 2.2 Segmentation label legend (from `seg`)

| Label | Meaning |
|---|---|
| `0` | Background |
| `1` | Necrotic core / non-enhancing tumor (NCR/NET) |
| `2` | Peritumoral edema (ED) |
| `4` | GD-enhancing tumor (ET) |

All non-zero labels are merged into a single binary **Whole Tumor (WT)** mask (`label > 0 → 1`), which is the segmentation target used throughout this project.

### 2.3 Engineered / derived variables

| Variable | How it's produced | Purpose |
|---|---|---|
| `y_labels` | `1` if the WT mask for a slice contains any tumor pixels, else `0` | Binary classification target for the CNN |
| `X_images` | FLAIR axial slice (index 77, the typical peak tumor slice), percentile-clipped (1st–99th) and normalized to `[0,1]`, resized to 128×128 | CNN / U-Net input |
| `y_masks` | WT binary mask, resized to 128×128 (nearest-neighbor) | U-Net segmentation target |
| `area_mm2`, `volume_mm3` | Pixel/voxel count × spacing (BraTS voxel spacing ≈ 1 mm³ isotropic) | Tumor size quantification |
| `equivalent_diameter_mm`, `circularity`, `perimeter_mm`, `bounding_box` | Derived from the predicted mask's contour (OpenCV) | Tumor shape/morphology description |
| `size_classification` / `risk_level` | 5-tier rule based on `area_mm2` (see §3.5) | Clinical severity triage |

---

## 3. Methodology

### 3.1 Data loading & preprocessing
A custom `BraTS2020Loader` class reads the 5 NIfTI volumes per patient, extracts one axial slice (`slice_idx = 77`), clips intensities to the 1st–99th percentile to suppress outlier voxels, normalizes to `[0,1]`, and resizes to 128×128. Of 369 patient folders, **368 loaded successfully** (1 failed/skipped). The loader also supports multi-slice sampling (`slice_range=(55,100)`, step 5) for richer training data, though the single-slice mode was used for the models trained here.

### 3.2 Exploratory Data Analysis
Multi-modal visualization (FLAIR / T1 / T1ce / T2 / segmentation side-by-side) for sample tumor patients, class distribution, and tumor-area distribution were plotted. Key finding: the dataset is **imbalanced at the slice level** — **327 tumor slices (88.9%) vs. 41 healthy slices (11.1%)** — because slice 77 sits near the typical tumor peak for BraTS volumes, so most patients show a tumor at that exact slice.

### 3.3 Train/Validation/Test split
A **patient-level stratified split** (avoiding data leakage between sets) produced:

| Split | Slices | Tumor | Healthy |
|---|---|---|---|
| Train | 249 | 221 | 28 |
| Validation | 45 | 40 | 5 |
| Test | 74 | 66 | 8 |

### 3.4 CNN tumor classifier
A custom 3-block CNN (32 → 64 → 128 filters, each block: Conv2D ×2 + BatchNorm + MaxPool + Dropout) feeding a dense head (256 → 128 → 1, sigmoid), **8,711,649 parameters**. Compiled with Adam (`lr=0.001`), binary cross-entropy loss, and tracked accuracy/precision/recall/AUC. Trained with on-the-fly augmentation (rotation, shift, flip, zoom) for up to 50 epochs with early stopping on `val_auc` (patience 12), `ReduceLROnPlateau`, and checkpointing — training stopped early at **epoch 13**, restoring weights from the best epoch.

### 3.5 U-Net tumor segmentation
A standard encoder–decoder U-Net (32→64→128→256→512 channels at the bottleneck, with skip connections), **7,771,297 parameters**, trained to predict the binary Whole-Tumor mask. Loss = **BCE + Dice loss** (`bce_dice_loss`), tracked with Dice coefficient, IoU (Jaccard), and binary accuracy. A class weight (`tumor ≈ 33.3× background`) was applied to counter the foreground/background pixel imbalance (tumors occupy a small fraction of each 128×128 slice). Trained for the full 60 epochs with early stopping on `val_dice_coefficient` (patience 20) — validation Dice climbed steadily from ~0.06 to **~0.79** by the final epochs.

### 3.6 Tumor area estimation & 5-tier severity analysis
A `TumorAnalyzer` class converts each predicted binary mask into clinical measurements (area, estimated volume, perimeter, circularity, equivalent diameter, bounding box), then assigns a severity tier using tumor area:

| Area (mm²) | Size class | Risk level |
|---|---|---|
| < 200 | Micro | Minimal Risk |
| 200–500 | Small | Low Risk |
| 500–1200 | Medium | Moderate Risk |
| 1200–2500 | Large | High Risk |
| ≥ 2500 | Very Large | Critical Risk |

### 3.7 Explainability — Grad-CAM
Grad-CAM is computed on a mid-encoder convolutional layer of the U-Net, highlighting which spatial regions most influenced the predicted tumor mask. Heatmaps are overlaid on the FLAIR input (masked to the brain region) and compared side-by-side with the ground-truth mask, for both tumor and healthy test cases.

### 3.8 End-to-end pipeline & clinical report generation
A `BrainTumorDetectionPipeline` class chains the two models together: **CNN classifies → if tumor detected, U-Net segments → `TumorAnalyzer` quantifies → a structured clinical report is generated.** The report includes patient ID, classification confidence, whole-tumor characteristics (area/volume/diameter/circularity), severity assessment, and a risk-tiered recommendation:

| Risk level | Recommendation |
|---|---|
| Critical Risk | 🔴 URGENT: Immediate neurosurgical consult |
| High Risk | 🟠 HIGH PRIORITY: Neurosurgical consultation |
| Moderate Risk | 🟡 Follow-up MRI in 3–6 months recommended |
| Low Risk | 🟢 Monitor with regular imaging (6–12 months) |
| Minimal Risk | ⚪ Annual surveillance MRI |

Every report ends with an explicit disclaimer that this is AI-assisted analysis requiring confirmation by a qualified radiologist/neurologist.

### 3.9 Diagnostic dashboard
A single multi-panel figure combines the input FLAIR slice, classification verdict + confidence, predicted segmentation mask, tumor measurements, and severity tier into one visual summary per patient.

---

## 4. Results

### 4.1 CNN classifier (test set, 74 slices: 66 tumor / 8 healthy)

| Metric | Value |
|---|---|
| Accuracy | 89.19% |
| Precision (overall) | 89.19% |
| Recall (Tumor class) | 100.00% |
| Recall (Healthy class) | **0.00%** |
| ROC-AUC | 0.50–0.67 (see note below) |
| Parameters | 8,711,649 |

**Confusion matrix (test set):**

|  | Predicted Healthy | Predicted Tumor |
|---|---|---|
| **Actual Healthy** | 0 | 8 |
| **Actual Tumor** | 0 | 66 |

### 4.2 U-Net segmentation (test set)

| Metric | Value |
|---|---|
| Dice Coefficient (aggregate) | 0.8728 |
| IoU / Jaccard | 0.7744 |
| Per-sample Dice — mean ± std | 0.7804 ± 0.2749 |
| Per-sample Dice — min / max | 0.00 / 1.00 |
| Parameters | 7,771,297 |

### 4.3 Severity cohort analysis (20 predicted tumor masks sampled)
- Mean tumor area: **592.0 mm²**, median 612.5 mm², range 69–1175 mm²
- Risk distribution: mostly **Moderate Risk**, with smaller Minimal/Low Risk groups — no High or Critical Risk cases in this sample.

### 4.4 Saved artifacts
- `brain_tumor_classifier_cnn_brats2020.h5`
- `brain_tumor_segmentation_unet_brats2020.h5`

---

## 5. Conclusions

1. **U-Net segmentation is the strongest, most trustworthy component of this pipeline.** A Dice of 0.87 (aggregate) / ~0.78 (per-sample mean) and an IoU of 0.77 are solid results for a tumor segmentation model trained on only 249 training slices — the skip-connection architecture and Dice+BCE loss with class weighting successfully compensated for the tumor region being a small fraction of each image.

2. **The CNN classifier's headline "89% accuracy" is misleading and does not reflect real discriminative skill.** The confusion matrix shows the model predicts **"Tumor" for every single test case**, including all 8 actually-healthy slices (0% recall on the Healthy class). Because the test set is 89% tumor by construction (a side effect of always sampling slice 77, which is near the typical tumor peak), a trivial "always predict tumor" rule achieves the same 89% accuracy — the model has not learned to discriminate tumor from healthy tissue. This is confirmed by prediction probabilities on the test set clustering tightly around ~0.77 (std ≈ 0.0004) for every sample, and by an ROC-AUC hovering near the random-guess baseline (0.50–0.67 depending on the epoch measured).

3. **Root cause: severe class imbalance driven by the single-slice sampling strategy**, not by an inherent model limitation. Fixing the fixed `slice_idx=77` choice (e.g., using the `load_multi_slice` method already defined in the loader, or explicitly sampling more healthy slices per patient) would likely resolve this and is the single highest-priority next step.

4. **The severity/triage layer and clinical report generator work as designed** — they correctly propagate the U-Net's segmentation into clinically meaningful, human-readable outputs (area, volume, shape, tiered risk, and a recommendation), which is valuable output even while the classification stage needs rework.

5. **Grad-CAM explainability is integrated but should be re-validated** once the classifier is retrained on a balanced dataset — currently, since the CNN always predicts "tumor," its saliency maps are not yet meaningfully diagnostic.

6. **This is a research prototype, not a clinical tool.** As stated explicitly in every generated report, all outputs require confirmation by a qualified radiologist or neurologist before any clinical action is taken.

### Limitations / recommended next steps
- **Rebalance the classification dataset** — either sample more healthy (tumor-free) slices per patient, use multiple slices across the tumor-peak range with corresponding healthy slices, or apply class weighting/oversampling (analogous to what was already done for the U-Net's pixel-level imbalance).
- **Small sample size overall** (368 slices from single-slice-per-patient sampling) limits both training data and the statistical reliability of the reported metrics; using `load_multi_slice()` to draw multiple axial slices per patient would substantially increase data volume.
- **Only the FLAIR modality was used** for the CNN/U-Net inputs, even though all 4 modalities (FLAIR, T1, T1ce, T2) were loaded and explored — a multi-modal (multi-channel) input could improve segmentation and especially classification accuracy.
- **Whole Tumor (WT) is a coarse target** — BraTS provides finer-grained sub-region labels (necrotic core, edema, enhancing tumor) that could support tumor sub-type-specific severity grading rather than a single merged mask.
- **Per-sample Dice variance is high** (0.7804 ± 0.2749, with some 0.0 cases) — some slices segment perfectly while others fail completely; inspecting the "worst case" visualizations would help identify whether these are genuinely hard cases (very small/ambiguous tumors) or preprocessing artifacts.

---

## 6. Repository contents

| Component | Description |
|---|---|
| BraTS 2020 data loader (`BraTS2020Loader`) | NIfTI loading, percentile-clip normalization, single- and multi-slice sampling |
| CNN classifier | Tumor vs. healthy binary classification |
| U-Net | Whole-Tumor binary segmentation |
| `TumorAnalyzer` | Area/volume/shape quantification + 5-tier severity classification |
| `BrainTumorDetectionPipeline` | End-to-end orchestration: classify → segment → analyze → report |
| Grad-CAM module | Model explainability visualization |
| Diagnostic dashboard | Multi-panel per-patient summary figure |
| `brain_tumor_classifier_cnn_brats2020.h5` | Saved CNN model |
| `brain_tumor_segmentation_unet_brats2020.h5` | Saved U-Net model |

---

## 7. Tech stack
Python · TensorFlow / Keras · NiBabel (NIfTI I/O) · OpenCV · scikit-image · scikit-learn (metrics, splitting) · NumPy · Matplotlib / Seaborn · tqdm

---

## 8. Disclaimer
⚠️ This is an AI research tool only. It has **not** been validated for clinical use. All outputs (classification, segmentation, severity, and generated reports) must be reviewed and confirmed by a qualified radiologist or neurologist before informing any clinical decision.
