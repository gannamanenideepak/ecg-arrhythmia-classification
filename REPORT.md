# ECG Beat Classification Project Report: MIT-BIH Arrhythmia Database

## 1. Executive Summary

This project implements an end-to-end, reproducible, CPU-friendly ECG beat classification pipeline conforming to the **Association for the Advancement of Medical Instrumentation (AAMI EC57)** standard and the **de Chazal inter-patient evaluation paradigm**. Using lead MLII from the MIT-BIH Arrhythmia Database, we extract 200-sample beat windows (at 360 Hz) along with rhythm context (4 RR timing features) and evaluate generalization performance on 22 unseen test patients (DS2, 49,697 beats).

We compare two models:
1. **1D CNN with RR Feature Fusion**: A compact deep learning model (11,989 parameters) combining hierarchical temporal convolutions with an RR timing projection head.
2. **Random Forest Baseline**: An ensemble of 100 decision trees trained on 32 handcrafted simple features (RR metrics, statistical moments, energy metrics, and downsampled morphological points).

Both models were trained strictly on DS1 patients with class-weighted loss functions to counteract severe class imbalance, without ever observing DS2 test patients.

---

## 2. Methodology & Experimental Setup

### 2.1 Database & Record Selection
The **MIT-BIH Arrhythmia Database** (PhysioNet) contains 48 half-hour two-lead ambulatory ECG recordings sampled at 360 Hz with 11-bit resolution over a 10 mV range.
Following the AAMI EC57 standard:
- **Paced records excluded**: `102`, `104`, `107`, and `217` are removed due to artificial pacing spikes and paced morphology.
- **Analyzed Lead**: Modified Lead II (MLII), which provides the clearest view of ventricular depolarization (QRS complex) and atrial activity (P wave) in 44 records. For record `114`, where channel order is swapped, the MLII channel was dynamically resolved by channel name.

### 2.2 AAMI EC57 Label Grouping
Individual MIT-BIH beat annotations are mapped to the 5 standard AAMI EC57 superclasses:

| AAMI Class | Superclass Name | Original MIT-BIH Annotation Symbols |
| :---: | :--- | :--- |
| **N** | Normal / Non-ectopic | `N` (Normal), `L` (Left BBB), `R` (Right BBB), `e` (Atrial escape), `j` (Nodal escape) |
| **S** | Supraventricular ectopic | `A` (Atrial premature), `a` (Aberrated atrial premature), `J` (Nodal premature), `S` (Supraventricular premature) |
| **V** | Ventricular ectopic | `V` (Premature ventricular contraction), `E` (Ventricular escape) |
| **F** | Fusion of ventricular and normal | `F` (Fusion beat) |
| **Q** | Unknown / Paced / Unclassifiable | `/` (Paced), `f` (Fusion of paced and normal), `Q` (Unclassifiable) |

Non-beat annotations (e.g. rhythm change markers `+`, noise `~`, comment `!`, etc.) were ignored.

### 2.3 Inter-Patient Split (de Chazal Benchmark)
To evaluate realistic clinical deployment where a model must diagnose patients it has never seen, we use the inter-patient division established by de Chazal et al. (2004):
- **DS1 Train (18 patients, 42,160 beats)**: `101`, `106`, `108`, `109`, `112`, `114`, `116`, `119`, `122`, `124`, `203`, `207`, `208`, `209`, `215`, `220`, `223`, `230`.
- **DS1 Validation (4 patients held out from DS1, 8,848 beats)**: `115`, `118`, `201`, `205`. Used for learning rate scheduling and checkpoint selection.
- **DS2 Test (22 patients, 49,697 beats)**: `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`. Kept completely untouched during training.

### 2.4 Preprocessing & Feature Engineering
1. **Beat Window Slicing**: For each annotated R-peak at sample index $r$, a window of 200 samples is extracted: 90 samples before the peak ($250\text{ ms}$, covering PR interval and P wave) and 110 samples after the peak ($305\text{ ms}$, covering ST segment and T wave).
2. **Beat Normalization**: Each cut beat waveform is normalized using per-beat z-score standardization:
   $$\hat{x}[n] = \frac{x[n] - \mu_x}{\sigma_x + 10^{-8}}$$
   This eliminates patient-specific DC baseline offsets and electrode contact variations.
3. **RR Interval Features (4 timing descriptors)**:
   - $RR_{\text{prev}}$: Interval from previous beat to current beat (seconds).
   - $RR_{\text{next}}$: Interval from current beat to next beat (seconds).
   - $RR_{\text{avg10}}$: Running average of the preceding 10 RR intervals (seconds).
   - $RR_{\text{ratio}}$: Ratio of previous RR interval to the running average ($RR_{\text{prev}} / RR_{\text{avg10}}$).
4. **Feature Standardization**: `StandardScaler` was fit strictly on DS1 Train and applied to validation and test sets.

### 2.5 Model Architectures
- **1D CNN with Multi-Modal Fusion**:
  - Waveform Branch: `Conv1D(1->16, k=7)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool(2)` $\to$ `Conv1D(16->32, k=5)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool(2)` $\to$ `Conv1D(32->64, k=3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AdaptiveAvgPool1D(1)` $\to$ 64-dim representation.
  - RR Branch: `Linear(4->16)` $\to$ `ReLU` $\to$ 16-dim representation.
  - Fusion Head: Concatenate (80 dims) $\to$ `Linear(80->32)` $\to$ `ReLU` $\to$ `Dropout(0.2)` $\to$ `Linear(32->5)`.
  - Trainable parameters: **11,989**.
- **Random Forest Baseline**:
  - Input: 32 handcrafted features (4 RR features + 8 statistical moments [mean, std, min, max, peak-to-peak, energy, QRS-segment energy, skewness, kurtosis, zero-crossings] + 20 downsampled waveform points).
  - 100 trees, maximum depth 12, `class_weight='balanced'`.

### 2.6 Class Imbalance Strategy
Training set class distribution is heavily skewed: N represents 89.0%, V represents 8.3%, S represents 1.7%, F represents 0.95%, and Q represents 0.02%.
To avoid majority-class dominance without gradient instability on ultra-rare classes, we used square-root smoothed inverse class frequency weights for cross-entropy loss:
$$w_c = \sqrt{\frac{N_{\text{total}}}{K \cdot N_c}}, \quad \bar{w}_c = \frac{w_c}{\frac{1}{K}\sum_k w_k}$$
Resulting weights: $w_N = 0.056, w_S = 0.403, w_V = 0.183, w_F = 0.539, w_Q = 3.819$.
The test set was kept in its natural clinical distribution with zero alteration.

---

## 3. Results on DS2 Inter-Patient Test Set

Both models were evaluated on the 49,697 beats across the 22 unseen patients of DS2.

### 3.1 Quantitative Performance Summary

| Metric | 1D CNN with RR Fusion | Random Forest Baseline |
| :--- | :---: | :---: |
| **Overall Accuracy** | **92.49%** | 88.59% |
| **Macro Average F1** | **40.96%** | 40.93% |
| **Weighted Average F1** | **92.10%** | 90.00% |
| **Macro Precision** | **41.50%** | 41.06% |
| **Macro Recall** | 40.79% | **42.88%** |

### 3.2 Per-Class Breakdown (Precision, Recall, F1-Score)

| Class | True Support in DS2 | 1D CNN Precision | 1D CNN Recall | 1D CNN F1 | RF Precision | RF Recall | RF F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N** | 44,245 | **95.6%** | **96.5%** | **96.0%** | 95.6% | 91.9% | 93.7% |
| **S** | 1,837 | **21.1%** | **13.2%** | **16.2%** | 18.6% | 12.3% | 14.8% |
| **V** | 3,220 | **90.8%** | 94.3% | **92.5%** | 88.8% | **96.0%** | 92.3% |
| **F** | 388 | 0.0% | 0.0% | 0.0% | **2.2%** | **14.2%** | **3.8%** |
| **Q** | 7 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### 3.3 Confusion Matrix Analysis

#### 1D CNN Confusion Matrix on DS2:
```
               Predicted N   Predicted S   Predicted V   Predicted F   Predicted Q
True N (44245)      42,688           876           166           514             1
True S  (1837)       1,483           242            72            40             0
True V  (3220)         154            28         3,037             1             0
True F   (388)         323             0            65             0             0
True Q     (7)           2             0             5             0             0
```

#### Random Forest Baseline Confusion Matrix on DS2:
```
               Predicted N   Predicted S   Predicted V   Predicted F   Predicted Q
True N (44245)      40,654           966           239         2,380             6
True S  (1837)       1,458           226           108            45             0
True V  (3220)          95            22         3,092            11             0
True F   (388)         294             1            38            55             0
True Q     (7)           3             0             4             0             0
```

---

## 4. In-Depth Failure Mode Analysis

The quantitative evaluation reveals two prominent failure modes that are well documented in clinical cardiology and inter-patient ECG research:

### 4.1 S vs. N Confusion (Supraventricular Ectopic vs. Normal)
- **Empirical Observation**: Out of 1,837 true Supraventricular ectopic (S) beats in DS2, the CNN misclassified **1,483 beats (80.7%)** as Normal (N).
- **Electrophysiological Root Cause**:
  1. *Identical Ventricular Depolarization*: Supraventricular ectopic beats (such as Premature Atrial Contractions [PACs] or Nodal Premature beats) originate in the atria or AV node, *above* the bifurcation of the bundle of His. Once the impulse traverses the AV node, it travels down the normal specialized His-Purkinje conduction system. As a result, ventricular depolarization proceeds normally, producing a QRS complex that is **morphologically identical** to a normal sinus beat.
  2. *Single-Lead MLII Limitation*: The hallmark signature of a PAC is an abnormal, premature, or inverted P wave. However, in lead MLII, the P-wave amplitude is small (typically $< 0.2\text{ mV}$), frequently obscured by baseline wander, or buried in the preceding T-wave of the previous cycle.
  3. *Inter-Patient Heart Rate Diversity*: In an inter-patient paradigm, patients exhibit widely varying baseline resting heart rates (from 55 bpm to 105 bpm). A premature beat in a patient with a slow resting rate (e.g., RR $= 0.75\text{ s}$ when baseline is $1.0\text{ s}$) has an absolute RR interval longer than a normal sinus beat in a patient with sinus tachycardia (e.g., RR $= 0.60\text{ s}$). Without patient-specific baseline calibration, the timing features overlap heavily with normal beats.

### 4.2 F vs. V and F vs. N Confusion (Fusion Beats)
- **Empirical Observation**: Out of 388 true Fusion (F) beats in DS2, the CNN misclassified **323 beats (83.2%) as Normal (N)** and **65 beats (16.8%) as Ventricular (V)**, detecting zero beats as F. The Random Forest classified 55 as F, but at a precision of only 2.2% (with 2,380 false positives from N).
- **Electrophysiological Root Cause**:
  1. *Dual Pacemaker Superposition*: A fusion beat occurs when electrical activation from a normal supraventricular pacemaker and an ectopic ventricular pacemaker collide within the ventricles simultaneously.
  2. *Continuous Morphological Spectrum*: A fusion beat is not a distinct morphological prototype; rather, it is a variable hybrid. If the sinus impulse activates 80% of the ventricular mass before collision, the beat resembles an N beat. If the ectopic ventricular pacemaker activates 80% of the mass, the beat resembles a wide, notched V beat (PVC).
  3. *Training Scarcity and Patient Specificity*: In the DS1 training set, 373 of the 401 total F beats came from a single record (`208`). The model effectively memorized the specific fusion morphology of patient 208, which failed to generalize to the unique fusion morphologies of unseen DS2 patients.

### 4.3 Why Ventricular (V) Classification Succeeds (94.3% Recall, 92.5% F1)
In contrast to S and F, Ventricular ectopic beats (PVCs) achieved excellent recall (94.3%) and precision (90.8%):
1. **Myocardial Conduction Delay**: PVCs originate directly in the ventricular myocardium and conduct through slow muscle-to-muscle pathways rather than the fast Purkinje network, creating an unmistakably widened, high-voltage QRS complex ($\Delta t > 120\text{ ms}$) and discordant T-wave.
2. **Compensatory Pause**: Unlike PACs (which reset the sinus node), PVCs rarely penetrate retrograde into the atria. Consequently, the sinus node fires on schedule, creating a classic full compensatory pause where $RR_{\text{prev}} + RR_{\text{next}} \approx 2 \cdot RR_{\text{normal}}$. This provides an unambiguous signal in the RR feature space ($RR_{\text{ratio}} \ll 1.0$ and $RR_{\text{next}} \gg 1.0$).

---

## 5. Architectural Comparison: 1D CNN vs. Random Forest

| Dimension | 1D CNN with RR Fusion | Random Forest Baseline |
| :--- | :--- | :--- |
| **Model Size** | **165 KB** (.pt file) | 7.0 MB (.joblib file) |
| **Parameters** | **11,989 parameters** | 100 deep decision trees |
| **Feature Input** | Raw normalized 200-sample wave + 4 RR features | 32 handcrafted summary features |
| **Overall Accuracy** | **92.49%** | 88.59% |
| **Weighted F1** | **92.10%** | 90.00% |
| **Normal (N) F1** | **96.0%** | 93.7% |
| **Ventricular (V) F1** | **92.5%** | 92.3% |
| **Supraventricular (S) F1**| **16.2%** | 14.8% |
| **Fusion (F) Precision** | 0.0% (conservative) | 2.2% (2,380 false positives) |

**Conclusion**: The 1D CNN delivers +3.9% higher overall accuracy, +2.1% higher weighted F1, and drastically fewer false alarms on Normal beats (only 514 false F predictions vs. 2,380 for Random Forest), while maintaining a 42x smaller disk footprint and fast CPU inference.

---

## 6. Recommendations for Future Improvements

1. **Patient-Specific Few-Shot Adaptation**: Using the first 5 minutes of an unseen patient's recording to fine-tune the classifier head or compute a patient-specific baseline RR distribution would significantly resolve the S vs. N overlap.
2. **Dual-Lead Fusion (MLII + V1/V5)**: S-beats and P-waves are often dramatically more prominent in lead V1. Incorporating the second channel available in MIT-BIH would improve PAC discrimination.
3. **Data Augmentation for Fusion Beats**: Synthesizing artificial fusion beats by linearly interpolating patient-specific N and V waveforms would provide the model with continuous transitional examples during training.

