# Forest Cover Type Classification

A deep learning multi-class classifier that predicts forest cover type for
30x30m cells in Roosevelt National Forest, northern Colorado, using 54
cartographic variables. Built with TensorFlow and Keras.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 0.9406 |
| Test loss | 0.1560 |
| Classes | 7 |
| Training rows | 581,012 |

Per-class recall ranges from 0.83 (Aspen) to 0.95 (Lodgepole Pine), with the
rare classes benefiting most from an extended training budget.

## Dataset

US Forest Service Region 2 Resource Information System data, with independent
variables derived from the US Geological Survey and USFS. 581,012 rows, 54
features (10 continuous cartographic measures, 4 binary wilderness-area
indicators, 40 binary soil-type indicators), 7 cover-type classes.

The data is heavily imbalanced — the two largest classes account for 85.2% of
all rows.

## Approach

1. **Preprocessing** — stratified 80/20 train/test split with a further 15%
   validation split; StandardScaler fitted on training data only and applied to
   the 10 continuous columns; the 44 binary columns left untouched.
2. **Baseline** — a fully connected network with softmax output, trained with
   sparse categorical crossentropy.
3. **Hyperparameter search** — five configurations varying depth, width,
   dropout and learning rate.
4. **Diagnosis** — all five configurations hit the epoch ceiling without
   EarlyStopping firing, indicating undertraining rather than overfitting.
   Retraining the winning architecture with a larger budget and
   ReduceLROnPlateau lifted validation accuracy from 0.9155 to 0.9404.
5. **Evaluation** — classification report, row-normalised confusion matrix, and
   per-class recall against class size.

## Key finding

The extended training budget helped the minority classes disproportionately:
Aspen recall rose 11 points and Cottonwood/Willow 9 points, while the majority
classes gained 0–5 points. Majority classes are seen tens of thousands of times
per epoch and converge early; rare classes need far more passes. Stopping early
penalises them specifically.

Errors are structured rather than random — the confusion matrix is near-diagonal
with off-diagonal mass concentrated in pairs that share overlapping elevation
bands (Aspen/Lodgepole Pine, Cottonwood/Willow/Ponderosa Pine).

## Files

| File | Description |
|---|---|
| `cover_data.ipynb` | Full notebook — preprocessing, training, tuning, evaluation |
| `report.md` | Written project report with analysis |
| `training_curves.png` | Accuracy and loss curves for the final model |
| `confusion_matrix.png` | Row-normalised confusion matrix |

## Stack

Python, TensorFlow, Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn.

## Running it

The notebook was developed in Google Colab. Upload `cover_data.csv` (available
from the Codecademy project files) to the session and run the cells in order.

---

Built as the final project for Codecademy's *Build Deep Learning Models with
TensorFlow*.
