# Forest Cover Type Classification — Project Report

## 1. Problem
Predict forest cover type (7 classes) for 30x30m cells in Roosevelt National Forest,
northern Colorado, using 54 cartographic variables only. Data: US Forest Service
Region 2 RIS, US Geological Survey and USFS. Multi-class classification.

## 2. Data
- Rows: 581,012  |  Features: 54  |  Classes: 7
- Missing values: 0
- 10 continuous features + 4 binary wilderness-area columns + 40 binary soil-type columns
- Heavily imbalanced: the two largest classes account for
  85.2% of all rows

## 3. Preprocessing
- Labels shifted from 1-7 to 0-6 for sparse_categorical_crossentropy
- Stratified 80/20 train/test split, then a stratified 15% validation split off train
- StandardScaler fitted on train only, applied to the 10 continuous columns
- Binary columns left untouched (scaling them would destroy their meaning)

## 4. Models tried
| config        | layers          |   dropout |     lr |   epochs_run |   val_accuracy |   val_loss |   seconds |
|:--------------|:----------------|----------:|-------:|-------------:|---------------:|-----------:|----------:|
| deep_extended | (256, 128, 64)  |       0   | 0.001  |          150 |         0.9404 |     0.156  |    1012   |
| deep          | (256, 128, 64)  |       0   | 0.001  |           40 |         0.9155 |     0.2128 |     252.8 |
| wide_lowlr    | (512, 256, 128) |       0.2 | 0.0005 |           40 |         0.9145 |     0.2133 |     758   |
| deep_drop     | (256, 128, 64)  |       0.2 | 0.001  |           40 |         0.9067 |     0.2285 |     354.3 |
| baseline      | (128, 64)       |       0   | 0.001  |           40 |         0.8825 |     0.2903 |     120.3 |
| small         | (64, 32)        |       0   | 0.001  |           40 |         0.8555 |     0.3516 |      94.5 |

Best configuration: **deep_extended**

### 4.1 How the search was steered
The first five configurations all ran to the full 40-epoch ceiling without
EarlyStopping ever firing, meaning val_loss was still falling when training
stopped. The models were undertrained, not overfitted. Two observations follow
from this:

- Dropout hurt rather than helped. The same (256, 128, 64) architecture scored
  0.9155 without dropout and 0.9067 with it. Dropout is a regularisation tool,
  and there was no overfitting to regularise.
- Width and a lower learning rate were not worth their cost. `wide_lowlr` took
  3x the training time to land 0.001 below the plain `deep` model.

The diagnosis was therefore a training-budget problem, not an architecture
problem. `deep_extended` re-runs the winning architecture with a 150-epoch
ceiling, patience raised to 10, and ReduceLROnPlateau to refine the final
descent. This lifted validation accuracy from 0.9155 to 0.9404
and cut validation loss from 0.2128 to 0.1560.

## 5. Results
- Test accuracy: 0.9408
- Test loss: 0.1604

Per-class recall:
| class             |   test_samples |   recall |
|:------------------|---------------:|---------:|
| Spruce/Fir        |          42368 |    0.939 |
| Lodgepole Pine    |          56661 |    0.952 |
| Ponderosa Pine    |           7151 |    0.94  |
| Cottonwood/Willow |            549 |    0.863 |
| Aspen             |           1899 |    0.831 |
| Douglas-fir       |           3473 |    0.874 |
| Krummholz         |           4102 |    0.93  |

See `training_curves.png` and `confusion_matrix.png`.

## 6. Analysis

**Accuracy alone is misleading.** Given the imbalance, a model that predicted
only the two majority classes would already score around
85%. Per-class recall is the metric
that matters here, and the confusion matrix is what makes the failure modes
legible.

**The extended training budget helped the rare classes most.** Comparing the
40-epoch model against the 150-epoch model:

| Class | 40 epochs | 150 epochs | Gain |
|---|---|---|---|
| Cottonwood/Willow | 0.774 | 0.86 | +0.09 |
| Aspen | 0.716 | 0.83 | +0.11 |
| Spruce/Fir | 0.892 | 0.94 | +0.05 |
| Ponderosa Pine | 0.910 | 0.94 | +0.03 |
| Lodgepole Pine | 0.951 | 0.95 | 0.00 |

The majority classes are seen tens of thousands of times per epoch and converge
early; the minority classes need far more passes before the network learns their
pattern. Stopping at 40 epochs was effectively penalising the rare classes
specifically.

**Errors are structured, not random.** The confusion matrix is near-diagonal,
with the off-diagonal mass concentrated in a few specific pairs:

- Aspen misclassified as Lodgepole Pine (0.13) — the single largest error
- Cottonwood/Willow misclassified as Ponderosa Pine (0.10)
- Spruce/Fir and Lodgepole Pine confused with each other (0.06 / 0.04)

Every one of these pairs shares an overlapping elevation band and similar
cartographic signature, so the model is failing where the input features are
genuinely ambiguous rather than failing arbitrarily. Most other cells are zero.

**Remaining headroom.** EarlyStopping still did not fire at 150 epochs — val_loss
was improving by roughly 0.0001 per epoch at the end. Further training would
keep gaining, but at a rate that does not justify the compute. This is a
deliberate stopping point, not a converged one.

**Overfitting check.** Final training accuracy was 0.958 against validation
0.940, a gap of under two points, and test accuracy (0.9408) tracks
validation closely. The model generalises.

## 7. Next steps
- Class weights or oversampling to push the Aspen and Cottonwood/Willow recall higher
- Batch normalisation to allow a higher learning rate and faster convergence
- Feature engineering: Euclidean distance from the horizontal/vertical
  distance-to-hydrology pair, which are currently two separate features
- Compare against a gradient-boosted tree baseline, which typically performs
  strongly on tabular data of this kind
