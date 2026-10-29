# MIT-BIH Arrhythmia — Five-Class Heartbeat Classification

Classifies a single heartbeat (187 signal samples) into one of five classes with a recurrent neural network.
The dataset is strongly imbalanced, so the project is about handling that imbalance and judging the result per class.

| Label | Class |
|---|---|
| 0 | Normal |
| 1 | S-type (supraventricular ectopic) |
| 2 | V-type (ventricular ectopic) |
| 3 | F-type (fusion) |
| 4 | Unknown |

## Data

`mitbih_train.csv` (87,554 beats) and `mitbih_test.csv` (21,892 beats), see the
[ECG overview](../README.md) for where to get them. They are **not included**; put both files next to the notebook.
About 83 % of the test beats are normal, and the F-type class has only 162 test beats.

## Approach

1. **Balance the training set.** Every class is brought to 20,000 beats: the normal class is randomly down-sampled, and the four rare classes are over-sampled with replacement. The result is 100,000 training beats of shape `(187, 1)`.
2. **Model** (67,461 parameters): `Masking` → `Bidirectional(GRU(64))` → `GRU(64)` → `Dense(64, ReLU)` → `Dropout(0.2)` → `Dense(5, softmax)`.
3. **Training.** Adam, categorical cross-entropy, batch size 128, 30 epochs, 15 % of the data used for validation, with early stopping and learning-rate reduction on the validation loss. The trained model is saved as `ecg_mitbih_model.keras`.
4. **Evaluation** on the separate official test file with a classification report and a confusion matrix.

## Results on the test set (21,892 beats)

Accuracy **0.97**, macro-average precision 0.80, recall 0.92, F1 **0.85**.

| Class | Precision | Recall | F1 | Beats |
|---|---|---|---|---|
| Normal | 0.99 | 0.97 | 0.98 | 18,118 |
| S-type | 0.63 | 0.84 | 0.72 | 556 |
| V-type | 0.90 | 0.96 | 0.92 | 1,448 |
| F-type | 0.51 | 0.86 | 0.64 | 162 |
| Unknown | 0.98 | 0.97 | 0.97 | 1,608 |

- The model finds most of the rare abnormal beats (recall 0.84–0.96 for S, V and F) even though they are a small share of the data — this is the benefit of balancing.
- The price is precision: only 63 % of the beats called S-type and 51 % of those called F-type really are, because many normal beats are flagged as abnormal. For screening this is usually the better side to err on, but it matters if false alarms are costly.
- Overall accuracy (0.97) is dominated by the normal class; macro-F1 (0.85) and the per-class table give a more honest picture, and the F-type class is the weakest.

## Limitations

- **The validation set is not representative.** `validation_split=0.15` takes the *last* 15 % of the training array, and the balanced array is ordered by class, so the validation data is made only of class 4 ("Unknown") samples — many of them duplicates of samples that are also in the training part. The validation loss and accuracy printed during training (and used by early stopping and the learning-rate schedule) therefore say little about the other classes. Shuffling before the split, or splitting before balancing, would fix this. The **test results above are not affected**: they come from the separate test file, which was not used for any decision.
- The notebook trains a single configuration (no hyper-parameter search) and uses the test file only for the final evaluation.

## Files

| File | Purpose |
|---|---|
| `ecg_mitbih_classification.ipynb` | Balancing, model, training, saving and evaluation (run on Kaggle with a GPU) |
| `ecg_mitbih_model.keras` | The trained model produced by the notebook (0.9 MB) |

## Run

Put `mitbih_train.csv` and `mitbih_test.csv` next to the notebook and run it top to bottom. Training takes a few
minutes per epoch on a Kaggle T4 GPU.
