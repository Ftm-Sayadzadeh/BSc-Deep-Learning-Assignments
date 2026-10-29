# PTB-DB — Normal vs. Abnormal Heartbeat

A binary classifier that decides whether a single ECG heartbeat is normal or abnormal, built as a small class
(`ECGClassifier`) that loads the data, trains a Conv1D + Bi-GRU network, evaluates it and can save and reload the
trained model.

## Data

`ptbdb_normal.csv` (4,046 beats) and `ptbdb_abnormal.csv` (10,506 beats), each beat being 187 signal values plus a
label; see the [ECG overview](../README.md) for where to get them. They are **not included**; put them in a
`ptbdb-dataset/` folder next to the notebook. About 72 % of the beats are abnormal.

## Approach

1. **Stratified split** into 70 % train / 15 % validation / 15 % test (seed 42), keeping the class ratio in each part. The test set has 2,183 beats (607 normal, 1,576 abnormal).
2. **Scaling.** `MinMaxScaler` fitted on the training beats only, then applied to validation and test.
3. **Model:** `Masking` → `Conv1D(32, 3)` + `BatchNormalization` → `Bidirectional(GRU(128))` → `Bidirectional(GRU(64))` → `Flatten` → `Dense(128)` + `Dropout(0.3)` → `Dense(64)` → `Dense(1, sigmoid)`.
4. **Training.** Adam (learning rate 1e-3), binary cross-entropy, batch size 128, 25 epochs, validated on the validation split after every epoch.
5. **Save / reload.** The notebook saves the trained model and the fitted scaler, reloads them and classifies single beats (`predict_new_sample`).

## Results on the test set (2,183 beats)

Accuracy **99.63 %**, macro-F1 0.995.

| Class | Precision | Recall | F1 | Beats |
|---|---|---|---|---|
| Normal (0) | 0.9983 | 0.9885 | 0.9934 | 607 |
| Abnormal (1) | 0.9956 | 0.9994 | 0.9975 | 1,576 |

- Almost every abnormal beat is found (recall 0.999); the few errors are mostly normal beats flagged as abnormal.
- Another run of the same network with early stopping and learning-rate reduction reached 99.45 %. A difference of 0.2 points is only about 4 beats out of 2,183, so the two runs should be treated as equivalent.

## Limitations

- **The split is by heartbeat, not by patient.** The CSV files carry no patient identifier, so beats of the same patient can end up in both the training and the test set. This tends to make accuracy on this dataset look higher than it would be on new patients, so the 99.6 % should be read as an upper bound for real-world use.
- The two classes are imbalanced (about 28 % normal); the per-class table above is therefore more informative than accuracy alone.

## Files

| File | Purpose |
|---|---|
| `ecg_ptbdb_classification.ipynb` | Data preparation, model, training, evaluation, saving and reloading |

The trained model (`ecg_model_final.keras`, about 38 MB) and the scaler are not included; the notebook produces them.

## Run

Put the CSV files in `ptbdb-dataset/` next to the notebook and run it top to bottom.
