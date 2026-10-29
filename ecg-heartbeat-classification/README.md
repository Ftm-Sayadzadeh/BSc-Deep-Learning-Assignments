# ECG Heartbeat Classification

Two projects that classify single heartbeats from ECG recordings with recurrent neural networks:

| Folder | Task | Test accuracy |
|---|---|---|
| [`mitbih-arrhythmia`](mitbih-arrhythmia) | Five arrhythmia classes (normal, S, V, F, unknown) | 97 % (macro-F1 0.85) |
| [`ptbdb-normal-vs-abnormal`](ptbdb-normal-vs-abnormal) | Normal vs. abnormal heartbeat | 99.6 % |

## Data

Both projects use the **ECG Heartbeat Categorization Dataset** from Kaggle, which is built from two PhysioNet
databases (the MIT-BIH Arrhythmia Database and the PTB Diagnostic ECG Database). Each row of a CSV file is one
heartbeat: 187 signal values followed by the class label (188 columns, no header).

| File | Rows | Used by |
|---|---|---|
| `mitbih_train.csv` | 87,554 | `mitbih-arrhythmia` |
| `mitbih_test.csv` | 21,892 | `mitbih-arrhythmia` |
| `ptbdb_normal.csv` | 4,046 | `ptbdb-normal-vs-abnormal` |
| `ptbdb_abnormal.csv` | 10,506 | `ptbdb-normal-vs-abnormal` |

The files are **not included** here (about 550 MB in total; `mitbih_train.csv` alone is 392 MB). Download them
from Kaggle and put them next to the notebooks:

- `mitbih-arrhythmia/`: `mitbih_train.csv` and `mitbih_test.csv`
- `ptbdb-normal-vs-abnormal/`: `ptbdb-dataset/ptbdb_normal.csv` and `ptbdb-dataset/ptbdb_abnormal.csv`

Please check the original sources for the license and citation of the data.

## Run

```bash
pip install -r ../requirements.txt
jupyter notebook
```

Then open the notebook of the project you want. A GPU makes training much faster.
