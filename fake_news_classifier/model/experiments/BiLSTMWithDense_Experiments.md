# Experiment Log (BiLSTMWithDense)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Agree, Discuss, Disagree) Left-to-Right/Up-to-Down

### Log 1 - Aug 13
#### Parameters

**Data:** Unbalanced PARTIAL data (agree, discuss, disagree): On 4/5 Folds

    Total Dataset:
        0    2365 (agree)
        1    2078 (discuss)
        2     558 (disagree)

**Model:**
* Individual input Bi-Directional LSTM
* Concatenate
* Batch Normalization
* Dropout
* Dense
* Dropout
* Dense 
* Dense (Classification Output)

**Params:** 
```python
# Model Params
SEQ_LEN = 500
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.2
NUM_LSTM_UNITS = 128
NUM_DENSE_HIDDEN = 128
LEARNING_RATE = 0.005

# Training Params
NUM_EPOCHS = 25
BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.2
```

#### Results

* Validation accuracy stayed relatively constant the entire time
* Validation loss bottomed out by epoch 3 - VERY overtrained
* Tested on 1 fold:
    * Accuracy: 0.5244755244755245
    * F1 Score (Macro): 0.3977639407190658
    * F1 Score (Micro): 0.5244755244755245
    * F1 Score (Weighted): 0.5097193342049723
    * Confusion matrix:
    ```
    [[0.52854123 0.41014799 0.06131078]
     [0.29807692 0.64903846 0.05288462]
     [0.41071429 0.54464286 0.04464286]]
    ```
  
    * Decent at predicting "discuss"
    * OK at predicting "Agree"
    * Bad at "disagrees"