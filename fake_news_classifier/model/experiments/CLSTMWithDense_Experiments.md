# Experiment Log (BiLSTMWithDense)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Agree, Discuss, Disagree) Left-to-Right/Up-to-Down

### Log 3 - Aug 13 (Dense Width Experiments)
#### Parameters

**Changes:**

* Batch size to 128
* CNN Kernel to 5 as established in log 2
* Dense Width to 256
* Training on ALL Data: 

### Log 2 - Aug 13 (Kernel Size Experiments)
#### Parameters

**Changes:**

* Batch size to 128
* CNN Kernel to 5 (Tried 10 - did not learn at all)

#### Results
* Trained on 1 fold
* Validation accuracy increased but slope very flat compared to training accuracy. Maxed out at epoch 10
* Validation loss relatively flat, but somewhat downwards. Spiked back up at epoch 17
* Tested on 1 fold:
    * Accuracy: 0.5814185814185814
    * F1 Score (Macro): 0.44438531161527317
    * F1 Score (Micro): 0.5814185814185814
    * F1 Score (Weighted): 0.5624413751568982
    * Confusion matrix:
    ```
    [[0.66454352 0.31847134 0.01698514]
     [0.34588235 0.61647059 0.03764706]
     [0.47619048 0.45714286 0.06666667]]
    ```

### Log 1 - Aug 13
#### Parameters

**Data:** Unbalanced PARTIAL data (agree, discuss, disagree): On 4/5 Folds

    Total Dataset:
        0    2365 (agree)
        1    2078 (discuss)
        2     558 (disagree)

**Model:**
* Individual inputs:
    * CNN (1D)
    * Dropout
    * Max-Pooling
    * Bi-Directional LSTM
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
DROPOUT = 0.5
NUM_CNN_UNITS = 256
CNN_KERNEL_SIZE = 3
NUM_LSTM_UNITS = 128
NUM_DENSE_HIDDEN = 64

# Training Params
NUM_EPOCHS = 25
BATCH_SIZE = 64
TRAIN_VAL_SPLIT = 0.2
```

#### Results
* Trained on 2 folds - results are from first fold
* Validation accuracy increased but slope very flat compared to training accuracy
* Validation loss relatively flat, but somewhat downwards. Spiked back up at epoch 17
* Tested on 1 fold:
    * Accuracy: 0.553
    * F1 Score (Macro): 0.4478193947613551
    * F1 Score (Micro): 0.553
    * F1 Score (Weighted): 0.5434834136580228
    * Confusion matrix:
    ```
    [[0.64718163 0.30688935 0.04592902]
     [0.37469586 0.55474453 0.07055961]
     [0.41818182 0.44545455 0.13636364]]
    ```
  
    * Decent at predicting "discuss"
    * OK at predicting "Agree"
    * Bad at "disagrees"