# Experiment Log (Declare)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Disagree, Part-Agree, Agree) Left-to-Right/Up-to-Down


### Log 1 - Aug 25 (Vary LSTM Units)
#### Parameters

**Data:** Unbalanced data (agree, discuss, disagree) with FNC Dataset - Max bias 1.5

    Total Dataset:
        0    6719 (false)
        1    7570 (partially true)
        2     5047 (true)

**Test Data:** Unbalanced data from given dataset

     Total Dataset:
        0    1529 (false)
        1    1255 (partially true)
        2    327 (true)


**Model:**
* See [paper](https://people.mpi-inf.mpg.de/~kpopat/publications/emnlp2018_Popat.pdf)

**Params:** 
* Tried with the params given in the paper (except SEQ_LEN)
```python
# Model Params
SEQ_LEN = 150
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.5
NUM_DENSE_HIDDEN = 32
LEARNING_RATE = 0.002

# Training Params
NUM_EPOCHS = 25
BATCH_SIZE = 128
TRAIN_VAL_SPLIT = 0.2
```

#### Results

##### 128 Unit LSTM

* Overtraining apparent, spiked in val_loss in last epoch
* Validation accuracy increased to a maximum of 0.63
* Tested on testing set:
    * Accuracy: 0.6033429765348762
    * F1 Score (Macro): 0.4446613524701119
    * F1 Score (Micro): 0.6033429765348762
    * F1 Score (Weighted): 0.5809365202444198
    * Confusion matrix:
    ```
    [[0.67298888 0.29758012 0.029431  ]
     [0.31633466 0.66693227 0.01673307]
     [0.35168196 0.6146789  0.03363914]]
    ```

##### 256 Unit LSTM

* Overtraining apparent, spiked in val_loss in last epoch
* Tested on testing set:
    * Accuracy: 0.555126968820315
    * F1 Score (Macro): 0.44560123252961853
    * F1 Score (Micro): 0.555126968820315
    * F1 Score (Weighted): 0.5585147533322925
    * Confusion matrix:
    ```
    [[0.58338784 0.27861347 0.13799869]
     [0.2812749  0.63266932 0.08605578]
     [0.27828746 0.59633028 0.12538226]]
    ```

##### 64 Unit LSTM

* Validation accuracy increased to a maximum of 0.63
* Tested on testing set:
    * Accuracy: 0.614914818386371
    * F1 Score (Macro): 0.44062655889667907
    * F1 Score (Micro): 0.614914818386371
    * F1 Score (Weighted): 0.5862989552590416
    * Confusion matrix:
    ```
    [[0.73054284 0.25506867 0.01438849]
     [0.35697211 0.63027888 0.012749  ]
     [0.3853211  0.59938838 0.01529052]]
    ```