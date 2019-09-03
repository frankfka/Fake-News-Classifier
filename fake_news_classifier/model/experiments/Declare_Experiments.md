# Experiment Log (Declare)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Disagree, Part-Agree, Agree) Left-to-Right/Up-to-Down



### Log 2 - Aug 26 (Vary Dense Units)

#### Parameters

**Data:** Unbalanced data (agree, discuss, disagree) with FNC Dataset - Max bias 1.5

```
Total Dataset:
    0    6719 (false)
    1    7570 (partially true)
    2     5047 (true)
```

**Test Data:** Unbalanced data from given dataset

```
 Total Dataset:
    0    1529 (false)
    1    1255 (partially true)
    2    327 (true)
```

Params:

- Tried with the params given in the paper (except SEQ_LEN)

```python
# Model Params
SEQ_LEN = 150
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.5
NUM_LSTM = 64
LEARNING_RATE = 0.002

# Training Params
NUM_EPOCHS = 25
BATCH_SIZE = 128
TRAIN_VAL_SPLIT = 0.2
```



#### Results

##### 256 Unit Dense

- Validation accuracy increased to a maximum of 0.63

- Tested on testing set:

  - Accuracy: 0.5818064930890389
	- F1 Score (Macro): 0.4362452616358024
	- F1 Score (Micro): 0.5818064930890389
	- F1 Score (Weighted): 0.5640065058929761
  - Confusion matrix:

  ```python
  [[0.62393721 0.32897319 0.0470896 ]
   [0.30916335 0.67011952 0.02071713]
   [0.33333333 0.62079511 0.04587156]]
  ```

##### 512 Unit Dense

- Validation accuracy increased to a maximum of 0.6432

- Tested on testing set:

  - Accuracy: 0.563805850208936
	- F1 Score (Macro): 0.43588804222626515
	- F1 Score (Micro): 0.563805850208936
	- F1 Score (Weighted): 0.5553071009631629
  - Confusion matrix:

  ```python
  [[0.6147809  0.3028123  0.0824068 ]
   [0.32031873 0.62868526 0.05099602]
   [0.37614679 0.54740061 0.0764526 ]]
  ```

##### 128 Unit Dense

- Validation accuracy increased to a maximum of 0.6399

- Tested on testing set:

  - Accuracy: Accuracy: 0.5531983285117326
INFO: F1 Score (Macro): 0.4313921692992828
INFO: F1 Score (Micro): 0.5531983285117326
INFO: F1 Score (Weighted): 0.5446229454023307
  - Confusion matrix:
  ```python
  [[0.58927404 0.33551341 0.07521256]
   [0.32031873 0.6310757  0.04860558]
   [0.32721713 0.58715596 0.08562691]]
  ```

##### 64 Unit Dense

- Validation accuracy increased to a maximum of 0.6383

- Tested on testing set:

  - Accuracy: 0.5914496946319512
INFO: F1 Score (Macro): 0.4292108672627555
INFO: F1 Score (Micro): 0.5914496946319512
INFO: F1 Score (Weighted): 0.5657274966884239
  - Confusion matrix:
  ```python
  [[0.65925441 0.32308698 0.0176586 ]
   [0.33545817 0.65737052 0.00717131]
   [0.3853211  0.59327217 0.02140673]]
  ```

##### 128 -> 64 Dense

- Validation accuracy increased to a maximum of 0.6311

- Tested on testing set:

  - Accuracy: Accuracy: 0.5628415300546448
INFO: F1 Score (Macro): 0.4261922853150067
INFO: F1 Score (Micro): 0.5628415300546448
INFO: F1 Score (Weighted): 0.5424960221023222

  - Confusion matrix:
  ```python
  [[0.75016351 0.18966645 0.06017005]
 [0.50039841 0.46055777 0.03904382]
 [0.52905199 0.39143731 0.0795107 ]]
  ```

##### 64 -> 32 Dense

- Validation accuracy increased to a maximum of 0.6298

- Tested on testing set:

  - Accuracy: 0.5576984892317582
INFO: F1 Score (Macro): 0.4387559857337739
INFO: F1 Score (Micro): 0.5576984892317582
INFO: F1 Score (Weighted): 0.550675341082421

  - Confusion matrix:
  ```python
  [[0.64159581 0.28188358 0.0765206 ]
   [0.36095618 0.57370518 0.06533865]
   [0.39143731 0.50458716 0.10397554]]
  ```

##### 64 -> 32 -> 32 Dense

- Validation accuracy increased to a maximum of 0.5967

- Tested on testing set:

- Accuracy: 0.5548055287688846
INFO: F1 Score (Macro): 0.42255733756223873
INFO: F1 Score (Micro): 0.5548055287688846
INFO: F1 Score (Weighted): 0.5400210049347405

- Confusion matrix:
  ```python
  [[0.51079137 0.43100065 0.05820798]
   [0.21752988 0.73784861 0.04462151]
   [0.23547401 0.70642202 0.05810398]]
  ```


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


##### 32 Unit LSTM

* Overtraining apparent, spiked in val_loss in last epoch
* Validation accuracy increased to a maximum of 0.63
* Tested on testing set:
    * Accuracy: 0.5911282545805208
    * F1 Score (Macro): 0.42297520864587534
    * F1 Score (Micro): 0.5911282545805208
    * F1 Score (Weighted): 0.5633673936874397
    * Confusion matrix:
    ```
    [[0.68083715 0.30215827 0.01700458]
     [0.36414343 0.63266932 0.00318725]
     [0.39755352 0.59021407 0.01223242]]
    ```

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
    ```python
    [[0.73054284 0.25506867 0.01438849]
     [0.35697211 0.63027888 0.012749  ]
     [0.3853211  0.59938838 0.01529052]]
    ```