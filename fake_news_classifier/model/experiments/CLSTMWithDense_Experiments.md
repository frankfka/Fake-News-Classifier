# Experiment Log (BiLSTMWithDense)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Disagree, Part-Agree, Agree) Left-to-Right/Up-to-Down

## To Try
* Training with max label bias = 1
* Training with no max label bias

## Update
This configuration seems to give pretty decent results (Default learning rate):

```
model.SEQ_LEN: 500,
model.EMB_DIM: 300,
model.CONV_KERNEL_SIZE: 5,
model.DENSE_UNITS: 1024,
model.CONV_UNITS: 256,
model.LSTM_UNITS: 128
```


The best results were found with the *FastText* tokenizer (max label bias = 1.5):

```
* Accuracy: 0.6162005785920925
* F1 Score (Macro): 0.47555855029405
* F1 Score (Micro): 0.6162005785920925
* F1 Score (Weighted): 0.6017482973478836

[[0.70111184 0.25376063 0.04512753]
 [0.30517928 0.65099602 0.0438247 ]
 [0.32721713 0.58715596 0.08562691]]
```

This was BEFORE num2word

### Log 6 - Aug 15 (LSTM Experiments)
#### Parameters

**Changes:**

* CNN Kernel to 5
* Dense Width to 1024 -> 1024
* CNN Size to 1024
* Training on data with max label bias of 1.5:
    ```
    0    2544
    1    2544
    2    1696
    ```

#### Results
* Trained on one fold

    ##### LSTM 1024 Units
    
    * Training time takes REALLY long - skipping for now
    
    ##### LSTM 32 Units
    
    ```python
    [[0.33005894 0.31434185 0.35559921]
     [0.1237721  0.46954813 0.40667976]
     [0.16764706 0.31764706 0.51470588]]
    ```
    * Accuracy: 0.42857142857142855
    * F1 Score (Macro): 0.4265500895504983
    * F1 Score (Micro): 0.42857142857142855
    * F1 Score (Weighted): 0.431397725165404
    * Fold 0 completed in 4157.172898054123 seconds

    ##### LSTM 64 Units
    ```python
    [[0.64636542 0.35363458 0.        ]
    [0.3280943  0.66994106 0.00196464]
    [0.41764706 0.57941176 0.00294118]]
    ```
    * Accuracy: 0.49410898379970547
    * F1 Score (Macro): 0.3784485395728188
    * F1 Score (Micro): 0.49410898379970547
    * F1 Score (Weighted): 0.4248178319838934

    ##### LSTM 256 Units 
    
    ```
    [[0.72102161 0.2043222  0.07465619]
     [0.47740668 0.44007859 0.08251473]
     [0.53235294 0.32941176 0.13823529]]
    ```

    * Accuracy: 0.4698085419734904
    * F1 Score (Macro): 0.41265868350846796
    * F1 Score (Micro): 0.4698085419734904
    * F1 Score (Weighted): 0.4389636812197684
    
    ##### LSTM 128 Units + 1028 CNN
    
    ```
    [[0.67779961 0.32023576 0.00196464]
     [0.33595285 0.66208251 0.00196464]
     [0.44705882 0.53529412 0.01764706]]
    ```
 
    * Accuracy: 0.5066273932253313
    * F1 Score (Macro): 0.3955433171646294
    * F1 Score (Micro): 0.5066273932253313
    * F1 Score (Weighted): 0.4404764794576529

### Log 5 - Aug 15 (CNN Experiments)
#### Parameters

**Changes:**

* CNN Kernel to 5
* Dense Width to 1024 -> 1024
* Training on data with max label bias of 1.5:
    ```
    0    2544
    1    2544
    2    1696
    ```

#### Results
* Trained on one fold

    ##### CNN 1024 Units
    * Accuracy: 0.4683357879234168
    * F1 Score (Macro): 0.41809996006686934
    * F1 Score (Micro): 0.4683357879234168
    * F1 Score (Weighted): 0.44462996427181584
    * Confusion Matrix:
    ```
    [[0.6345776  0.28290766 0.08251473]
     [0.37328094 0.51669941 0.11001965]
     [0.46764706 0.38529412 0.14705882]]
    ```
    
    #### CNN 1024 -> 1024 Units
    * Did not work whatsoever


## Experiment Cut-Off: Findings
    * Kernel size doesn't have too big of a difference (between 3 and 5) - 5 might be slightly better
    * Dense width is good at 1024 - started predicting agree
    * Dense depth is good at 2 layers - 3 layers would over-fit (but haven't tried 1)
    * Dropout just before prediction layer may have a small beneficial effect

### Log 4 - Aug 14 (Dense Depth Experiments)
#### Parameters

**Changes:**

* CNN Kernel to 5
* Dense Width to (1024->512->256) / (1024->512->512) 
* Training on data with max label bias of 1.5:
    ```
    0    2544
    1    2544
    2    1696
    ```

#### Results
* Trained on one fold

    ##### (1024->512->256) Dense Layers
    * Accuracy: 0.4801178203240059
    * F1 Score (Macro): 0.3629701649032504
    * F1 Score (Micro): 0.4801178203240059
    * F1 Score (Weighted): 0.40814097334850025
    * Confusion matrix:
    ```
    [[0.75442043 0.24557957 0.        ]
     [0.47347741 0.52652259 0.        ]
     [0.48823529 0.51176471 0.        ]]
    ```
  
  ##### (1024->512->512) Dense Layers
    * Accuracy: 0.4955817378497791
    * F1 Score (Macro): 0.3782053022683091
    * F1 Score (Micro): 0.4955817378497791
    * F1 Score (Weighted): 0.4245443243520919
    * Confusion matrix:
    ```
    [[0.55992141 0.44007859 0.        ]
     [0.23772102 0.76031434 0.00196464]
     [0.35294118 0.64411765 0.00294118]]
    ```
  
  ##### (1024->512->512) Dense Layers
    * Accuracy: 0.48232695139911635
    * F1 Score (Macro): 0.3790401537752219
    * F1 Score (Micro): 0.4823269513991163
    * F1 Score (Weighted): 0.42014021529877704
    * Confusion matrix:
    ```
    [[0.46168959 0.50884086 0.02946955]
     [0.1827112  0.80746562 0.00982318]
     [0.21764706 0.75588235 0.02647059]]
    ```
  
  ##### (1024->1024->1024) Dense Layers
    * Accuracy: 0.47128129602356406
    * F1 Score (Macro): 0.3455957855606859
    * F1 Score (Micro): 0.471281296023564
    * F1 Score (Weighted): 0.3886043921584442
    * Confusion matrix:
    ```
    [[0.37131631 0.62868369 0.        ]
     [0.11394892 0.88605108 0.        ]
     [0.25588235 0.74411765 0.        ]]
    ```

### Log 3 - Aug 13/14 (Dense Width Experiments)
#### Parameters

**Changes:**

* Batch size to 128
* CNN Kernel to 5 as established in log 2
* Dense Width to 256, 1024, 512, 128, 32, 256 -> 128?
* Training on data with max label bias of 1.5:
    ```
    0    2544
    1    2544
    2    1696
    ```

#### Results
* Trained on one fold

    ##### 256 Dense Layer
    * Accuracy: 0.49337260677466865
    * F1 Score (Macro): 0.3812399215190465
    * F1 Score (Micro): 0.49337260677466865
    * F1 Score (Weighted): 0.42515892838646474
    * Confusion matrix:
    ```
    [[0.49508841 0.49312377 0.01178782]
     [0.18467583 0.81139489 0.00392927]
     [0.27352941 0.71176471 0.01470588]]
    ```
  
    ##### 32 Dense Layer
    * Accuracy: 0.4657332350773766
    * F1 Score (Macro): 0.36016956895102026
    * F1 Score (Micro): 0.4657332350773766
    * F1 Score (Weighted): 0.4031965410180302
    * Confusion matrix:
    ```
    [[0.69941061 0.28683694 0.01375246]
     [0.44400786 0.53634578 0.01964637]
     [0.51917404 0.4719764  0.00884956]]
    ```
  
   ##### 1024 Dense Layer
    * *NOTE:* This was able to predict AGREE!!
    * Accuracy: 0.44624447717231225
    * F1 Score (Macro): 0.4435914059449373
    * F1 Score (Micro): 0.44624447717231225
    * F1 Score (Weighted): 0.4519195329334358
    * Confusion matrix:
    ```
    [[0.42043222 0.29862475 0.28094303]
     [0.17092338 0.46561886 0.36345776]
     [0.21176471 0.33235294 0.45588235]]
    ```
    
    * After adding Dropout before classification layer:
    * Accuracy: 0.46612665684830634
    * F1 Score (Macro): 0.4218670251943633
    * F1 Score (Micro): 0.46612665684830634
    * F1 Score (Weighted): 0.44470108894324306
    * Confusion matrix:
    ```
    [[0.46561886 0.41453831 0.11984283]
     [0.26915521 0.66208251 0.06876228]
     [0.26470588 0.56176471 0.17352941]]
    ```
  
  ##### 512 Dense Layer (TODO)
    * Accuracy: 0.4657332350773766
* F1 Score (Macro): 0.40030695080906825
* F1 Score (Micro): 0.4657332350773766
* F1 Score (Weighted): 0.4287160127389453
    * Confusion matrix:
    ```
    [[0.39489194 0.49115914 0.11394892]
     [0.18467583 0.76817289 0.04715128]
     [0.19764012 0.68436578 0.1179941 ]]
    ```


### Log 2 - Aug 13 (Kernel Size Experiments)
*RESULTS: Kernel Size 3 vs. 5 - not much difference. Model worse with 10*

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