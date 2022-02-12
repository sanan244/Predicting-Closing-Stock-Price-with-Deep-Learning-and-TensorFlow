Predicting-Closing Stock Price-with-Deep-Learning-and-TensorFlow

This project aims to predict the closing price of a stock for companies based on a continous daily stock history of many S&P 500 companies. Some attributes are excluded from training as they do not have significant correlation to the closing price target. The dataset was found on kaggle(https://www.kaggle.com/andrewmvd/sp-500-stocks/version/48). The specific file in the archive is 'sp500_stocks.csv'

![Screen Shot 2022-02-12 at 2 42 07 PM](https://user-images.githubusercontent.com/76133001/153731054-f242462a-6c27-45a0-a24c-d4716466243e.png)


Results of the model may vary depending on the randomness of the split training data.
Example results:
Pre-processing file...
...Dataset:
                Date Symbol   Adj Close       Close        High         Low        Open     Volume  Date_code
1        2010-01-04    MMM   59.892292   83.019997   83.449997   82.669998   83.089996  3043700.0          0
2        2010-01-05    MMM   59.517159   82.500000   83.230003   81.699997   82.800003  2847000.0          1
3        2010-01-06    MMM   60.361195   83.669998   84.599998   83.510002   83.879997  5268500.0          2
4        2010-01-07    MMM   60.404488   83.730003   83.760002   82.120003   83.320000  4470100.0          3
5        2010-01-08    MMM   60.830139   84.320000   84.320000   83.300003   83.690002  3405800.0          4
...             ...    ...         ...         ...         ...         ...         ...        ...        ...
1540245  2022-01-27    ZTS  187.660004  187.660004  192.779999  187.570007  190.339996  4606200.0       3038
1540246  2022-01-28    ZTS  195.300003  195.300003  195.580002  185.720001  188.850006  2963800.0       3039
1540247  2022-01-31    ZTS  199.789993  199.789993  200.460007  196.000000  196.000000  2591500.0       3040
1540248  2022-02-01    ZTS  198.869995  198.869995  202.460007  196.100006  200.990005  2143200.0       3041
1540249  2022-02-02    ZTS  202.169998  202.169998  203.410004  198.490005  198.490005  2494300.0       3042

[1470924 rows x 9 columns]
...Original X train data:
           Adj Close        High         Low        Open      Volume  Date_code
553039   127.464348  155.100006  151.669998  153.710007    315300.0        987
872708    26.619455   35.849998   35.110001   35.270000  11231800.0        407
514373    34.456669   35.096668   34.123333   34.500000   9915900.0       1968
326099   159.679855  165.139999  162.380005  163.529999   2233800.0       2792
171040    77.282646  112.820000  109.980003  111.959999    486600.0        239
...             ...         ...         ...         ...         ...        ...
57957     44.328747   63.580002   62.599998   62.700001   1281900.0          6
1011491   35.694614   40.240002   39.160000   39.619999   4108800.0       1936
659318    48.048325   65.150002   63.950001   64.269997    586300.0        517
414728   239.204865  242.130005  238.979996  242.000000    522200.0       2971
840703    15.077754   19.309999   19.160000   19.290001   8349000.0       1948

[735462 rows x 6 columns]
...Normalized train data
 [[0.0007239  0.00086065 0.00086136 0.0008627  0.0000188  0.00064525]
 [0.00015118 0.00019893 0.0001994  0.00019795 0.00066955 0.00026608]
 [0.00019569 0.00019475 0.00019379 0.00019363 0.00059111 0.00128658]
 ...
 [0.00027288 0.00036152 0.00036318 0.00036072 0.00003495 0.00033799]
 [0.0013585  0.00134357 0.00135721 0.00135822 0.00003113 0.00194229]
 [0.00008563 0.00010715 0.00010881 0.00010827 0.0004977  0.0012735 ]]
2022-02-12 13:38:46.405108: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
fitting model...
Epoch 1/3
22984/22984 [==============================] - 56s 2ms/step - loss: 0.0014 - val_loss: 1.0591e-06
Epoch 2/3
22984/22984 [==============================] - 52s 2ms/step - loss: 7.4869e-07 - val_loss: 3.4348e-07
Epoch 3/3
22984/22984 [==============================] - 52s 2ms/step - loss: 1.8861e-07 - val_loss: 7.4651e-08
predicting...
WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor. Received: inputs=(<tf.Tensor 'IteratorGetNext:0' shape=(None, 6) dtype=float64>, <tf.Tensor 'IteratorGetNext:1' shape=(None, 1) dtype=float64>). Consider rewriting this model with the Functional API.
...Rescale factor for outputs: [176971.74453236]
        Correct values  Predictions
0            83.019997    90.003694
1            83.730003    89.581760
2            84.320000    90.140822
3            83.500000    90.325418
4            83.370003    89.834920
...                ...          ...
735457      195.250000   143.905701
735458      189.839996   140.804489
735459      187.660004   139.454302
735460      195.300003   141.268616
735461      198.869995   145.720015

[735462 rows x 2 columns]

Process finished with exit code 0
