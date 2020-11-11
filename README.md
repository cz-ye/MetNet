## MetNet: A convolutional neural networks for predicting N6-methyadenosine sites based on RNA sequence information

An archived project in 2016. Written in Python 2.7, with Theano 0.8 and Keras 1.0.


### Pre-processing
```
./site2samp.py --mode transcript_train --path ./train_data/ --window 700
./site2samp.py --mode transcript_test --path ./test_data/ --window 700
./samp2array.py --path ./train_data/ --winodw 701
./samp2array.py --path ./train_data/ --winodw 701
```
The scripts above take genome sequence data and coordinates of candidate m6A sites as inputs and output one-hot encoded RNA sequences.
For training and validation: a 4 (one-hot encoding of the 4 nucleotides) * m (# samples) * k (length of the input sequence)  Numpy array `pos_data.npy` and a 4 * n * k Numpy array `neg_data.npy` under the directory fed to argument `--train` or `--test` of the main function.
For prediction: a 4 * m * k Numpy array `data.npy` under the directory fed to argument `--test`.

### Train and validate model
```
./CNN_main.py --train ./train_data/ --test ./test_data/ -v 1
```

### Predict new sites using the model
```
./CNN_main.py --weights ./weights/param0 --test ./test_data/
```

