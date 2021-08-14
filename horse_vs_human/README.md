# Human or Horse classifier

## CNN model is trained on CGI data from scratch

Download data from [here](https://laurencemoroney.com/datasets.html) or use it directly from [Tensorflow](https://www.tensorflow.org/datasets/catalog/horses_or_humans) and keep in the below format if not using it from tensorflow dataset


```
data
    ├── horse_or_human
    │   ├── horses
    │   └── humans
    ├── test
    └── validation_horse_or_human
        ├── horses
        └── humans
```

- visualize the intermediate features
- use ImageDataGenerator to read data from folders
- Try different conv layers & dense layer for better accuracy
- use different optimizers