# MNIST_Pytorch_Lightning
A PyTorch Lightning example on MNIST Dataset

# Requirements

These libraries are required -
```
PyTorch
scikit-learn
PyTorch Lightning
```

# Training
Change the hyperparameters in run.sh and execute it using the following command:
```
bash run.sh
```
The confusion matrix will be stored in the folder `visualizations` and the logs will be stored in `tensorboard_logs`.  

# Tensorboard
To start tensorboard run the following command:
```
tensorboard --logdir='tensorboard_logs'
```
