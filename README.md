# Goal-Conditioned Neural ODEs with Guaranteed Safety and Stability for Learning-Based All-Pairs Motion Planning

This repository contains the code for our CDC2026 submission: [Goal-Conditioned Neural ODEs with Guaranteed Safety and Stability for Learning-Based All-Pairs Motion Planning](https://arxiv.org/html/2604.02821v1) (Dechuan Liu, Ruigang Wang, and Ian R. Manchester). 

This code has been tested with Python 3.12.3.

## A few results
We learn a bi-Lipschitz diffeomorphism g to encode the geometry of obstacles.
![Bi-Lipschitz Diffeomorphism](/results/2D-corridor/learnt_model.png)

We construal goal-conditioned neural ODE, which induces the safe, stable, all-pairs motion planning with formal guarantees.
![Motion Planning](/results/2D-corridor/demo-video/trajectory_and_point_multi_(0.6875,%200.4375).gif)

## Installation and Setup
The neural network used to construct bi-lipschitz diffeomorphism mapping is based on [Bi-lipschitz Neural Network](https://github.com/acfr/RobustNeuralNetworks/tree/main). Please install as stated.

Please also install the packages listed in `requirements.txt`

## Organisation of this Repository
This repositroy is structured as follows.

`src/`: contains all source code used to run experiments, process results, and generate plots.

`results/`: contains all plots and saved model weights used to produce the main results figures in the paper.

## Usage
Run following script to generate the dataset by RRT:
```
python src/dataset_generation.py 
```

Run following script to train and save the models: 
```
python src/train_model.py
```

Run following script to visualize the results: 
```
python src/visualization.py
```

## Contact
For any questions or bugs, please raise an issue or contact Ruigang (Ray) Wang (ruigang.wang@sydney.edu.au) or Dechuan Liu (dechuan.liu@sydney.edu.au)

