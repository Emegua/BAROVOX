# Covert Channel Attack on Pressure Sensors


## To Setup the Environment
Please use [Anaconda](https://www.anaconda.com/) for virtual environment management ([install guide](https://www.anaconda.com/products/individual-d)).
Then, run the command to initialize/activate a conda env: 
```
$ conda create --name [env_name] python=3.9
$ conda activate [env_name]
```
To install requirements, run the following commands: 
```
$ sudo apt-get install graphviz libgraphviz-dev pkg-config (ask your system manager to do so).
$ conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
$ conda install pyg -c pyg -c conda-forge
$ python -m pip install numpy r2pipe wandb seaborn pandas click 
```

## Usages
### 1. Setup wandb integration for visualizing the results
Our project incoporates Weight & Bias ([wandb](https://wandb.ai/site)) that allows users to visualize and track machine learning training in real-time. It can be easily integrated with Python programs that leverage popular deep learning frameworks like Pytorch, Tensorflow, or Keras. Go through the following steps: 

1. Sign up for a free account on [wandb website](https://wandb.ai/site).
2. Verify if wandb is properly installed in your work environment.
    ```
    $ wandb --version
    ```
3. Login into your wandb account
    ```
    $ wandb login
    ```
    It will ask you to provide an API from your wandb profile, click on the link and copy the API and paste it here.

4. We parameterize the wandb usage, so make sure you pass the correct arguments when using wandb in our scripts. 
- `--wandb_enable` - use this to enable wandb. It doesn't need a value. \
- `--wandb_project` - this is the name of the project where you are sending the new run. You should first create the project using the [Quickstart guide](https://docs.wandb.ai/quickstart).
- `--wandb_entity` - this specifies the username or team name where you're sending runs. This entity must exist before you can send runs there, so make sure to create your account or team in the UI before starting to log runs.
    
*NB*: *Wandb is enabled in our project by default, but users can disable it by passing a command line argument when they trian/evaluate our models. Please refer to step 4.*


### 2. Model Training/Evaluation
- Users can specify the following parameters
    ```--model
    --dta_path
    --data_type
    --n_fft
    --sample_rate
    --wandb_enable
    --wandb_project
    --wandb_entity
    --lr
    ```

    You can train and evaluate the graph learning method as follows: 
    
    ```
    $ python train.py --model  [cnn|resnet] --data_type [mnist|speechcommand] --train --n_fft=[256|128|512|...] --lr=[0.001|...] --wandb_enable --wandb_project [wandb project name] --wandb_entity [wandb project entity]
    ```
    Users can adjust other parameters as well. You may found the complete list of command line arguments here.  

    `learning_rate` #  The initial learning rate for GCN.\
    `epochs` # Number of epochs to train.\
    `layers` # Number of hidden units.\
    `dropout` # Dropout rate (1 - keep probability).\
    `batch_size` # Number of graphs in a batch.\
    `device` # The device to run on models (cuda or cpu) cpu in default.\
    `num_layers` # Number of layers in the GCN.\
    `test_step` # The interval between mini evaluation along the training process.\
    `num_classes` # number of classes. 3 while using cfg, 4 while using cg
    `precache_path` # Path to networkx graphs.\
    `gt` # [cg|cfg]\
    `print_graphs` # [yes|no]\
    `wandb_enable`\
    `wandb_project` # wandb project name.\
    `wandb_entity` # wandb project entity.

    Example 1: Train ResNet model:
    ```
    python train.py --train --model="resnet" --data_type "speechcommand" --wandb_enable --n_fft=256 --sample_rate=2000 --model_path="saved_model/spect/test_model" --epoch 20 --lr 0.0001 --wandb_entity cca-pressure --wandb_project finetuning
    ```

    Example 2: Train CNN model
    ```
    python train.py --train --model="cnn" --data_type "speechcommand" --wandb_enable --sample_rate=2000 --model_path="saved_model/spect/test_model_cnn" --epoch 20 --lr 0.0001  --wandb_entity cca-pressure --wandb_project finetuning
    ```