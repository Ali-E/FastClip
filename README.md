## PowerQR and FastClip

In this repository, you can find the implementation of our methods for extracting and clipping the spectral norm of convolutional layers, dense layers, batch norm layers, and their concatenation. The details of our methods and the results of our experiments are presented in our paper. We have included a simple interface and tutorial for using our clipping methods which can be used as an effective regularization method in deep neural networks to improve their generalization and make them more robust against adversarial attacks.


## Setting up the environment:


First create a new virtual environment:

```
python -m venv fastclip
```

and then install all the needed libraries with the versions used for producing the results of the paper:

```
python3 -m pip install -r requirements.txt
```

## SimpleConv model:

For replicating the experiments of the paper that was done using the simple model with one convolutional layer and one dense layer, we have prepared an easy to use jupyter-notebook named ```simple_conv_results.ipynb```. 


## Simple tutorial:

For a simple illustration of how to incorporate our methods in your model and how to use that for composition of convolutional and batch norm layers, please refer to the jupyter-notebook named ```how_to_use.ipynb```.

## How to train a model with a clipping method:

To perform other experiments of the paper, navigate to the ```experiments``` folder. Other than the simple convolutional model presented in the provided notebooks, our experiments are done using two other models, ResNet18 and DLA, which can be defined using the argument ```--model``` at each step of the pipeline.

### Clipping methods:
The clipping methods are:

```fastclip```: FastClip (ours)

```gouk```: Gouk et al. (2021)

```miyato```: Miyato et al. (2018)

```nsedghi```: Senderovich et al. (2022)

```lip4conv```: Delattre et al. (2023)

```all```: run for all the methods

### Modes: 
There are 3 different modes you can use:

```wBN```: with batch normalization layers intact.

```noBN```: with all batch normalization layers removed.

```clipBN```: using the clipping method introduced by Gouk et al. (2021)

```all```: run for all the 3 modes


To run a model once using one of the clipping methods use the command below:

```
python batch_job_submit.py --method fastclip --model ResNet18 --seed 1 --mode wBN
```
If you do not specify a seed value the experiment will run for each of the seeds in [10**i for i in range(10)]:

```
python batch_job_submit.py --method fastclip --model ResNet18 --mode wBN
```


## How to evaluate a model:

To evaluate the accuracy of a trained model on the test set and an adversarial attack, you can simply use:

```
python batch_attack_submit.py cifar logs/cifar/ResNet18_models/method_ --method fastclip --mode wBN --attack pgd --model ResNet18 --seed 1
```
The first argument is the dataset (```cifar``` or ```mnist```) and the second argument is the common part of the address to the trained model (up to the name of the methods). The command above uses PGD attack with the default parameters to generate adversarial examples for the model. You can change it to CW for Carlini-Wagner attack. For each of the seeds, a csv file will be created with the name of the applied attack and its parameters inside the corresponding directory.

### Note:

If you have installed the packages and still get this error:

```
ImportError: cannot import name 'zero_gradients' from 'torch.autograd.gradcheck'
```

It is because of a minor version mismatch between torch and advertorch libraries. You can resolve this by the following simple modifications:

1. locate the file below in your virtual environment (located in your home directory by default):

    ```address_to_environment/fastclip/lib/python3.10/site-packages/advertorch/attacks/fast_adaptive_boundary.py```

2. Comment out line 14: ```from torch.autograd.gradcheck import zero_gradients```
3. replace line 85: ```zero_gradients(im)``` with ```im.zero_grad()```
4. deactivate and re-activate your environment.


### Collecting attack results:

To collect the attack results from the files that are generated for each seed, you can use this command:

```
python collect_attack_results.py logs/cifar/ResNet18_models/method_ --method fastclip --mode wBN
```
This will generate 2 csv files in the parent directory, one with the results from all the methods and modes and seed and the other with the results from all models and modes averaged among different seeds. 

## Computing Layer-wise spectral norms:

To compute the spectral norm of each convolutional layer of a trained model with either of the clipping methods, you can use the command below:

```
python compute_trained_SVs.py cifar fastclip logs/cifar/ResNet18_models/method_ --seed 1
```

It will generate a file in the parenting directory of the methods (```logs/cifar/ResNet18_models``` in this case) that has the average value and standard deviation of spectral norm of each layer of the model averaged among all the seeds (if you do not pass a specific seed as the argument). By using ```all``` as the method name (second argument), it will run this for all the methods.

Also, FastClip is able to keep track of the spectral norm of all of its layers during the training by default, which can be observed by using the following command:

```
tensorboard --logdir logs/cifar/ResNet18_models/method_fastclip__1/
```

You can replace the argument with the address to the other models.


## References:

For the details of our methods, comprehensive results and discussions, please refer to our [paper](https://arxiv.org/abs/2402.16017) (also in the proceedings of AISTATS 2024).



Here is the list of papers for the clipping methods we used in our comparisons:

1. Miyato, Takeru, et al. "Spectral normalization for generative adversarial networks." arXiv preprint arXiv:1802.05957 (2018).
2. Gouk, Henry, et al. "Regularisation of neural networks by enforcing lipschitz continuity." Machine Learning 110 (2021): 393-416.
3. Senderovich, Alexandra, et al. "Towards practical control of singular values of convolutional layers." Advances in Neural Information Processing Systems 35 (2022): 10918-10930.
4. Delattre, Blaise, et al. "Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration." arXiv preprint arXiv:2305.16173 (2023).

For the clipping method by Miyato et al., we used the pytorch implementation from this repository: https://github.com/christiancosgrove/pytorch-spectral-normalization-gan  

For the clipping methods by Senderovich et al. (2022) and Gouk et al. (2021) we used the code provided by Senderovich et al: https://github.com/WhiteTeaDragon/practical_svd_conv 

For the clipping method by Delattre et al. (2023), we used their code: https://github.com/blaisedelattre/lip4conv


For the models (ResNet18 and DLA) and training them on cifar-10 we used codes from this repository: https://github.com/kuangliu/pytorch-cifar/tree/master

For the adversarial attacks and and MNIST data, we used the code from this repository: https://github.com/AI-secure/Transferability-Reduced-Smooth-Ensemble/tree/main 

