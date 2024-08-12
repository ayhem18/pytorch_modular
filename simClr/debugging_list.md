# How to evaluate: 
1. Train a model with SimClr
2. save the checkpoint
3. iterate through the dataset, for each sample find the closest samples using cosine similarity
4. 



# Iteration 1: 
Alexnet with STL10 dataset
* batch size: 256
* learning rate 0.1, using SGD with the annealing learning rate scheduler

# Evaluation results: 

* Best training loss after 20 epochs: 5
* Semantically such a high loss is a problem: 

    1. ![img](./images/iteration_1_nearest_neighbors_res_1.png)
    2. ![img](./images/iteration_1_nearest_neighbors_res_2.png)

The observations: 

* the model might be using the color as a shortcut. The solution would be to use heavier color transformations

* It is possible that all images are close to each others. As we can see the similarity measures are very close for the nearest 5 neighbors. One possible solution is to use the dot product instead of the cosine similarity. This can also be debugged by tracking the variance of similarities across training

* The paper authors use the LARS optimizer. The latter is claimed to be more suitable for large batch training. The learning rate is set to 0.3 * (batch size / 256). 
Using this setting might lead to more stable training

* easier ?? dataset
* different backbone maybe (Vision Transfer)


# Iteration 2: 
