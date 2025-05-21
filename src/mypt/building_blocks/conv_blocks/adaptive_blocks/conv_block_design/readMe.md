# Overview

The main objective is to write a simple algorithm that can design a convolutional block that produces an output of certain dimensions, given an input of certain dimensions.

My brief research on the topic of convolutional neural networks design concluded with a few design ideas/principles common across popular and influentials papers (e.g VGG, ResNet, Inception, ConvNEt): 

1. Using smaller kernels (usually 3 x 3 and 5 x 5) but modern ConvNet use 7 x 7 kernels. The limit is 7 x 7 kernels
2. Using pooling layers with at most 2 x 2 kernels (or even better convolutional layers with stride 2)
3. decreasing the kernel size as the networks goes deeper
4. most papers stack between 2 to 6 consecutive convolutional layers before applying a pooling or a strided convolutional layer 


The main idea of the algorithm is to find a combination of convolutional layers and pooling layers that produces an output of certain dimensions, given an input of certain dimensions. 

Let's write down important points: 

1. Convolutional layers are set of have a stride of 1, 0 padding and kernel size of either 3, 5, 7. Hence

$$
dim\_out = (dim\_in - kernel\_size + 1) 
$$

2. Similarly the input dimension of a convolutional layers can be computed by 

$$
dim\_in = dim\_out + kernel\_size - 1
$$

3. Pooling layers (or strided convolutional layers) both have a stride of 2 and a kernel size of 2 x 2. Hence 

$$
dim\_out = \lfloor\frac{dim\_in - 2}{2}\rfloor + 1
$$

4. There are 2 possible input dimension given a certain output dimension of a pooling layer: 

$$ 
dim\_in = 2 \times (dim\_out - 1) + 2 = 2 \times dim\_out
\\
\text{or}
\\
dim\_in = 2 \times (dim\_out - 1) + 3 = 2 \times dim\_out + 1
$$

5. Since we know that the kernels sizes can only be 3, 5, 7 and the number of consecutive convolutional layers can are limited to small range [n_min, n_max] (defaults to [2, 6]), we can compute all possible valid kernel combinations, and hence 

    * for a given input: compute all possible output dimensions (and the block that produces it) 
    * for a given output: compute all possible input dimensions (and corresponding blocks) 


6. Let's assume that a certain function that accepts a convolutional block and returns a cost exists. In other words, we can evaluate the quality of a convolutional block. 



7. Hence we can solve the problem using dynamic programming. Let's list the components: 

- $g(n\_min, n\_max, min\_kernel, max\_kernel, dim\_in)$: a function that returns a list of possible blocks of length between $n\_min$ and $n\_max$ with kernel size between $min\_kernel$ and $max\_kernel$ that produce $dim\_in$ (sorted)

- using $g$, we can find all possible $dim\_out$ for a given $dim\_in$ and vice versa. 


- This brings us the first part of the algorithm: the base cases:

f(dim, max_kernel_size) = represents the minimum cost of a convolutional block that produces an output of dimension $dim$ with a maximum kernel size of $max\_kernel\_size$ (the kernel sizes need to be sorted in descending order)

Given the final output dimension, we can find the minimum costs of all the blocks that produce this output dimension as well as the dimensions of the input to these blocks. 


Those dimensions and costs represent the bases cases. 


- Now for an arbitrary dimension $dim$, and max kernel size $k$, we proceed as follows: 

1. Generate all possible blocks

2. for each block $B$, compute the minimum kernel size in that block: $k_B$ 

3. for each block $B$ compute the output dimension: $dim_B$  

4. to compute $f(dim, k)$, we need to find $f(dim_B, ks)$ for all ks less than $K_b$ (since the kernel sizes are sorted in descending order) and their corresponding blocks.

5. after finding the best block for $(dim_B, ks)$, we build the new block by combining $B$ with the block in question and pass it to the cost function. 


We can find the block of decreasing filter size that produces the output (of given dimension) with the lowest cost using dynamic programming. 





