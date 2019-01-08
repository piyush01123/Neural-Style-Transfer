### Gatys' Algorithm
1. Start with a content image C, a style image S and a random image G
2. Use a pretrained CNN model (authors used VGG) graph  in Tensorflow and mark first 4 conv nodes
3. Define your content cost as the L2 norm between C and G
4. Pass S and G to the graph and define style cost as the weighted sum of L2 norms of
gram matrices at those 4 conv nodes between S and G
4. Define the cost function as sum of the content and style costs and the optimizer using Adam.
5. Update the generated image by minimizing the cost.
6. Repeat step 5 for some finite number (1000) of times after which hopefully we have a stylized image.


### Johnson's Algorithm (only works for fixed style image S)
1. We initialize a image transformation network N that produces the same shape output as input like a FCN
##### Training (Use MiniBatch SGD with Adam)
2. Pass C through N to get a G.
3. Pass C, S and G through VGG (We are going to use VGG as a loss network only)
4. Define Feature reconstruction loss as L2 norm between features of layers of C and G. This works as content cost
5. Define Style reconstruction loss as L2 norm between gram matrices of features of layers of S and G. This works as style cost.
(The authors call these losses as perceptual loss functions)
6. Resize each image in training dataset (80k from MS-COCO) to 256x256 and train N with batch size of 4 (thus making 20k batches) for 40 k iterations (thus 2 epochs for the entire dataset). Dont pass any batch through the network more than twice (We dont want N to tune to specific C images).
##### Inference
7. Any new image C can be stylized just by passing it through N. This makes it fast and deployable on devices like Android.

