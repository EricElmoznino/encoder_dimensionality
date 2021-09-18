# Encoder Dimensionality

---
# Paper

## Outline

### Introduction

#### Draft 3
- CNNs are excellent models of visual cortex.
- CNNs work by transforming the manifold of images into a representation that is useful for visual tasks.
- A poorly understood property of CNN models of visual cortex is how their performance relates to the dimensionality of these manifolds.
- The dimensionality of a representation has significant implications for nearly every aspect of computation, including expressiveness, generalization, downstream decoding, and control (TODO: references for each). 
- There are disagreements about whether the dimensinoality of useful visual representations should be high or low.
- Here, we investigate the dimensionality of CNN models of visual cortex and how it relates to the quality of the model. Specifically, we compute both the effective dimensionality of diverse CNNs as well as their ability to explain neural activity in high-level regions of the ventral stream. 
- Our subsequent findings reveal a surprising computational principle underlying the performance of deep learning in visual neuroscience: performance scales as models embed visual features in a higher dimensional space.

#### Draft 2
- A central goal in the computational neuroscience of vision is to characterize the geometry of latent manifolds underlying neural population responses and, in particular, their dimensionality. The dimensionality of a representation has significant implications for nearly every aspect of computation, including expressiveness, generalization, downstream decoding, and control (TODO: references for each). A mechanistic theory of human vision and how it interacts with other neural processes, then, necessitates a better understanding of the dimensionailty of visual representations.
- Investigating neural dimensionality, however, is a difficult task.
    - Requires large naturalistic stimulus sets.
    - Recording tools have limited resolution.
    - Significant noise in both the recording devices and the neural responses.
    - Lack of a mechanistic model to understand how the dimensionality and manifold of the representation is transformed along the visual hierarchy.
- DNN models of visual cortex have achieved remarkable performance in explaining neural activity, and provide a mechanistic model for addressing scientific questions in-silico while sidestepping the above limitations.
- Here, we leverage deep CNNs to gain insight into the dimensionality of representations in visual cortex. Specifically, we compute both the effective dimensionality of diverse CNNs as well as their ability to explain neural activity in high-level regions of the ventral stream. Our subsequent findings reveal a surprising computational principle underlying the performance of deep learning in visual neuroscience: performance scales as models embed visual features in a higher dimensional space.

#### Draft 1
- Context:
    - Introduce CNNs as models of the ventral stream, and the idea of encoding models.
    - Discuss the fact that we still know very little about the representational properties of these models that underlie their ability to explain neural activity.
    - Dimensionality is an important geometric property that can largely determine the computations a representation subserves.
- Low dimensionality:
    - In the context of classification tasks, low-dimensionality is thought to be essential for good generalization on a fixed set of categories (TODO: references). Thus, it has been argued that both the ventral stream and computer vision models reduce the dimensionality of object representations along their hierarchy (TODO: references).
    - In addition, significant progress has been made in neuroscience by modeling neural dynamics using low-dimensional latent states, especially in the context of motor processing (TODO: references).
- High dimensionality:
    - On the other hand, high-dimensional representations are more expressive in that they can encode a greater quantity of information. 
    - In terms of theory, it has been argued that neural representations should be high-dimensional in order to subserve a large number of downstream tasks, where different linear readouts could be performed in different contexts (TODO: references). 
    - Similarly, there is also reason to believe that dimensionality should scale with the complexity of a task, and that the set of stimuli and behaviours in most neuroscience experiments is too constrained (TODO: references). 
    - In regards to vision, a recent theory of the ventral stream showed that in order for a representation to generalize to novel objects, the manifolds of those objects should be high-dimensional (The Geometry of Concept Learning).
    - Empirically, it has been found that the importance of category-orthogonal dimensions increases along the ventral stream (Explicit information for category-orthogonal object properties increases along the ventral stream).
    - Moreover, the eigenvalues of neural activity in mouse V1 were found to decay according to a power law, which results in representations that are as expressive and high-dimensional as possible yet still smooth and differentiable. Subsequently, (On 1/n neural representation and robustness) found that regularizing DNNs representations to increase their dimensionality resulted in significantly better robustness to adversarial attacks.
- Contribution:
    - Given the importance of dimensionlity for a representation and the vast numer of conflicting arguments surrounding it, we sought to empirically investigate the question by comparing computational models of visual cortex.
    - Specifically, we investigated the relationship between dimensionality and encoding performance across a wide array of CNNs trained on different tasks, datasets, and architectures, with the goal of answering a simple question: are models with high-dimensional representations better or worse at explaining neural activity in visual cortex?
    
### Background

### Methodology
- Explain intrinsic dimensionality, embedding dimensionality, and our effective dimensionality metric.
    - **FIGURE**: Show 3 plots, each with a 2D manifold in a 3D space.
        - First plot: Flat 2D plane. Intrinsic dimensionality is 2, embedding dimensionality is 2, and ED is 2.
        - Second plot: Slightly numpy 2D surface. Intrinsic dimensionality is 2, embedding dimensionality is 3, and ED is 2.x. ED weighs dimensions with high-variance larger, and is the continuous equivalent of picking a certain number of PCs to explain a certain percent of variance.
        - Third plot: A very curved 2D surface. Intrinsic dimensionality is 2, embedding dimensionality is 3, and ED is 3.
- Overview of method. Take multiple CNNs, and for each:
    1. Compute the ED of each layer on a large naturalistic dataset.
    2. Fit an encoding model from each layer to a brain region (in our case Monkey IT or human LOC).
    3. Assess the relationship between the ED of a representation and its performance in explaining activity in the brain.
    - **FIGURE**: Diagram depicting each of these components and how they connect, similar to what I had in my poster. Path for fitting encoding model, path for computing ED, and connections from them leading to a scatterplot of ED vs. encoding performance.
- Fitting encoding models. Discuss the dataset, and how we fit CNN activations to the neural data.
- Computing ED. Discuss the dataset, the spatial max-pooling, and give the equation. Discuss inter-class dimensionality and intra-class dimensionality, and why we might care about both (manifolds for classification vs. manifolds describing objects).
    - **FIGURE**: Show the distinction between global ED, inter-class ED, and inter-class ED with some examples manifolds.
- CNN models. Discuss where they come from, what datasets they were trained on, what tasks, the architectures.

### Results
- Dimensionality global/inter/intra correlate with encoding performance.
    - **FIGURE**: Main figure where we show the correlation between global/inter/intra dimensionality with encoding performance. Standard scatterplot.
- Dimensionality global/inter/intra increase along the hierarchy.
    - **FIGURE**: Normalize every trained model's dimenensionality to be between 0-1 across the layers. Plot all models together.
- Dimensionality correlates with encoding performance even within layers.
    - **FIGURE**: For ResNet18 and ResNet50 models, show a bar plot with layer depth on the x-axis and dim/encoding correlation on the y-axis, labeling significance.
- Training increases global/inter/intra dimensionality.
    - **FIGURE**: Show dimensionality across the layers with and without training.
- Higher dimensional models perform better at object classification. Specifically, dimensionality correlates best with prototype classification on novel object categories. Maybe discuss briefly in the context of The Geometry of Concept Learning.
    - **Figure**: Linear/Protytype/ImageNet/ImageNet21k dimensionality vs. classification performance plots.

### Discussion

### Supplementary
- **FIGURE**: global/inter/intra dimensionality by layer. There's a correlation within layers for inter/intra, but not really for global.
- **FIGURE**: global/inter/intra dimensionality without max-pooling.
 
## References
- Unsupervised neural network models of the ventral visual stream
    - Used their models
    - Show that models which transfer better to many visual tasks
      on OOD are better encoding models
-  Explicit information for category-orthogonal object properties increases along the ventral stream
    - To argue that intra-dimensionality increases
-  Performance-optimized hierarchical models predict neural responses in higher visual cortex
    - Early Yamins paper showing that object classification correlates with performance
-  Deep neural networks: A new framework for modeling biological vision and brain information processing
    - Early Kriegeskorte paper discussing object classification CNN encoding models
- The Geometry of Concept Learning
    - Argues that the ventral stream and IT are optimized for extracting representations suitable for few-shot prototype learning of novel objects, and discusses the computational properties that such representations would require.
- The V1 paper that finds eigenspectra decay with a power law to be as expressive as possible while still being smooth/differentiable
- The 1/n paper that found if you make the eigenspectra decay slower, models are more robust to adversarial attacks.
    - Also has lots of good references on dimensionality in neuroscience/ML in the sections starting with:
      - "In this work, we use adversarial robustness"
      - "Existing results in theoretical neuroscience point to the fact"
- TODO: papers on the usefulness of high-dimensional representations which I've read. Some of these might be referenced in the presentation I made for Mick.
- TODO: Sara Solla mentioned that someone at Stanford argues that the dimensionality of neural manifolds is actually large if you look at a large number of tasks.
- TODO: read Interpreting neural computations by examining intrinsic and embedding dimensionality of neural activity

## Notes
- Present all results on BrainScore, replicate figures on Object2Vec in appendix.
- Find paper that mentions out-of-distribution classification performance is a good predictor of encoding accuracy. If there is none, it will increase our contribution and allow us to emphasize this property more.
    - *Unsupervised neural network models of the ventral visual stream* makes the argument that models which transfer better to a wide variety of tasks (e.g. object classification, pose-estimation, etc.) on an OOD dataset are better models of visual cortex.
- Low-dimensionality not important for generalization, what's more important is that the manifold is flat (because then if you have a good basis set you can describe any point in the space). Would be interesting to see if the manifold was curved as well.
- ISOMAP or other intrinsic dimensionality neighbour like 2NN. Curvature can be computed as ratio between geodesic distances between pairs of points in ISOMAP vs. euclidean distances.

---
# To Do
- Histograms/KDEs of voxel-wise encoding accuracies as a function of dimensionality. Does the entire distribution shift to the right? Is the change restricted to the tails?
- Counter-example where dimensionality is high but performance is bad.
- Intrinsic dimensionality. Is this phenomenon specific to the embedding dimensionality only?
- ~~Look at dimensionality of our CNN models on the fMRI datasets we're fitting encoding models too. This is relevant for answering the question of what to make of the fact that the neural representations are low-dimensional, but the model representations that are good at explaining them are high-dimensional (but high-dimensional on a different dataset).~~
    - Our working hypothesis is that the neural representations are just low-dimensional because the neural dataset is small.
    - Have done this before, and it turned out that the CNNs had lower dimensionality, but still higher than the fMRI data and exhibited the same trend where higher dimensionality translated to better encoding performance.
- ~~Use new classes and images from "The Geometry of Concept Learning~~"
- ~~Try prototype vs. linear regression generalization error as explanatory factors for a number of different number of training examples. It could be that a model's capacity for few-shot prototype-learning generalization error is the best explanatory factor for model performance, in which case the equation from "The Geometry of Concept Learning" would provide the theoretical explanation as to why different geometrical properties of the representations (e.g. high dimensionality) are correlated with encoding performance.~~

---
# Notes

---
# Readings
