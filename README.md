# Encoder Dimensionality

---
# Paper

## Outline
- d

## References
- Unsupervised neural network models of the ventral visual stream
    - Used their models
    - Show that models which transfer better to many visual tasks
      on OOD are better encoding models
-  Explicit information for category-orthogonal object properties 
   increases along the ventral stream
    - To argue that intra-dimensionality increases
-  Performance-optimized hierarchical models predict neural 
   responses in higher visual cortex
    - Early Yamins paper showing that object classification
      correlates with performance
-  Deep neural networks: A new framework for modeling biological 
   vision and brain information processing
    - Early Kriegeskorte paper discussing object classification 
      CNN encoding models
- The Geometry of Concept Learning
    - Argues that the ventral stream and IT are optimized for
      extracting representations suitable for few-shot prototype learning 
      of novel objects, and discusses the computational properties
      that such representations would require.
- TODO: papers on the usefulness of high-dimensional representations which
  I've read. Some of these might be referenced in the presentation I made for Mick.

## Notes
- Present all results on BrainScore, replicate figures on Object2Vec in
  appendix.
- Find paper that mentions out-of-distribution classification performance
  is a good predictor of encoding accuracy. If there is none, it will 
  increase our contribution and allow us to emphasize this property more.
    - *Unsupervised neural network models of the ventral 
      visual stream* makes the argument that models which transfer better to
      a wide variety of tasks (e.g. object classification, pose-estimation, etc.)
      on an OOD dataset are better models of visual cortex. 

---
# To Do
- Look at dimensionality of our CNN models on the fMRI datasets we're fitting
  encoding models too. This is relevant for answering the question of what to
  make of the fact that the neural representations are low-dimensional, but
  the model representations that are good at explaining them are high-dimensional
  (but high-dimensional on a different dataset).
    - Our working hypothesis is that the neural representations are just 
      low-dimensional because the neural dataset is small.
    - Have done this before, and it turned out that the CNNs had lower
      dimensionality, but still higher than the fMRI data and exhibited
      the same trend where higher dimensionality translated to better
      encoding performance.
- ~~Use new classes and images from "The Geometry of Concept Learning~~"
- ~~Try prototype vs. linear regression generalization error as 
  explanatory factors for a number of different number of training
  examples. It could be that a model's capacity for few-shot
  prototype-learning generalization error is the best explanatory
  factor for model performance, in which case the equation
  from "The Geometry of Concept Learning" would provide the
  theoretical explanation as to why different geometrical 
  properties of the representations (e.g. high dimensionality)
  are correlated with encoding performance.~~

---
# Notes

---
# Readings
