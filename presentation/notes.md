#Semantic Video Classification by Fusing Multimodal High-level Features (25 minutes)

1. Introduction

 * Video Classification
    * Rapidly growing video-recording capabilities
    * Amount of media content generated is very large
    * Need new effective methods to organize media (automatic recognition/tagging)
 * Motivation
    * Low-level vs high-level features (edges, colors, SIFT, BoW vs object, face, scene recognition)
     * Semantic gap -> "Attributes": high-level semantically meaningful representation
     * Objects as attributes for scene representation
     * Describe real world scenes by collecting the responses of many object detectors
2. Related Work
 * Object Bank
 * Action Bank
 * Dense Trajectories

3. Method Overview
  1. Frame extraction
  2. Feature fusion
  3. Classification w/ SVM

4. Details of the method
  1. Object Bank
  2. Action Bank
  3. Fusion methods
  4. Training of classifiers

5. Experiments & Results
  1. Choice of dataset
  2. Cross-validation
  3. Evaluation method

6. Conclusion
  1. AB & OB are complementary
  2. Promising results with more robust object & action detectors
  3. More potential with deep learning methods
