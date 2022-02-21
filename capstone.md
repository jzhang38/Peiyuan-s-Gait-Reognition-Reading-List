<h1 align="center">Peiyuan's Gait Recognition Reading List</h1>

A place to keep track of the Gait Recognitionn papers I read, starting from 2022 Feb.
I note down every paper that I've read, but only summarize and make comments for those I find interesting.


# Computer Vision --  Gait Recognnintion 


+ [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition]()<br>
1. As a periodic motion, gait can be represented as a single period. Within one period, it was observed that the silhouette in each position has unique appearance. Thus, the author assume the appearance of a silhouette has contained its position information.
2. Input: a set a gait silhouettes --> use CNN to extract frame level features --> Set pooling to obtain set level features --> Horizontal Pyramid Mapping
3. each perid will finally generate a 15872 dimensional representation to represent this person. During testing, Elidean distance will be calculated and the rank 1 result will be the output. 

+ [Gait Lateral Network: Learning Discriminative and Compact Representations for Gait Recognition]() <br>
1. We notice that the silhouettes for different subjects only have subtle differences in many cases, which makes it vital to explore the shallow features encoding the local spatial structural information for gait recognition.
2. learn a compact and discriminative representation for each subject.


+ [Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation]() <br>
1. Motivation: However, the representations based on global information often neglect the details of the gait frame, while local region based descriptors cannot capture the relations among neighboring regions, thus reduc- ing their discriminativenes
2. Global and Local Feature Extractor(GLFE): composed of multiple Global and Local Convolutional layers (GLConv)
3.  Local Temporal Aggregation (LTA) 
4. Use both Triplet Loss and Cross Entropy Loss
+ [In defense of the triplet loss for person re-identification]() <br>



+ [RealGait: Gait Recognition for Person Re-Identification]() <br>
1. The difference between ReID and Gait recognition
2. Current popular datasets have major drawbacks: they are collected in indoor environment where the subjects are INSTRUCTED to walk instead of walking freely.
3. The author convert an existing ReID into a Gait Recognition dataset by extracting silhouettes.
4. SOTA model on CASIA-B such as GaitGL and GaitSet have poor performance in the wild.

+ [Learning Rich Features for Gait Recognition by Integrating Skeletons and Silhouettes]() <br>

+ [RFMask: A Simple Baseline for Human Silhouette Segmentation with Radio Signals]()<br>

+ [End-to- end model-based gait recognition]()<br>
1. What is model-based gait recognition: try to model the 3 dimensional moving pattern directly from skeletonns.

+ [Multi-Camera Trajectory Forecasting: Pedestrian Trajectory Prediction in a Network of Cameras]()<br>
1. Multi-Camera Trajectory Forecasting
2. Tracking objects (pedestrians) across a large camera net- work requires simultaneously running state-of-the-art algo- rithms for object detection, tracking, and RE-ID.  A successful MCTF model can address this issue by preempting the location of an object-of-interest in a distributed camera network, thereby enabling the system to monitor only selected cameras intelligently. 

+ [ Simple online and realtime tracking with a deep association metric]()<br>


+ [Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking]()<br>
0. Perform object detection, feature extraction, and tracking in one go.
1. The input is two adjacent frame (so called a node), generate paired bounding boxes corresponding to the two frames. They use simple IOU score to track the bounding boxes by comparing the same frame's boxes in different nodes.
2. Feature encoder: ResNet-50 + FPN
3. Paired Box Regression: 
+ [Mask R-CNN]() <br>


+ [Faster R-CNN]()<br>
1. Region Proposal Network share parameter with the detection network.
2. introduce anchor box.

+ [Fast R-CNN]()<br>

+ [Fully convolutional networks for semantic segmentation.]() <br>
+ [MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation]()<br>
1. Per-pixel classification: every pixel conduct multi-class classification.
2. Mask Classification: Different mask is generated for different class. There is a per-pixel BINARY classification loss and per mask classificationn loss.
+ [Masked-attention Mask Transformer for Universal Image Segmentation]()<br>
1. 


+[Mask2Former for Video Instance Segmentation]() <br>

+ [Gait Recognition via Disentangled Representation Learning]()<br>
# Computer Vision --  Person Re-Identification
Person ReID basically has the same objective as Gait Recognition. The only difference is that the model input for ReID is a image, while for Gait Recognition it is a video.
+ [Horizontal pyramid matching for person re-identification]()<br>
1. Motivation: missing body part may greatly influennce the model performance for model exploitinng global representation.
2. Use Cross-Entropy Loss.
+ [Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking]
+ [A Discriminatively Learned CNN Embedding for Person Re-identification]() <br>
1. Verification model: take a pair of images as imput and formuate the task as a binary classification. Use triplet loss. Drawbacks: they only use weak re-ID labels, and do not take all the annotated information into consideration. Therefore, the verification network lacks the consideration of the relationship between the image pairs and other images in the dataset.
2. Identification model: formulate the task as multi-class classification. Drawbacks: During testing, the feature is extracted from a fully connected layer and then normalized. The similarity of two images is thus computed by the Euclidean distance between their normalized CNN embed- dings. The major drawback of the identification model is that the training objective is different from the testing procedure, i.e.,it does not account for the similarity measurement between image pairs, which can be problematic during the pedestrian retrieval process.
3. We find that the contrastive loss leads to over-fitting when the number of images is limited.

# Ideas
1. There is a gap between training loss and tesing. Instead, we can always choose a sample as probe during training and calculate cross entropy loss.
GaitGL without crossEntropyLoss 
2. Pretrain out Gait Recognition model of Person ReID dataset. 
3. Contrastive Learning with data augmentation
4. Ramdom Sample a fixed length 的影响
5. Few shot learnin: train a small model to change the last few layer's parameter? 
6. Use existing large scale video dataset such as dataset for activity detection. -- > do human annotation. --> this is only single view
