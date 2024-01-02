Trained a ResNet on the TensorFlow flowers dataset on standard and corrupted labels (mislabelled features) to test the effects of mislabelling.


Currently, only binary classification is supported although I am working on a multiclass variant.

This project has the goal to explore the effects of training a model on corrupted, inaccurate labels. For these purposes, the project will consist of binary image classification through a ResNet on the TensorFlow flowers dataset. This dataset consists of 5 different types of flowers: dandelions, daisys, tulips, sunflowers, and roses. In hommage to the Spiderman Across the Spiderverse movie that came out this year and the song Sunflower by Swae Lee, I've decided to simplify the datasets to flowers which are sunflowers and those which are not for binary classification.

The ResNet network will be the default ResNet50 that comes prepackaged with TensorFlow, which consists of 50 total layers, 48 of which are convolutional layers. The ResNet model has been chosen because it comes pretrained on ImageNet, which makes it already excellent at image classification and therefore, will make the effects of corrupted data more apparent. The experiment will consist of training a control ResNet model on the standard dataset, then training identical ResNet models on corrupted datasets and comparing the decrease in accuracy. Finally, we will explore a robust learning technique: using the L1 loss: we will train a final set of ResNets on the corrupted datasets but now use the L1 loss to see its results. The L1 loss is considered more robust because it grows slower and therefore would theoretically have less changes per each corrupted label. 

Results indicate that the standard ResNet trained on the TensorFlow flowers dataset obtains a sunflower classification accuracy of 94.79% while the ones trained on increasingly more corrupt datasets have significantly worse accuracies that range from 0 to 66.66%. Finally, ResNets trained on the corrupted datasets but instead using L1 loss fare slightly better, with accuracies that range from 59.38 to 79.17%. These were all trained on the same sample of training images, although with different levels of corruption, and tested on the same validation set, which consists of 100 images the models have never seen before.

Remark: training all 11 models take roughly 6-7 (on my machine each model takes roughly 30 mins to train through 5 epochs) and unfortunately, the pretrained .h5 models cannot be submitted because each file takes roughly 300mb of space. This is with a reduction of the training set to 600 images instead of almost 3000.


![image](https://github.com/JiaFengYu/corrupted_resnet/assets/48167665/c1f22820-48d7-48c6-95fc-60327ac1de68)
