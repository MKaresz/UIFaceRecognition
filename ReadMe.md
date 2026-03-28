Android Application for Automatic Age, Gender & Expression (Smile) Recognition on an Edge Device
This project implements a mobile application capable of automatically recognizing age, gender, and facial expression from camera images.
The goal of the project is to deploy a complete inference pipeline—from model training to on‑device execution—on Android phones or tablets.
The system is inspired by real-world applications described in Kumar et al., 2022, where automated visual analysis can support:

guiding customers in retail environments
profiling individuals in large video datasets
routing patients in medical facilities

The app detects faces and runs three classifiers—age, gender, and expression—directly on the device using optimized machine‑learning models for edge deployment.
A MobileNetV2-based model was trained on the CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html by Liu, Ziwei; Luo, Ping; Wang, Xiaogang; and Tang, Xiaoou).

Key Features:
On-device inference (edge computing) Runs fully offline on Android using Kotlin, TensorFlow Lite, LiteRT, and MLKit for face detection and cropping.

Multiple inference model options:
Quantized INT8
Float16 (F16)
Float32 (F32)

Automatic face extraction prior to classification. by using MLKit (https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart)

Multi-output attribute classification
Predicts:
- Age groups: 0–19, 20–39, 40–59, 60–99
- Gender: male / female
- Expression: smiling / not smiling

How to use:
* Spinner menu - Choose the model from the top (I8Q, F16, F32)
* Take Picture - Button: Single or multi-face inference, predictions for all faces in an image are combined into a single formatted output. 
* Evaluate - Button: Runs inference on a fixed set of 20 images using the Evaluate button.

Evaluation dataset
A balanced set of 20 images randomly selected from the CelebA test set:
50% adults / 50% elderly
50% male / 50% female
50% happy / 50% sad

Results of the evaluation pictures on Motorola Moto G14 CPU (Android 14, Octa-core (2x2.0 GHz Cortex-A75 & 6x1.8 GHz Cortex-A55):  
-------------------------------------------------------------------------------------------------------------------  
I8Q_CELEBA_MODEL.TFLITE  
eval_set/182643.jpg: gender:Male age:20-39 smile:not Smiling >> Gender:0.99609375, age:0.99609375, smile:0.0, time:17  
eval_set/182648.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.9375, age:0.2890625, smile:0.16015625, time:17  
eval_set/182686.jpg: gender:Male age:20-39 smile:Smiling >> Gender:0.99609375, age:0.83984375, smile:0.99609375, time:18  
eval_set/182753.jpg: gender:Male age:20-39 smile:Smiling >> Gender:0.99609375, age:0.984375, smile:0.83984375, time:17  
eval_set/182797.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.99609375, age:0.40625, smile:0.98828125, time:18  
eval_set/182918.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.85546875, age:0.46875, smile:0.8046875, time:17  
eval_set/182967.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.99609375, age:0.00390625, smile:0.015625, time:17  
eval_set/182968.jpg: gender:Male age:60-99 smile:Smiling >> Gender:0.99609375, age:0.01171875, smile:0.9375, time:18  
eval_set/182986.jpg: gender:Male age:60-99 smile:Smiling >> Gender:0.99609375, age:0.31640625, smile:0.94921875, time:17  
eval_set/182988.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.99609375, age:0.04296875, smile:0.05078125, time:17  
eval_set/183309.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.53125, age:0.53125, smile:0.99609375, time:17  
eval_set/183346.jpg: gender:Female age:20-39 smile:Smiling >> Gender:0.0, age:0.78125, smile:0.98828125, time:17  
eval_set/183405.jpg: gender:Female age:60-99 smile:Smiling >> Gender:0.0, age:0.12890625, smile:0.98828125, time:18  
eval_set/183426.jpg: gender:Female age:20-39 smile:not Smiling >> Gender:0.0, age:0.99609375, smile:0.14453125, time:17  
eval_set/183434.jpg: gender:Female age:20-39 smile:not Smiling >> Gender:0.0, age:0.95703125, smile:0.12890625, time:17  
eval_set/183512.jpg: gender:Female age:40-59 smile:Smiling >> Gender:0.0, age:0.4375, smile:0.99609375, time:17  
eval_set/183534.jpg: gender:Female age:60-99 smile:Smiling >> Gender:0.01171875, age:0.265625, smile:0.9375, time:17  
eval_set/183566.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.83984375, age:0.2890625, smile:0.00390625, time:17  
eval_set/183578.jpg: gender:Female age:60-99 smile:Smiling >> Gender:0.03515625, age:0.14453125, smile:0.9921875, time:17  
eval_set/183603.jpg: gender:Female age:60-99 smile:not Smiling >> Gender:0.0, age:0.265625, smile:0.1015625, time:17  

-------------------------------------------------------------------------------------------------------------------  
F16_CELEBA_MODEL.TFLITE  
eval_set/182643.jpg: gender:Male age:20-39 smile:not Smiling >> Gender:0.9999754, age:0.99665916, smile:0.0018077183, time:31  
eval_set/182648.jpg: gender:Male age:40-59 smile:not Smiling >> Gender:0.8711648, age:0.34008896, smile:0.20737803, time:31  
eval_set/182686.jpg: gender:Male age:20-39 smile:Smiling >> Gender:0.9999926, age:0.8611016, smile:0.998538, time:31  
eval_set/182753.jpg: gender:Male age:20-39 smile:Smiling >> Gender:0.99999595, age:0.98497325, smile:0.93014014, time:30  
eval_set/182797.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.9999593, age:0.45185977, smile:0.981135, time:31  
eval_set/182918.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.8218767, age:0.59313107, smile:0.8539336, time:30  
eval_set/182967.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.9999953, age:0.005794643, smile:0.02232888, time:31  
eval_set/182968.jpg: gender:Male age:60-99 smile:Smiling >> Gender:0.9999716, age:0.011585699, smile:0.96627253, time:31  
eval_set/182986.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.99999756, age:0.53301793, smile:0.95840573, time:31  
eval_set/182988.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.99997795, age:0.12733997, smile:0.05413746, time:31  
eval_set/183309.jpg: gender:Female age:40-59 smile:Smiling >> Gender:0.4720846, age:0.5735346, smile:0.9938124, time:31  
eval_set/183346.jpg: gender:Female age:20-39 smile:Smiling >> Gender:4.0341753E-8, age:0.8852125, smile:0.9916299, time:30  
eval_set/183405.jpg: gender:Female age:60-99 smile:Smiling >> Gender:2.2322693E-5, age:0.20992312, smile:0.98093516, time:30  
eval_set/183426.jpg: gender:Female age:20-39 smile:Smiling >> Gender:2.1156088E-6, age:0.99896866, smile:0.54401654, time:30  
eval_set/183434.jpg: gender:Female age:20-39 smile:not Smiling >> Gender:2.428172E-4, age:0.9634616, smile:0.13967668, time:32  
eval_set/183512.jpg: gender:Female age:40-59 smile:Smiling >> Gender:4.7133535E-6, age:0.39224064, smile:0.999389, time:31  
eval_set/183534.jpg: gender:Female age:40-59 smile:Smiling >> Gender:0.004014741, age:0.3637242, smile:0.843292, time:30  
eval_set/183566.jpg: gender:Female age:40-59 smile:not Smiling >> Gender:0.44523942, age:0.47175023, smile:0.012552142, time:31  
eval_set/183578.jpg: gender:Female age:60-99 smile:Smiling >> Gender:0.008039198, age:0.15623383, smile:0.992935, time:31  
eval_set/183603.jpg: gender:Female age:60-99 smile:not Smiling >> Gender:8.6255756E-4, age:0.27896988, smile:0.17616543, time:31  

-------------------------------------------------------------------------------------------------------------------  
F32_CELEBA_MODEL.TFLITE  
eval_set/182643.jpg: gender:Male age:20-39 smile:not Smiling >> Gender:0.9999747, age:0.9966175, smile:0.0017946248, time:31  
eval_set/182648.jpg: gender:Male age:40-59 smile:not Smiling >> Gender:0.868739, age:0.33960834, smile:0.20480789, time:31  
eval_set/182686.jpg: gender:Male age:20-39 smile:Smiling >> Gender:0.9999928, age:0.858689, smile:0.99852264, time:30  
eval_set/182753.jpg: gender:Male age:20-39 smile:Smiling >> Gender:0.9999958, age:0.984916, smile:0.9292957, time:31  
eval_set/182797.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.9999586, age:0.45475018, smile:0.9814773, time:31  
eval_set/182918.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.8218641, age:0.59241605, smile:0.8547178, time:31  
eval_set/182967.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.9999954, age:0.0058569815, smile:0.022162259, time:31  
eval_set/182968.jpg: gender:Male age:60-99 smile:Smiling >> Gender:0.99997216, age:0.01158898, smile:0.96604884, time:32  
eval_set/182986.jpg: gender:Male age:40-59 smile:Smiling >> Gender:0.99999756, age:0.5292115, smile:0.9575375, time:31  
eval_set/182988.jpg: gender:Male age:60-99 smile:not Smiling >> Gender:0.9999773, age:0.12524192, smile:0.051229656, time:31  
eval_set/183309.jpg: gender:Female age:40-59 smile:Smiling >> Gender:0.4781532, age:0.5735521, smile:0.9938683, time:31  
eval_set/183346.jpg: gender:Female age:20-39 smile:Smiling >> Gender:3.923519E-8, age:0.88713527, smile:0.9916427, time:31  
eval_set/183405.jpg: gender:Female age:60-99 smile:Smiling >> Gender:2.1953325E-5, age:0.21170932, smile:0.98068845, time:31  
eval_set/183426.jpg: gender:Female age:20-39 smile:Smiling >> Gender:2.105147E-6, age:0.99895144, smile:0.54237944, time:30  
eval_set/183434.jpg: gender:Female age:20-39 smile:not Smiling >> Gender:2.3793388E-4, age:0.9637578, smile:0.13851322, time:31  
eval_set/183512.jpg: gender:Female age:40-59 smile:Smiling >> Gender:4.5943925E-6, age:0.39012277, smile:0.99940354, time:31  
eval_set/183534.jpg: gender:Female age:40-59 smile:Smiling >> Gender:0.0040014577, age:0.3614857, smile:0.8420006, time:30  
eval_set/183566.jpg: gender:Female age:40-59 smile:not Smiling >> Gender:0.45049405, age:0.46664852, smile:0.012481222, time:31  
eval_set/183578.jpg: gender:Female age:60-99 smile:Smiling >> Gender:0.0081842635, age:0.15691839, smile:0.9927054, time:31  
eval_set/183603.jpg: gender:Female age:60-99 smile:not Smiling >> Gender:8.9578936E-4, age:0.27810448, smile:0.17400137, time:31  

## Technologies Used
* Tensorflow training, Kotlin (Android)
* TensorFlow training & Lite, LiteRT
* ML Kit for face detection**

This project demonstrates a fully functional mobile implementation of an age, gender, and expression recognition system, 
deployed entirely on-device. It highlights the feasibility of edge-based visual recognition for real-world applications 
such as retail assistance, healthcare routing, and large-scale video analysis evan in realtime.
