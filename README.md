# Embedded Machine Learning Courserware

Welcome to the Edge Impulse open courseware for embedded machine learing! This repository houses a collection of slides, reading material, project prompts, and sample questions to get you started creating your own embedded machine learning course. You will also have access to videos that cover much of the material. You are welcome to share these videos with your class either in the classroom or let students watch them on their own time.

Please note that the content in this repository is not inteded to be a full semester-long course. Rather, you are encouraged to pull from the modules, rearrange the ordering, make modifications, and use as you see fit to integrate the content into your own curriculum.

Content is divided into separate *modules*. Each module is assumed to be about a week's worth of material, and each section within a module contains about 60 minutes of presentation material. Modules also contain example quiz/test questions, practice problems, and hands-on assignments.

## License

Unless otherwise noted, slides, sample questions, and project prompts are released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are welcome to use and modify them for educational purposes.

The YouTube videos in this repository are shared via the standard YouTube license. You are allowed to show them to your class or provide links for students (and others) to watch.

## Professional Development

Much of the material found in this repository is curated from a collection of online courses with permission from the original creators. You are welcome to take the courses (as professional development) to learn the material in a guided fashion or refer students to the courses for additional learning opportunities.

 * [Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning) - Coursera course by Edge Impulse that introduces neural networks and deep learning conceps and applies them to embedded systems. Hands-on projects rely on training and deploying machine learning models with Edge Impulse. Free with optional paid certificate.
 * [Copmuter Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning) - Follow-on Coursera course that covers image classification and object detection using convolutional neural networks. Hands-on projects rely on training and deploying models with Edge Impulse. Free with optional paid certificate.
 * [Tiny Machine Learing (TinyML)](https://www.edx.org/professional-certificate/harvardx-tiny-machine-learning) - EdX course by [Vijay Reddi](https://scholar.harvard.edu/vijay-janapa-reddi), [Laurence Moroney](https://laurencemoroney.com/), [Pete Warden](https://petewarden.com/), and [Lara Suzuki](https://larissasuzuki.com/). Hands-on projects rely on Python code in Google Colab to train and deploy models to embedded systems with TensorFlow Lite for Microcontrollers. Paid course.

## Prerequisites

Students should be familiar with the following topics to complete the example questions and hands-on assignments:

* **Algebra**
   * Solving linear equations
* **Probability and Statistics**
   * Expressing probabilities of independent events
   * Normal distributions
   * Mean and median
* **Programming**
   * Arduino/C++ programming (conditionals, loops, arrays/buffers, pointers, functions)
   * Python programming (conditionals, loops, arrays, functions, NumPy)

*Optional prerequisites*: many machine learning concepts can be quite advanced. While these advanced topics are briefly discussed in the slides and videos, they are not required for quiz questions and hands-on projects. If you would like to dig deeper into such concepts in your course, students may need to be familiar with the following:

* **Linear algebra**
  * Matrix addition, subtraction, and multiplication
  * Dot product
  * Matrix transposition and inversion
* **Calculus**
  * The derivative and chain rule are important for backpropagation (a part of model training)
  * Integrals and summation are used to find the area under a curve (AUC) for some model evaluations
* **Digital signal processing (DSP)**
  * Sampling rate
  * Nyquist–Shannon sampling theorem
  * Fourier transform and fast Fourier transform (FFT)
  * Spectrogram
* **Machine learning**
  * Logistic regression
  * Neural networks
  * Backpropagation
  * Gradient descent
  * Softmax function
  * K-means clustering
* **Programming**
  * C++ programming (objects, callback functions)
  * Microcontrollers (hardware interrupts, direct memory access, double buffering, real-time operating systems)

## Learning Objectives

By the end of the modules, students should be able to:


3. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)

7. Describe the requirements for collecting a good dataset and what factors can create a biased dataset
8. Demonstrate the ability to collect a dataset using a variety of sensors (e.g. accelerometer, microphone, camera)
9. Describe broadly what happens during machine learning model training
10. Describe the difference between model training and inference
11. Describe why test and validation datasets are needed
12. Evaluate a model's performance by calculating accuracy, precision, recall, and F1 scores
13. Demonstrate the ability to train a machine learning model with a given dataset and evaluate its performance
14. Demonstrate the ability to perform inference on an embedded system to solve a problem
15. Provide examples of how anomaly detection with embedded systems can be used to solve problems
16. Demonstrate the ability to detect anomalies from one or more sensors
17. Describe the differences between image classification, object detection, and image segmentation
18. Describe how embedded computer vision can be used to solve problems
19. Describe how convolution and pooling operations are used to filter and downsample images
20.  Describe how convolutional neural networks differ from dense neural networks and how they can be used to solve computer vision problems
21.  Describe how machine learning can be used to classify sounds
22.  Describe the machine learning operations (MLOps) lifecycle

## Required Hardware

Students will need a computer and Internet access to perform machine learning model training and hands-on exercises with the Edge Impulse Studio and Google Colab. Students are encouraged to use the [Arduino Tiny Machine Learning kit](https://store-usa.arduino.cc/products/arduino-tiny-machine-learning-kit) to practice performing inference on an embedded device.

## Syllabus

### Module 1: Introduction to Machine Learning

This module provides an overview of machine learning techniques and terminology with a focus on neural networks and deep learning for classification (supervised learning). It is a broad overview that can be skipped if students have background knowledge of machine learning already.

#### Learning Objectives

1. Describe the differences between artificial intelligence, machine learning, deep learning, and computer vision
2. Provide examples of how machine learning can be used to solve problems (that traditional deterministic programming cannot)
3. Describe the limitations of machine learning
4. Describe the ethical concerns of machine learning
5. Describe the differences between supervised and unsupervised machine learning

#### Lecture Material

##### Section 1

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.1.1 | What is machine learning? | [video](https://www.youtube.com/watch?v=fK8elevliKI&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=3) [slides](***) | [[1]](***)
| 1.1.2 | The machine learning paradigm | [slides](***) | [[3]](***)


## Attribution

### [1] Slides and written material for "[Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
### [2] Slides and written material for "[Computer Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
### [3] Slides and written material for "[TinyML Courseware](https://github.com/tinyMLx/courseware)" by [TinyMLx](https://github.com/tinyMLx) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).