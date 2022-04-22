# Embedded Machine Learning Courserware

[![GitHub Actions Status Badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/workflows/push/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/workflows/push.yml)

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

## Feedback

If you find errors or have suggestions about how to make this material better, please let us know! You may [create an issue](https://github.com/edgeimpulse/course-embedded-machine-learning/issues) describing your feedback or [create a pull request](https://docs.github.com/es/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) if you are familiar with Git.

## Required Hardware and Software

Students will need a computer and Internet access to perform machine learning model training and hands-on exercises with the Edge Impulse Studio and Google Colab. Students are encouraged to use the [Arduino Tiny Machine Learning kit](https://store-usa.arduino.cc/products/arduino-tiny-machine-learning-kit) to practice performing inference on an embedded device.

A Google account is required for [Google Colab](https://colab.research.google.com/).

An Edge Impulse is required for the [Edge Ipmulse Studio](https://edgeimpulse.com/).

Students will need to install the latest [Arduino IDE](https://www.arduino.cc/en/software).

## Preexisting Datasets and Projects

This is a collection of preexisting datasets, Edge Impulse projects, or curation tools to help you get started with your own edge machine learning projects. With a public Edge Impulse project, note that you can clone the project to your account and/or download the dataset from the Dashboard.

### Motion

* [Edge Impulse continuous gesture (idle, snake, up-down, wave) dataset](https://docs.edgeimpulse.com/docs/pre-built-datasets/continuous-gestures)
* [Alternate motion (idle, up-down, left-right, circle) project](https://studio.edgeimpulse.com/public/76063/latest)

### Sound

* [Running faucet dataset](https://docs.edgeimpulse.com/docs/pre-built-datasets/running-faucet)
* [Google speech commands dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
* [Keyword spotting dataset curation and augmentation script](https://github.com/ShawnHymel/ei-keyword-spotting/blob/master/ei-audio-dataset-curation.ipynb)

### Image Classification

* [Electronic components dataset](https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/Datasets/electronic-components-png.zip?raw=true)
* [Image data augmentation script](https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/2.3.5%20-%20Project%20-%20Data%20Augmentation/solution_image_data_augmentation.ipynb)

### Object Detection

* [Face detection project](https://studio.edgeimpulse.com/public/87291/latest)


## Syllabus

### Module 1: Machine Learning on the Edge

This module provides an overview of machine learning and how it can be used to solve problems. It also introduces the idea of running machine learning algorithms on resource-constrained devices, such as microcontrollers. It covers some of the limitations and ethical concerns of machine learning. Finally, it demonstrates a few examples of Python in Google Colab, which will be used in early modules to showcase how programming is often performed for machine learning with TensorFlow and Keras.

#### Learning Objectives

1. Describe the differences between artificial intelligence, machine learning, and deep learning
2. Provide examples of how machine learning can be used to solve problems (that traditional deterministic programming cannot)
3. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
4. Describe the limitations of machine learning
5. Describe the ethical concerns of machine learning
6. Describe the differences between supervised and unsupervised machine learning

#### Section 1: Machine Learning on the Edge

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.1.1 | What is machine learning? | [video](https://www.youtube.com/watch?v=fK8elevliKI&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=3) [slides](1.1.1-what-is-machine-learning.pptx) | [[1]](https://github.com/edgeimpulse/course-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.2 | Machine learning on embedded devices | [video](https://www.youtube.com/watch?v=Thg_EK9xxVk&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=5) [slides](1.1.2-machine-learning-on-embedded-devices.pptx) | [[1]](https://github.com/edgeimpulse/course-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.3 | What is tiny machine learning? | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.3-what-is-tiny-machine-learning.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.4 | TinyML case studies | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.4-tinyml-case-studies.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.5 | How do we enable TinyML? | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.5-how-do-we-enable-tinyml.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.6 | Why the future of mahcine learning is tiny and bright | [blog](https://petewarden.com/2018/06/11/why-the-future-of-machine-learning-is-tiny/) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.1.7 | Example assessment questions | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.7-example-assessment-questions.docx) | [[1]](https://github.com/edgeimpulse/course-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Limitations and Ethics

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.2.1 | Limitations and Ethics of Machine Learning | [video](https://www.youtube.com/watch?v=bjR9dwBNTNc&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=4) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.1-limitations-and-ethics.pptx) | [[1]](https://github.com/edgeimpulse/course-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.2 | What am I building and what's the goal? | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.2-what-am-i-building.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.3 | Who am I building this for? | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.3-who-am-i-building-this-for.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.4 | What are the consequences? | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.4-what-are-the-consequences.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.5 | The limitations of machine learning | [blog](https://towardsdatascience.com/the-limitations-of-machine-learning-a00e0c3040c6) | |
| 1.2.6 | The future of AI; bias amplification and algorithm determinism | [blog](https://digileaders.com/future-ai-bias-amplification-algorithmic-determinism/) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.2.7 | Example assessment questions | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.7-example-assessment-questions.docx) | [[1]](https://github.com/edgeimpulse/course-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Getting Started with Colab

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.3.1 | Getting Started with Google Colab | [video](https://www.youtube.com/watch?v=inN8seMm7UI) | |
| 1.3.2 | Introduction to Colab | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.2-intro-to-colab.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.3.3 | Welcome to Colab! | [colab](https://colab.research.google.com/notebooks/intro.ipynb) | |
| 1.3.4 | Colab tips | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.4-colab-tips.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.3.5 | Why TensorFlow? | [video](https://www.youtube.com/watch?v=yjprpOoH5c8) | |
| 1.3.6 | Sample TensorFlow code | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.6-sample-tensorflow-code.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.3.7 | 101 exercises for Python fundamentals | [colab](https://colab.research.google.com/github/ryanorsinger/101-exercises/blob/main/101-exercises.ipynb) | |

### Module 2: Getting Started with Deep Learning

This module provides an overview of neural networks and how they can be used to make predictions. Simple examples are given in Python (Google Colab) for students to play with and learn from. If you do not wish to explore basic machine learning with Keras in Google Colab, you may skip this module to move on to using the Edge Impulse graphical environment. Note that some later exercises rely on Google Colab for curating data, visualizing neural networks, etc.

#### Learning Objectives

1. Provide examples of how machine learning can be used to solve problems (that traditional deterministic programming cannot)
2. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
3. Describe challenges associated with running machine learning algorithms on embedded systems
4. Describe broadly how a mathematical model can be used to generalize trends in data
5. Explain how the training process results from minimizing a loss function
6. Describe why datsets should be broken up into training, validation, and test sets
7. Explain how overfitting occurs and how to identify it
8. Demonstrate the ability to train a dense neural network using Keras and TensorFlow

#### Section 1: Machine Learning Paradigm

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.1.1 | The machine learning paradigm | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.1-the-machine-learning-paradigm.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.2 | Finding patterns | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.2-finding-patterns.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.3 | Thinking about loss | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.3-thinking-about-loss.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.4 | Minimizing loss | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.4-minimizing-loss.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.5 | First neural network | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.5-first-neural-network.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.6 | More neural networks | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.6-more-neural-networks.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.7 | Neural networks in action | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.7-neural-networks-in-action.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.1.8 | Exercise: exploring loss | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.8-exploring-loss.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.9 | Exercise: minimizing loss | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.9-minimizing-loss.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.10 | Assignment: linear regression | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.10-linear-regression.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.11 | Solution: linear regression | [doc](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.11-solution-linear-regression.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.12 | Example assessment questions | [doc](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.11-example-assessment-questions.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Building Blocks of Deep Learning

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.2.1 | Introduction to neural networks | [video](https://www.youtube.com/watch?v=c1pgVaEFxjM&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=14) [slides](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.1-introduction-to-neural-networks.pptx) | [[1]](https://github.com/edgeimpulse/course-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.2 | Initialization and learning | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.2-initialization-and-learning.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.3 | Understanding neurons in code | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.3-understanding-neurons-in-code.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.4 | Neural network in code | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.4-neural-network-in-code.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.5 | Introduction to classification | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.5-introduction-to-classification.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.6 | Training, validation, and test data | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.6-training-validation-and-test-data.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.7 | Training, validation, and test data | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.7-realities-of-coding.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.2.8 | Exercise: neurons in action | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.8-neurons-in-action.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.9 | Exercise: multi-layer neural network | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.9-multi-layer-neural-network.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.10 | Exercise: dense neural network | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.10-dense-neural-network.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.11 | Assignment: explore neural networks | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.11-explore-neural-networks.ipynb) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.12 | Solution: explore neural networks | [doc](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.12-solution-explore-neural-networks.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.13 | Example assessment questions | [doc](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.13-example-assessment-questions.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Embedded Machine Learning Challenges

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.3.1 | Challenges for tiny ML part A | [slides](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.1-challenges-for-tinyml-a.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.2 | Challenges for tiny ML part B | [slides](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.2-challenges-for-tinyml-b.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.3 | Challenges for tiny ML part C | [slides](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.3-challenges-for-tinyml-c.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.4 | Challenges for tiny ML part D | [slides](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.3-challenges-for-tinyml-d.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.3.4 | Example assessment questions | [doc](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.4-example-assessment-questions.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

### Module 3: Machine Learning Workflow

In this module, students will get an understanding of how data is collected and used to train a machine learning model. They will have the opportunity to collect their own dataset, upload it to Edge Impulse, and train a model using the graphical interface. From there, they will learn how to evaluate a model using a confusion matrix to calculate precision, recall, accuracy, and F1 scores.

#### Learning Objectives

1. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
2. Describe challenges associated with running machine learning algorithms on embedded systems
3. Describe why datsets should be broken up into training, validation, and test sets
4. Explain how overfitting occurs and how to identify it
5. Describe broadly what happens during machine learning model training
6. Describe the difference between model training and inference
7. Describe why test and validation datasets are needed
8. Evaluate a model's performance by calculating accuracy, precision, recall, and F1 scores
9. Demonstrate the ability to train a machine learning model with a given dataset and evaluate its performance

#### Section 1: Machine Learning Worflow

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.1.1 | Tiny ML applications | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.1.1.tinyml-applications.3.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.2 | The role of sensors in tiny ML | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.1.2.role-of-sensors.3.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.3 | The machine learning lifecycle | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.1.3.machine-learning-lifecycle.3.pptx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.4 | The machine learning lifecycle | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.1.4.machine-learning-lifecycle.3.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.5 | The machine learning lifecycle | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.1.5.machine-learning-workflow.3.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.1.6 | Example assessment questions |  [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.1.6.example-assessment-questions.3.docx) | [[3]](https://github.com/edgeimpulse/course-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Data Collection

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.2.1 | Introduction to data engineering | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.1.introduction-to-data-engineering.3.docx) | [[3]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.2 | What is data engineering | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.2.what-is-data-engineering.3.pptx) | [[3]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.3 | Using existing datasets | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.3.using-existing-datasets.3.pptx) | [[3]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.4 | Responsible data collection | [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.4.responsible-data-collection.3.pptx) | [[3]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.5 | Getting started with edge impulse | [video](https://www.youtube.com/watch?v=lVr4pGeSQKg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=8) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.5.getting-started-with-edge-impulse.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.6 | Data collection with edge impulse | [video](https://www.youtube.com/watch?v=IiJKqHRRuD4&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=9) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.6.data-collection-with-edge-impulse.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.2.7 | Example assessment questions | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.2.7.example-assessment-questions.1.docx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Model Training and Evaluation

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.3.1 | Feature extraction from motion data | [video](https://www.youtube.com/watch?v=oDFxBjcvrQU&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=10) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.1.feature-extraction-from-motion-data.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.2 | Feature selection in Edge Impulse |  [video](https://www.youtube.com/watch?v=xQ3GBkYhXcU&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=11) [tutorial](https://docs.edgeimpulse.com/docs/tutorials/continuous-motion-recognition) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.3 | Machine learning pipeline | [video](https://www.youtube.com/watch?v=Cf1SL-EeQOQ&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=12) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.3.machine-learning-pipeline.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.4 | Model training in edge impulse | [video](https://www.youtube.com/watch?v=44v2e6JktbE&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=15) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.4.model-training-in-edge-impulse.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.5 | How to evaluate a model | [video](https://www.youtube.com/watch?v=jUiyXCwauJA&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=16) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.5.how-to-evaluate-a-model.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.6 | Underfitting and overfitting | [video](https://www.youtube.com/watch?v=6zExT6TucZg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=17) [slides](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.6.underfitting-and-overfitting.1.pptx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.3.7 | Motion detection project | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.7.motion-detection-project.1.docx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.8 | Example assessment questions | [doc](https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/Module%203%20-%20Machine%20Learning%20Workflow/3.3.8.example-assessment-questions.1.docx) | [[1]](https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

### Module 4: Model Deployment

This module covers why quantization is important for models running on embedded systems and some of the limitations. It also shows how to use a model for inference and set an appropriate threshold to minimize false positives or false negatives, depending on the system requirements. Finally, it covers the steps to deploy a model trained on Edge Impulse to an Arduino board.

#### Learning Objectives

1. 

#### Section 1: Quantization

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

#### Section 2: Live Inference

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

#### Section 3: Using the Edge Impulse Arduino Library

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

## TEMPLATE

### Module [x]: [name]

[description]

#### Learning Objectives

#### Section 1: [name]

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

#### Section 2: [name]

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

#### Section 3: [name]

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|

## Attribution

##### [1] Slides and written material for "[Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
##### [2] Slides and written material for "[Computer Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
##### [3] Slides and written material for "[TinyML Courseware](https://github.com/tinyMLx/courseware)" by [TinyMLx](https://github.com/tinyMLx) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).