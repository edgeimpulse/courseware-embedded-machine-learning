# Embedded Machine Learning Courseware

[![Markdown link check status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mlc.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mlc.yml) [![Markdown linter status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mdl.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mdl.yml) [![Spellcheck status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/spellcheck.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/spellcheck.yml)

Welcome to the Edge Impulse open courseware for embedded machine learning! This repository houses a collection of slides, reading material, project prompts, and sample questions to get you started creating your own embedded machine learning course. You will also have access to videos that cover much of the material. You are welcome to share these videos with your class either in the classroom or let students watch them on their own time.

Please note that the content in this repository is not intended to be a full semester-long course. Rather, you are encouraged to pull from the modules, rearrange the ordering, make modifications, and use as you see fit to integrate the content into your own curriculum.

For example, many of the lectures and examples from the TinyML Courseware (given by [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40)) go into detail about how TensorFlow Lite works along with advanced topics like quantization. Feel free to skip those sections if you would just like an overview of embedded machine learning and how to use it with Edge Impulse.

Content is divided into separate *modules*. Each module is assumed to be about a week's worth of material, and each section within a module contains about 60 minutes of presentation material. Modules also contain example quiz/test questions, practice problems, and hands-on assignments.

## License

Unless otherwise noted, slides, sample questions, and project prompts are released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are welcome to use and modify them for educational purposes.

The YouTube videos in this repository are shared via the standard YouTube license. You are allowed to show them to your class or provide links for students (and others) to watch.

## Professional Development

Much of the material found in this repository is curated from a collection of online courses with permission from the original creators. You are welcome to take the courses (as professional development) to learn the material in a guided fashion or refer students to the courses for additional learning opportunities.

* [Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning) - Coursera course by Edge Impulse that introduces neural networks and deep learning concepts and applies them to embedded systems. Hands-on projects rely on training and deploying machine learning models with Edge Impulse. Free with optional paid certificate.
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

## Feedback and Contributing

If you find errors or have suggestions about how to make this material better, please let us know! You may [create an issue](https://github.com/edgeimpulse/course-embedded-machine-learning/issues) describing your feedback or [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) if you are familiar with Git.

This repo uses automatic link checking and spell checking. If continuous integration (CI) fails after a push, you may find the dead links or misspelled words, fix them, and push again to re-trigger CI. If dead links or misspelled words are false positives (i.e. purposely malformed link or proper noun), you may update [.mlc_config.json](.mlc-config.json) for links to ignore or [.wordlist.txt](.wordlist.txt) for words to ignore.

## Required Hardware and Software

Students will need a computer and Internet access to perform machine learning model training and hands-on exercises with the Edge Impulse Studio and Google Colab. Students are encouraged to use the [Arduino Tiny Machine Learning kit](https://store-usa.arduino.cc/products/arduino-tiny-machine-learning-kit) to practice performing inference on an embedded device.

A Google account is required for [Google Colab](https://colab.research.google.com/).

An Edge Impulse is required for the [Edge Impulse Studio](https://edgeimpulse.com/).

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
| 1.1.1 | What is machine learning | [video](https://www.youtube.com/watch?v=RDGCGho5oaQ&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=3) [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.1.what-is-machine-learning.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.2 | Machine learning on embedded devices | [video](https://www.youtube.com/watch?v=Thg_EK9xxVk&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=6) [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.2.machine-learning-on-embedded-devices.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.3 | What is tiny machine learning | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.3.what-is-tiny-machine-learning.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.4 | Tinyml case studies | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.4.tinyml-case-studies.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.5 | How do we enable tinyml | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.5.how-do-we-enable-tinyml.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.1.7 | Example assessment questions | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.7.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Limitations and Ethics

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.2.1 | Limitations and ethics | [video](https://www.youtube.com/watch?v=bjR9dwBNTNc&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=4) [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.1.limitations-and-ethics.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.2 | What am I building? | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.2.what-am-i-building.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.3 | Who am I building this for? | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.3.who-am-i-building-this-for.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.4 | What are the consequences? | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.4.what-are-the-consequences.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.5 | The limitations of machine learning | [blog](https://towardsdatascience.com/the-limitations-of-machine-learning-a00e0c3040c6) | |
| 1.2.6 | The future of AI; bias amplification and algorithm determinism | [blog](https://digileaders.com/future-ai-bias-amplification-algorithmic-determinism/) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.2.7 | Example assessment questions | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.7.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Getting Started with Colab

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.3.1 | Getting Started with Google Colab | [video](https://www.youtube.com/watch?v=inN8seMm7UI) | |
| 1.3.2 | Intro to colab | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.2.intro-to-colab.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.3.3 | Welcome to Colab! | [colab](https://colab.research.google.com/notebooks/intro.ipynb) | |
| 1.3.4 | Colab tips | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.4.colab-tips.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 1.3.5 | Why TensorFlow? | [video](https://www.youtube.com/watch?v=yjprpOoH5c8) | |
| 1.3.6 | Sample tensorflow code | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.6.sample-tensorflow-code.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

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
6. Describe why datasets should be broken up into training, validation, and test sets
7. Explain how overfitting occurs and how to identify it
8. Demonstrate the ability to train a dense neural network using Keras and TensorFlow

#### Section 1: Machine Learning Paradigm

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.1.1 | The machine learning paradigm | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.1.the-machine-learning-paradigm.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.2 | Finding patterns | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.2.finding-patterns.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.3 | Thinking about loss | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.3.thinking-about-loss.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.4 | Minimizing loss | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.4.minimizing-loss.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.5 | First neural network | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.5.first-neural-network.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.6 | More neural networks | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.6.more-neural-networks.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.7 | Neural networks in action | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.7.neural-networks-in-action.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.1.8 | Exploring loss | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.8.exploring-loss.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.9 | Minimizing loss | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.9.minimizing-loss.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.10 | Linear regression | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.10.linear-regression.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.11 | Solution linear regression | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.11.solution-linear-regression.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.12 | Example assessment questions | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.12.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Building Blocks of Deep Learning

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.2.1 | Introduction to neural networks | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.1.introduction-to-neural-networks.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.2 | Initialization and learning | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.2.initialization-and-learning.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.3 | Understanding neurons in code | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.3.understanding-neurons-in-code.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.4 | Neural network in code | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.4.neural-network-in-code.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.5 | Introduction to classification | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.5.introduction-to-classification.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.6 | Training validation and test data | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.6.training-validation-and-test-data.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.7 | Realities of coding | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.7.realities-of-coding.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.2.8 | Neurons in action | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.8.neurons-in-action.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.9 | Multi layer neural network | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.9.multi-layer-neural-network.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.10 | Dense neural network | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.10.dense-neural-network.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.11 | Explore neural networks | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.11.explore-neural-networks.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.12 | Solution explore neural networks | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.12.solution-explore-neural-networks.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.13 | Example assessment questions | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.13.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Embedded Machine Learning Challenges

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.3.1 | Challenges for tinyml a | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.1.challenges-for-tinyml-a.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.2 | Challenges for tinyml b | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.2.challenges-for-tinyml-b.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.3 | Challenges for tinyml c | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.3.challenges-for-tinyml-c.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.4 | Challenges for tinyml d | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.4.challenges-for-tinyml-d.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.3.5 | Example assessment questions | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.5.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

### Module 3: Machine Learning Workflow

In this module, students will get an understanding of how data is collected and used to train a machine learning model. They will have the opportunity to collect their own dataset, upload it to Edge Impulse, and train a model using the graphical interface. From there, they will learn how to evaluate a model using a confusion matrix to calculate precision, recall, accuracy, and F1 scores.

#### Learning Objectives

1. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
2. Describe challenges associated with running machine learning algorithms on embedded systems
3. Describe why datasets should be broken up into training, validation, and test sets
4. Explain how overfitting occurs and how to identify it
5. Describe broadly what happens during machine learning model training
6. Describe the difference between model training and inference
7. Describe why test and validation datasets are needed
8. Evaluate a model's performance by calculating accuracy, precision, recall, and F1 scores
9. Demonstrate the ability to train a machine learning model with a given dataset and evaluate its performance

#### Section 1: Machine Learning Workflow

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.1.1 | Tinyml applications | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.1.1.tinyml-applications.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.2 | Role of sensors | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.2.role-of-sensors.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.3 | Machine learning lifecycle | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.1.3.machine-learning-lifecycle.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.4 | Machine learning lifecycle | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.4.machine-learning-lifecycle.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.5 | Machine learning workflow | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.5.machine-learning-workflow.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.1.6 | Example assessment questions | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.6.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Data Collection

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.2.1 | Introduction to data engineering | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.2.1.introduction-to-data-engineering.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.2 | What is data engineering | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.2.what-is-data-engineering.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.3 | Using existing datasets | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.3.using-existing-datasets.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.4 | Responsible data collection | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.4.responsible-data-collection.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.5 | Getting started with edge impulse | [video](https://www.youtube.com/watch?v=lVr4pGeSQKg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=8) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.5.getting-started-with-edge-impulse.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.6 | Data collection with edge impulse | [video](https://www.youtube.com/watch?v=IiJKqHRRuD4&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=9) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.6.data-collection-with-edge-impulse.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.2.7 | Example assessment questions | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.2.7.example-assessment-questions.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Model Training and Evaluation

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.3.1 | Feature extraction from motion data | [video](https://www.youtube.com/watch?v=oDFxBjcvrQU&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=10) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.3.1.feature-extraction-from-motion-data.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.2 | Feature selection in Edge Impulse | [video](https://www.youtube.com/watch?v=xQ3GBkYhXcU&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=11) [tutorial](https://docs.edgeimpulse.com/docs/tutorials/continuous-motion-recognition) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.3 | Machine learning pipeline | [video](https://www.youtube.com/watch?v=Cf1SL-EeQOQ&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=12) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.3.3.machine-learning-pipeline.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.4 | Model training in edge impulse | [video](https://www.youtube.com/watch?v=44v2e6JktbE&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=15) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.3.4.model-training-in-edge-impulse.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.5 | How to evaluate a model | [video](https://www.youtube.com/watch?v=jUiyXCwauJA&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=16) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.3.5.how-to-evaluate-a-model.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.6 | Underfitting and overfitting | [video](https://www.youtube.com/watch?v=6zExT6TucZg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=17) [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.3.6.underfitting-and-overfitting.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.3.7 | Motion detection project | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.3.7.motion-detection-project.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 3.3.8 | Example assessment questions | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.3.8.example-assessment-questions.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

### Module 4: Model Deployment

This module covers why quantization is important for models running on embedded systems and some of the limitations. It also shows how to use a model for inference and set an appropriate threshold to minimize false positives or false negatives, depending on the system requirements. Finally, it covers the steps to deploy a model trained on Edge Impulse to an Arduino board.

#### Learning Objectives

1. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
2. Describe challenges associated with running machine learning algorithms on embedded systems
3. Describe broadly what happens during machine learning model training
4. Describe the difference between model training and inference
5. Demonstrate the ability to perform inference on an embedded system to solve a problem

#### Section 1: Quantization

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.1.1 | Why quantization | [doc](Module%204%20-%20Model%20Deployment/4.1.1.why-quantization.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.2 | Post-training quantization | [slides](Module%204%20-%20Model%20Deployment/4.1.2.post-training-quantization.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.3 | Quantization-aware training | [slides](Module%204%20-%20Model%20Deployment/4.1.3.quantization-aware-training.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.4 | TensorFlow vs TensorFlow Lite | [slides](Module%204%20-%20Model%20Deployment/4.1.4.tensorflow-vs-tensorflow-lite.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.5 | TensorFlow computational graph | [doc](Module%204%20-%20Model%20Deployment/4.1.5.tensorflow-computational-graph.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.1.6 | Post-training quantization | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%204%20-%20Model%20Deployment/4.1.6.post-training-quantization.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.7 | Example assessment questions | [doc](Module%204%20-%20Model%20Deployment/4.1.7.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Embedded Microcontrollers

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.2.1 | Embedded systems | [slides](Module%204%20-%20Model%20Deployment/4.2.1.embedded-systems.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.2 | Diversity of embedded systems | [doc](Module%204%20-%20Model%20Deployment/4.2.2.diversity-of-embedded-systems.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.3 | Embedded computing hardware | [slides](Module%204%20-%20Model%20Deployment/4.2.3.embedded-computing-hardware.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.4 | Embedded microcontrollers | [doc](Module%204%20-%20Model%20Deployment/4.2.4.embedded-microcontrollers.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.5 | TinyML kit peripherals | [doc](Module%204%20-%20Model%20Deployment/4.2.5.tinyml-kit-peripherals.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.6 | TinyML kit peripherals | [slides](Module%204%20-%20Model%20Deployment/4.2.6.tinyml-kit-peripherals.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.7 | Arduino core, frameworks, mbedOS, and bare metal | [doc](Module%204%20-%20Model%20Deployment/4.2.7.arduino-core-frameworks-mbedos-and-bare-metal.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.8 | Embedded ML software | [slides](Module%204%20-%20Model%20Deployment/4.2.8.embedded-ml-software.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.2.8 | Embedded ml software | [slides](Module%204%20-%20Model%20Deployment/4.2.8.embedded-ml-software.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.9 | Example assessment questions | [doc](Module%204%20-%20Model%20Deployment/4.2.9.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Deploying a Model to an Arduino Board

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.3.1 | Using a model for inference | [video](https://www.youtube.com/watch?v=UKeZFIqMk2U&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=18) [slides](Module%204%20-%20Model%20Deployment/4.3.1.using-a-model-for-inference.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 4.3.2 | Testing inference with a smartphone | [video](https://www.youtube.com/watch?v=OWakb-oDAOg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=19) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 4.3.3 | Deploy model to arduino | [video](https://www.youtube.com/watch?v=uUh61R8Hu0o&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=20) [slides](Module%204%20-%20Model%20Deployment/4.3.3.deploy-model-to-arduino.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 4.3.4 | Deploy model to Arduino | [tutorial](https://docs.edgeimpulse.com/docs/tutorials/running-your-impulse-locally/running-your-impulse-arduino) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.3.5 | Example assessment questions | [doc](Module%204%20-%20Model%20Deployment/4.3.5.example-assessment-questions.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

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

### [1] Slides and written material for "[Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### [2] Slides and written material for "[Computer Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### [3] Slides and written material for "[TinyML Courseware](https://github.com/tinyMLx/courseware)" by [TinyMLx](https://github.com/tinyMLx) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
