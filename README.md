<!-- omit in toc -->
# Embedded Machine Learning Courseware

[![Markdown link check status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mlc.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mlc.yml) [![Markdown linter status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mdl.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mdl.yml) [![Spellcheck status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/spellcheck.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/spellcheck.yml) [![HitCount](https://hits.dwyl.com/edgeimpulse/courseware-embedded-machine-learning.svg?style=flat-square&show=unique)](http://hits.dwyl.com/edgeimpulse/courseware-embedded-machine-learning)

Welcome to the Edge Impulse open courseware for embedded machine learning! This repository houses a collection of slides, reading material, project prompts, and sample questions to get you started creating your own embedded machine learning course. You will also have access to videos that cover much of the material. You are welcome to share these videos with your class either in the classroom or let students watch them on their own time.

This repository is part of the Edge Impulse University Program. Please see this page for more information on how to join: [edgeimpulse.com/university](https://edgeimpulse.com/university).

<!-- omit in toc -->
## How to Use This Repository

Please note that the content in this repository is not intended to be a full semester-long course. Rather, you are encouraged to pull from the modules, rearrange the ordering, make modifications, and use as you see fit to integrate the content into your own curriculum.

For example, many of the lectures and examples from the TinyML Courseware (given by [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40)) go into detail about how TensorFlow Lite works along with advanced topics like quantization. Feel free to skip those sections if you would just like an overview of embedded machine learning and how to use it with Edge Impulse.

In general, content from [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) cover theory and hands-on Python coding with Jupyter Notebooks to demonstrate these concepts. Content from [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) and [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) cover hands-on demonstrations and projects using Edge Impulse to deploy machine learning models to embedded systems.

Content is divided into separate *modules*. Each module is assumed to be about a week's worth of material, and each section within a module contains about 60 minutes of presentation material. Modules also contain example quiz/test questions, practice problems, and hands-on assignments.

If you would like to see more content than what is available in this repository, please refer to the [Harvard TinyMLedu site](http://tinyml.seas.harvard.edu/) for additional course material.

<!-- omit in toc -->
## License

Unless otherwise noted, slides, sample questions, and project prompts are released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are welcome to use and modify them for educational purposes.

The YouTube videos in this repository are shared via the standard YouTube license. You are allowed to show them to your class or provide links for students (and others) to watch.

<!-- omit in toc -->
## Professional Development

Much of the material found in this repository is curated from a collection of online courses with permission from the original creators. You are welcome to take the courses (as professional development) to learn the material in a guided fashion or refer students to the courses for additional learning opportunities.

* [Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning) - Coursera course by Edge Impulse that introduces neural networks and deep learning concepts and applies them to embedded systems. Hands-on projects rely on training and deploying machine learning models with Edge Impulse. Free with optional paid certificate.
* [Copmuter Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning) - Follow-on Coursera course that covers image classification and object detection using convolutional neural networks. Hands-on projects rely on training and deploying models with Edge Impulse. Free with optional paid certificate.
* [Tiny Machine Learing (TinyML)](https://www.edx.org/professional-certificate/harvardx-tiny-machine-learning) - EdX course by [Vijay Janapa Reddi](https://scholar.harvard.edu/vijay-janapa-reddi), [Laurence Moroney](https://laurencemoroney.com/), [Pete Warden](https://petewarden.com/), and [Lara Suzuki](https://larissasuzuki.com/). Hands-on projects rely on Python code in Google Colab to train and deploy models to embedded systems with TensorFlow Lite for Microcontrollers. Paid course.

<!-- omit in toc -->
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

<!-- omit in toc -->
## Feedback and Contributing

If you find errors or have suggestions about how to make this material better, please let us know! You may [create an issue](https://github.com/edgeimpulse/course-embedded-machine-learning/issues) describing your feedback or [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) if you are familiar with Git.

This repo uses automatic link checking and spell checking. If continuous integration (CI) fails after a push, you may find the dead links or misspelled words, fix them, and push again to re-trigger CI. If dead links or misspelled words are false positives (i.e. purposely malformed link or proper noun), you may update [.mlc_config.json](.mlc-config.json) for links to ignore or [.wordlist.txt](.wordlist.txt) for words to ignore.

<!-- omit in toc -->
## Required Hardware and Software

Students will need a computer and Internet access to perform machine learning model training and hands-on exercises with the Edge Impulse Studio and Google Colab. Students are encouraged to use the [Arduino Tiny Machine Learning kit](https://store-usa.arduino.cc/products/arduino-tiny-machine-learning-kit) to practice performing inference on an embedded device.

A Google account is required for [Google Colab](https://colab.research.google.com/).

An Edge Impulse is required for the [Edge Impulse Studio](https://edgeimpulse.com/).

Students will need to install the latest [Arduino IDE](https://www.arduino.cc/en/software).

<!-- omit in toc -->
## Preexisting Datasets and Projects

This is a collection of preexisting datasets, Edge Impulse projects, or curation tools to help you get started with your own edge machine learning projects. With a public Edge Impulse project, note that you can clone the project to your account and/or download the dataset from the Dashboard.

<!-- omit in toc -->
### Motion

* [Edge Impulse continuous gesture (idle, snake, up-down, wave) dataset](https://docs.edgeimpulse.com/docs/pre-built-datasets/continuous-gestures)
* [Alternate motion (idle, up-down, left-right, circle) project](https://studio.edgeimpulse.com/public/76063/latest)

<!-- omit in toc -->
### Sound

* [Running faucet dataset](https://docs.edgeimpulse.com/docs/pre-built-datasets/running-faucet)
* [Google speech commands dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
* [Keyword spotting dataset curation and augmentation script](https://github.com/ShawnHymel/ei-keyword-spotting/blob/master/ei-audio-dataset-curation.ipynb)
* [Multilingual spoken words corpus](https://mlcommons.org/en/multilingual-spoken-words/)

<!-- omit in toc -->
### Image Classification

* [Electronic components dataset](https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/Datasets/electronic-components-png.zip?raw=true)
* [Image data augmentation script](https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/2.3.5%20-%20Project%20-%20Data%20Augmentation/solution_image_data_augmentation.ipynb)

<!-- omit in toc -->
### Object Detection

* [Face detection project](https://studio.edgeimpulse.com/public/87291/latest)

<!-- omit in toc -->
## Syllabus

* [Module 1: Machine Learning on the Edge](#module-1-machine-learning-on-the-edge)
  * [Learning Objectives](#learning-objectives)
  * [Section 1: Machine Learning on the Edge](#section-1-machine-learning-on-the-edge)
  * [Section 2: Limitations and Ethics](#section-2-limitations-and-ethics)
  * [Section 3: Getting Started with Colab](#section-3-getting-started-with-colab)
* [Module 2: Getting Started with Deep Learning](#module-2-getting-started-with-deep-learning)
  * [Learning Objectives](#learning-objectives-1)
  * [Section 1: Machine Learning Paradigm](#section-1-machine-learning-paradigm)
  * [Section 2: Building Blocks of Deep Learning](#section-2-building-blocks-of-deep-learning)
  * [Section 3: Embedded Machine Learning Challenges](#section-3-embedded-machine-learning-challenges)
* [Module 3: Machine Learning Workflow](#module-3-machine-learning-workflow)
  * [Learning Objectives](#learning-objectives-2)
  * [Section 1: Machine Learning Workflow](#section-1-machine-learning-workflow)
  * [Section 2: Data Collection](#section-2-data-collection)
  * [Section 3: Model Training and Evaluation](#section-3-model-training-and-evaluation)
* [Module 4: Model Deployment](#module-4-model-deployment)
  * [Learning Objectives](#learning-objectives-3)
  * [Section 1: Quantization](#section-1-quantization)
  * [Section 2: Embedded Microcontrollers](#section-2-embedded-microcontrollers)
  * [Section 3: Deploying a Model to an Arduino Board](#section-3-deploying-a-model-to-an-arduino-board)
* [Module 5: Anomaly Detection](#module-5-anomaly-detection)
  * [Learning Objectives](#learning-objectives-4)
  * [Section 1: Introduction to Anomaly Detection](#section-1-introduction-to-anomaly-detection)
  * [Section 2: K-means Clustering and Autoencoders](#section-2-k-means-clustering-and-autoencoders)
  * [Section 3: Anomaly Detection in Edge Impulse](#section-3-anomaly-detection-in-edge-impulse)
* [Module 6: Image Classification with Deep Learning](#module-6-image-classification-with-deep-learning)
  * [Learning Objectives](#learning-objectives-5)
  * [Section 1: Image Classification](#section-1-image-classification)
  * [Section 2: Convolutional Neural Network (CNN)](#section-2-convolutional-neural-network-cnn)
  * [Section 3: Analyzing CNNs, Data Augmentation, and Transfer Learning](#section-3-analyzing-cnns-data-augmentation-and-transfer-learning)
* [Module 7: Object Detection and Image Segmentation](#module-7-object-detection-and-image-segmentation)
  * [Learning Objectives](#learning-objectives-6)
  * [Section 1: Introduction to Object Detection](#section-1-introduction-to-object-detection)
  * [Section 2: Image Segmentation and Constrained Object Detection](#section-2-image-segmentation-and-constrained-object-detection)
  * [Section 3: Responsible AI](#section-3-responsible-ai)
* [Module 8: Keyword Spotting](#module-8-keyword-spotting)
  * [Learning Objectives](#learning-objectives-7)
  * [Section 1: Audio Classification](#section-1-audio-classification)
  * [Section 2: Spectrograms and MFCCs](#section-2-spectrograms-and-mfccs)
  * [Section 3: Deploying a Keyword Spotting System](#section-3-deploying-a-keyword-spotting-system)

<!-- omit in toc -->
## Course Material

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
| 1.1.3 | What is tiny machine learning | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.3.what-is-tiny-machine-learning.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.4 | Tinyml case studies | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.4.tinyml-case-studies.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.5 | How do we enable tinyml | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.5.how-do-we-enable-tinyml.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.1.7 | Example assessment questions | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.7.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Limitations and Ethics

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.2.1 | Limitations and ethics | [video](https://www.youtube.com/watch?v=bjR9dwBNTNc&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=4) [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.1.limitations-and-ethics.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.2 | What am I building? | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.2.what-am-i-building.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.3 | Who am I building this for? | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.3.who-am-i-building-this-for.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.4 | What are the consequences? | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.4.what-are-the-consequences.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.2.5 | The limitations of machine learning | [blog](https://towardsdatascience.com/the-limitations-of-machine-learning-a00e0c3040c6) | |
| 1.2.6 | The future of AI; bias amplification and algorithm determinism | [blog](https://digileaders.com/future-ai-bias-amplification-algorithmic-determinism/) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.2.7 | Example assessment questions | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.2.7.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Getting Started with Colab

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.3.1 | Getting Started with Google Colab | [video](https://www.youtube.com/watch?v=inN8seMm7UI) | |
| 1.3.2 | Intro to colab | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.2.intro-to-colab.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.3.3 | Welcome to Colab! | [colab](https://colab.research.google.com/notebooks/intro.ipynb) | |
| 1.3.4 | Colab tips | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.4.colab-tips.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 1.3.5 | Why TensorFlow? | [video](https://www.youtube.com/watch?v=yjprpOoH5c8) | |
| 1.3.6 | Sample tensorflow code | [doc](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.3.6.sample-tensorflow-code.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

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
| 2.1.1 | The machine learning paradigm | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.1.the-machine-learning-paradigm.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.2 | Finding patterns | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.2.finding-patterns.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.3 | Thinking about loss | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.3.thinking-about-loss.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.4 | Minimizing loss | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.4.minimizing-loss.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.5 | First neural network | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.5.first-neural-network.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.6 | More neural networks | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.6.more-neural-networks.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.7 | Neural networks in action | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.7.neural-networks-in-action.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.1.8 | Exploring loss | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.8.exploring-loss.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.9 | Minimizing loss | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.9.minimizing-loss.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.10 | Linear regression | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.10.linear-regression.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.11 | Solution linear regression | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.11.solution-linear-regression.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.1.12 | Example assessment questions | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.1.12.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Building Blocks of Deep Learning

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.2.1 | Introduction to neural networks | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.1.introduction-to-neural-networks.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.2 | Initialization and learning | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.2.initialization-and-learning.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.3 | Understanding neurons in code | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.3.understanding-neurons-in-code.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.4 | Neural network in code | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.4.neural-network-in-code.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.5 | Introduction to classification | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.5.introduction-to-classification.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.6 | Training validation and test data | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.6.training-validation-and-test-data.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.7 | Realities of coding | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.7.realities-of-coding.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.2.8 | Neurons in action | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.8.neurons-in-action.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.9 | Multi layer neural network | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.9.multi-layer-neural-network.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.10 | Dense neural network | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.10.dense-neural-network.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.11 | Challenge: explore neural networks | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.11.challenge-explore-neural-networks.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.12 | Solution: explore neural networks | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.12.solution-explore-neural-networks.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.2.13 | Example assessment questions | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.2.13.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Embedded Machine Learning Challenges

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.3.1 | Challenges for tinyml a | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.1.challenges-for-tinyml-a.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.2 | Challenges for tinyml b | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.2.challenges-for-tinyml-b.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.3 | Challenges for tinyml c | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.3.challenges-for-tinyml-c.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 2.3.4 | Challenges for tinyml d | [slides](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.4.challenges-for-tinyml-d.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 2.3.5 | Example assessment questions | [doc](Module%202%20-%20Getting%20Started%20with%20Deep%20Learning/2.3.5.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

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
| 3.1.1 | Tinyml applications | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.1.1.tinyml-applications.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.2 | Role of sensors | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.2.role-of-sensors.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.3 | Machine learning lifecycle | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.1.3.machine-learning-lifecycle.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.4 | Machine learning lifecycle | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.4.machine-learning-lifecycle.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.1.5 | Machine learning workflow | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.5.machine-learning-workflow.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.1.6 | Example assessment questions | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.1.6.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Data Collection

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 3.2.1 | Introduction to data engineering | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.2.1.introduction-to-data-engineering.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.2 | What is data engineering | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.2.what-is-data-engineering.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.3 | Using existing datasets | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.3.using-existing-datasets.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 3.2.4 | Responsible data collection | [slides](Module%203%20-%20Machine%20Learning%20Workflow/3.2.4.responsible-data-collection.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
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
| 3.3.7 | Project: Motion detection | [doc](Module%203%20-%20Machine%20Learning%20Workflow/3.3.7.motion-detection-project.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
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
| 4.1.1 | Why quantization | [doc](Module%204%20-%20Model%20Deployment/4.1.1.why-quantization.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.2 | Post-training quantization | [slides](Module%204%20-%20Model%20Deployment/4.1.2.post-training-quantization.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.3 | Quantization-aware training | [slides](Module%204%20-%20Model%20Deployment/4.1.3.quantization-aware-training.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.4 | TensorFlow vs TensorFlow Lite | [slides](Module%204%20-%20Model%20Deployment/4.1.4.tensorflow-vs-tensorflow-lite.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.5 | TensorFlow computational graph | [doc](Module%204%20-%20Model%20Deployment/4.1.5.tensorflow-computational-graph.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.1.6 | Post-training quantization | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%204%20-%20Model%20Deployment/4.1.6.post-training-quantization.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.1.7 | Example assessment questions | [doc](Module%204%20-%20Model%20Deployment/4.1.7.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Embedded Microcontrollers

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.2.1 | Embedded systems | [slides](Module%204%20-%20Model%20Deployment/4.2.1.embedded-systems.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.2 | Diversity of embedded systems | [doc](Module%204%20-%20Model%20Deployment/4.2.2.diversity-of-embedded-systems.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.3 | Embedded computing hardware | [slides](Module%204%20-%20Model%20Deployment/4.2.3.embedded-computing-hardware.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.4 | Embedded microcontrollers | [doc](Module%204%20-%20Model%20Deployment/4.2.4.embedded-microcontrollers.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.5 | TinyML kit peripherals | [doc](Module%204%20-%20Model%20Deployment/4.2.5.tinyml-kit-peripherals.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.6 | TinyML kit peripherals | [slides](Module%204%20-%20Model%20Deployment/4.2.6.tinyml-kit-peripherals.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.7 | Arduino core, frameworks, mbedOS, and bare metal | [doc](Module%204%20-%20Model%20Deployment/4.2.7.arduino-core-frameworks-mbedos-and-bare-metal.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.8 | Embedded ML software | [slides](Module%204%20-%20Model%20Deployment/4.2.8.embedded-ml-software.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.2.8 | Embedded ml software | [slides](Module%204%20-%20Model%20Deployment/4.2.8.embedded-ml-software.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 4.2.9 | Example assessment questions | [doc](Module%204%20-%20Model%20Deployment/4.2.9.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Deploying a Model to an Arduino Board

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.3.1 | Using a model for inference | [video](https://www.youtube.com/watch?v=UKeZFIqMk2U&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=18) [slides](Module%204%20-%20Model%20Deployment/4.3.1.using-a-model-for-inference.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 4.3.2 | Testing inference with a smartphone | [video](https://www.youtube.com/watch?v=OWakb-oDAOg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=19) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 4.3.3 | Deploy model to arduino | [video](https://www.youtube.com/watch?v=uUh61R8Hu0o&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=20) [slides](Module%204%20-%20Model%20Deployment/4.3.3.deploy-model-to-arduino.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 4.3.4 | Deploy model to Arduino | [tutorial](https://docs.edgeimpulse.com/docs/deployment/running-your-impulse-arduino) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 4.3.5 | Example assessment questions | [doc](Module%204%20-%20Model%20Deployment/4.3.5.example-assessment-questions.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

### Module 5: Anomaly Detection

This module describes several approaches to anomaly detection and why we might want to use it in embedded machine learning.

#### Learning Objectives

1. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
2. Describe challenges associated with running machine learning algorithms on embedded systems
3. Describe broadly what happens during machine learning model training
4. Describe the difference between model training and inference
5. Demonstrate the ability to perform inference on an embedded system to solve a problem
6. Describe how anomaly detection can be used to solve problems

#### Section 1: Introduction to Anomaly Detection

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 5.1.1 | Introduction to anomaly detection | [doc](Module%205%20-%20Anomaly%20Detection/5.1.1.introduction-to-anomaly-detection.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.1.2 | What is anomaly detection? | [slides](Module%205%20-%20Anomaly%20Detection/5.1.2.what-is-anomaly-detection.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.1.3 | Challenges with anomaly detection | [slides](Module%205%20-%20Anomaly%20Detection/5.1.3.challenges-with-anomaly-detection.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.1.4 | Industry and TinyML | [doc](Module%205%20-%20Anomaly%20Detection/5.1.4.industry-and-tinyml.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.1.5 | Anomaly detection datasets | [slides](Module%205%20-%20Anomaly%20Detection/5.1.5.anomaly-detection-datasets.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.1.6 | MIMII dataset paper | [doc](Module%205%20-%20Anomaly%20Detection/5.1.6.mimii-dataset-paper.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.1.7 | Real and synthetic data | [doc](Module%205%20-%20Anomaly%20Detection/5.1.7.real-and-synthetic-data.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 5.1.8 | Example assessment questions | [doc](Module%205%20-%20Anomaly%20Detection/5.1.8.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: K-means Clustering and Autoencoders

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 5.2.1 | K-means clustering | [slides](Module%205%20-%20Anomaly%20Detection/5.2.1.k-means-clustering.0.pptx?raw=true) |  |
| 5.2.2 | Autoencoders | [slides](Module%205%20-%20Anomaly%20Detection/5.2.2.autoencoders.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.2.3 | Autoencoder model architecture | [doc](Module%205%20-%20Anomaly%20Detection/5.2.3.autoencoder-model-architecture.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 5.2.4 | K-means clustering for anomaly detection | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%205%20-%20Anomaly%20Detection/5.2.4.k-means-clustering-for-anomaly-detection.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.2.5 | Autoencoders for anomaly detection | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%205%20-%20Anomaly%20Detection/5.2.5.autoencoders-for-anomaly-detection.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.2.6 | Challenge autoencoders | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%205%20-%20Anomaly%20Detection/5.2.6.challenge-autoencoders.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.2.7 | Solution autoencoders | [doc](Module%205%20-%20Anomaly%20Detection/5.2.7.solution-autoencoders.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 5.2.8 | Example assessment questions | [doc](Module%205%20-%20Anomaly%20Detection/5.2.8.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Anomaly Detection in Edge Impulse

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 5.3.1 | Anomaly detection in edge impulse | [video](https://www.youtube.com/watch?v=7Vz3S17nPWg&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=21) [slides](Module%205%20-%20Anomaly%20Detection/5.3.1.anomaly-detection-in-edge-impulse.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 5.3.2 | Industrial embedded machine learning demo | [video](https://www.youtube.com/watch?v=bo2mFzeft-o&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=22) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 5.3.3 | Project: Motion classification and anomaly detection | [doc](Module%205%20-%20Anomaly%20Detection/5.3.3.project-motion-classification-and-anomaly-detection.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

### Module 6: Image Classification with Deep Learning

This module introduces the concept of image classification, why it is important in machine learning, and how it can be used to solve problems. Convolution and pooling operations are covered, which form the building blocks for convolutional neural networks (CNNs). Saliency maps and Grad-CAM are offered as two techniques for visualizing the inner gs of CNNs. Data augmentation is introduced as a method for generating new data from existing data to train a more robust model. Finally, transfer learning is shown as a way of reusing pretrained models.

#### Learning Objectives

1. Describe the differences between image classification, object detection, and image segmentation
2. Describe how embedded computer vision can be used to solve problems
3. Describe how convolution and pooling operations are used to filter and downsample images
4. Describe how convolutional neural networks differ from dense neural networks and how they can be used to solve computer vision problems

#### Section 1: Image Classification

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 6.1.1 | What is computer vision? | [video](https://www.youtube.com/watch?v=fK8elevliKI&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=3) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.1.1.what-is-computer-vision.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.1.2 | Overview of digital images | [video](https://www.youtube.com/watch?v=BdLJ9Lk1I1M&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=4) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.1.2.overview-of-digital-images.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.1.3 | Dataset collection | [video](https://www.youtube.com/watch?v=uH9-Nhe8XGw&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=5) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.1.3.dataset-collection.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.1.4 | Overview of image classification | [video](https://www.youtube.com/watch?v=c20ditpjGjo&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=6) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.1.4.what-is-image-classification.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.1.5 | Training an image classifier with Keras | [video](https://www.youtube.com/watch?v=ygzvKFgUUTA&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=8) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 6.1.6 | Example assessment questions | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.1.6.example-assessment-questions.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Convolutional Neural Network (CNN)

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 6.2.1 | Image convolution | [video](https://www.youtube.com/watch?v=glSdYcpP_v8&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=14) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.1.image-convolution.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.2 | Pooling layer | [video](https://www.youtube.com/watch?v=E9TNa_6Askc&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=15) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.2.pooling-layer.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.3 | Convolutional neural network | [video](https://www.youtube.com/watch?v=30ikzV8Fi-0&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=16) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.3.convolutional-neural-network.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.4 | CNN in keras | [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.4.cnn-in-keras.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.5 | Mapping features to labels | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.5.mapping-features-to-labels.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.6 | Training a CNN in Edge Impulse| [video](https://www.youtube.com/watch?v=fihp_CqlcZU&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=17) [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.6.training-a-cnn-in-edge-impulse.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 6.2.7 | Exploring convolutions | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.7.exploring-convolutions.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.8 | Convolutional neural networks | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.8.convolutional-neural-networks.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.9 | Challenge: CNN | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.9.challenge-cnn.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.10 | Solution: CNN | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.10.solution-cnn.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.2.11 | Example assessment questions | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.2.11.example-assessment-questions.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Analyzing CNNs, Data Augmentation, and Transfer Learning

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 6.3.1 | CNN visualizations | [video](https://www.youtube.com/watch?v=TmOgYgY0fTc&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=18) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.1.cnn-visualizations.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.2 | Data augmentation | [video](https://www.youtube.com/watch?v=AB4dLvW5mus&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=19) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.2.data-augmentation.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.3 | TensorFlow datasets | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.3.tensorflow-datasets.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.4 | Avoiding overfitting with data augmentation | [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.4.avoiding-overfitting-with-data-augmentation.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.5 | Dropout regularization | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.5.dropout-regularization.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.6 | Exploring loss functions and optimizers | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.6.exploring-loss-functions-and-optimizers.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.7 | Transfer learning and MobileNet | [video](https://www.youtube.com/watch?v=93eczumOpx8&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=20) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.7.transfer-learning-and-mobilenet.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.8 | Transfer learning with Edge Impulse | [video](https://www.youtube.com/watch?v=93eczumOpx8&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=20) [slides](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.7.transfer-learning-and-mobilenet.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 6.3.9 | Saliency and Grad-CAM | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.9.saliency-and-grad-cam.2.ipynb) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.10 | Image transforms demo | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.10.image-transforms-demo.2.ipynb) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.11 | Challenge: image data augmentation | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.11.challenge-image-data-augmentation.2.ipynb) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.12 | Solution: image data augmentation | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.12.solution-image-data-augmentation.2.ipynb) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 6.3.13 | Example assessment questions | [doc](Module%206%20-%20Image%20Classification%20with%20Deep%20Learning/6.3.13.example-assessment-questions.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

### Module 7: Object Detection and Image Segmentation

In this module, we look at object detection, how it differs from image classification, and the unique set of problems it solves. We also briefly examine image segmentation and discuss constrained object detection. Finally, we look at responsible AI as it relates to computer vision and AI at large.

#### Learning Objectives

1. Describe the differences between image classification, object detection, and image segmentation
2. Describe how embedded computer vision can be used to solve problems
3. Describe how image segmentation can be used to solve problems
4. Describe how convolution and pooling operations are used to filter and downsample images
5. Describe how convolutional neural networks differ from dense neural networks and how they can be used to solve computer vision problems
6. Describe the limitations of machine learning
7. Describe the ethical concerns of machine learning
8. Describe the requirements for collecting a good dataset and what factors can create a biased dataset

#### Section 1: Introduction to Object Detection

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 7.1.1 | Introduction to object detection | [video](https://www.youtube.com/watch?v=hV-YVzXEVYg&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=23) [slides](Module%207%20-%20Object%20Detection/7.1.1.introduction-to-object-detection.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.1.2 | Object detection performance metrics | [video](https://www.youtube.com/watch?v=4gE4TkGHm2E&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=24) [slides](Module%207%20-%20Object%20Detection/7.1.2.object-detection-performance-metrics.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.1.3 | Object detection models | [video](https://www.youtube.com/watch?v=Gqjx0b-QkYM&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=25) [slides](Module%207%20-%20Object%20Detection/7.1.3.object-detection-models.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.1.4 | Training an object detection model | [video](https://www.youtube.com/watch?v=HHebB8uiyPc&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=26) [slides](Module%207%20-%20Object%20Detection/7.1.4.training-an-object-detection-model.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.1.5 | Digging deeper into object detection | [doc](Module%207%20-%20Object%20Detection/7.1.5.digging-deeper-into-object-detection.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 7.1.6 | Example assessment questions | [doc](Module%207%20-%20Object%20Detection/7.1.6.example-assessment-questions.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Image Segmentation and Constrained Object Detection

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 7.2.1 | Image segmentation | [video](https://www.youtube.com/watch?v=LxAxH4f7EHo&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=28) [slides](Module%207%20-%20Object%20Detection/7.2.1.image-segmentation.2.pptx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.2.2 | Multi-stage Inference Demo | [video](https://www.youtube.com/watch?v=i8eE0p49qQI&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=29) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.2.3 | Reusing Representations with Mat Kelcey | [video](https://www.youtube.com/watch?v=sjXUgK9YaSc&list=PL7VEa1KauMQqQw7duQEB6GSJwDS06dtKU&index=30) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.2.4 | tinyML Talks: Constrained Object Detection on Microcontrollers with FOMO | [video](https://www.youtube.com/watch?v=VzJZM5p24Tc) | |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 7.2.5 | Project: Deploy an object detection model | [doc](Module%207%20-%20Object%20Detection/7.2.5.project-deploy-an-object-detection-model.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 7.2.6 | Example assessment questions | [doc](Module%207%20-%20Object%20Detection/7.2.6.example-assessment-questions.2.docx?raw=true) | [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Responsible AI

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 7.3.1 | Dataset collection | [slides](Module%207%20-%20Object%20Detection/7.3.1.dataset-collection.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 7.3.2 | The many faces of bias | [doc](Module%207%20-%20Object%20Detection/7.3.2.the-many-faces-of-bias.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 7.3.3 | Biased datasets | [slides](Module%207%20-%20Object%20Detection/7.3.3.biased-datasets.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 7.3.4 | Model fairness | [slides](Module%207%20-%20Object%20Detection/7.3.4.model-fairness.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 7.3.5 | Google what if tool | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%207%20-%20Object%20Detection/7.3.5.google-what-if-tool.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 7.3.6 | Example assessment questions | [doc](Module%207%20-%20Object%20Detection/7.3.6.example-assessment-questions.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

### Module 8: Keyword Spotting

In this module, we create a functioning keyword spotting (also known as "wake word detection") system. To do so, we must introduce several concepts unique to audio digital signal processing and combine it with image classification techniques.

#### Learning Objectives

1. Describe how machine learning can be used to classify sounds
2. Describe how sound classification can be used to solve problems
3. Describe the major components in a keyword spotting system
4. Demonstrate the ability to train and deploy a sound classification system

#### Section 1: Audio Classification

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 8.1.1 | Introduction to audio classification | [slides](Module%208%20-%20Keyword%20Spotting/8.1.1.introduction-to-audio-classification.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.1.2 | Audio data capture | [slides](Module%208%20-%20Keyword%20Spotting/8.1.2.audio-data-capture.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.1.3 | What is keyword spotting | [slides](Module%208%20-%20Keyword%20Spotting/8.1.3.what-is-keyword-spotting.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.1.4 | Keyword spotting challenges | [slides](Module%208%20-%20Keyword%20Spotting/8.1.4.keyword-spotting-challenges.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.1.5 | Keyword spotting application architecture | [doc](Module%208%20-%20Keyword%20Spotting/8.1.5.keyword-spotting-application-architecture.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.1.6 | Keyword spotting datasets | [slides](Module%208%20-%20Keyword%20Spotting/8.1.6.keyword-spotting-datasets.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.1.7 | Keyword spotting dataset creation | [doc](Module%208%20-%20Keyword%20Spotting/8.1.7.keyword-spotting-dataset-creation.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 8.1.8 | Example assessment questions | [doc](Module%208%20-%20Keyword%20Spotting/8.1.8.example-assessment-questions.3.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 2: Spectrograms and MFCCs

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 8.2.1 | Keyword spotting data collection | [slides](Module%208%20-%20Keyword%20Spotting/8.2.1.keyword-spotting-data-collection.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.2.2 | Spectrograms and mfccs | [doc](Module%208%20-%20Keyword%20Spotting/8.2.2.spectrograms-and-mfccs.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.2.3 | Keyword spotting model | [slides](Module%208%20-%20Keyword%20Spotting/8.2.3.keyword-spotting-model.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.2.4 | Audio feature extraction | [slides](Module%208%20-%20Keyword%20Spotting/8.2.4.audio-feature-extraction.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.2.5 | Review of convolutional neural networks | [slides](Module%208%20-%20Keyword%20Spotting/8.2.5.review-of-convolutional-neural-networks.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.2.6 | Modifying the neural network | [slides](Module%208%20-%20Keyword%20Spotting/8.2.6.modifying-the-neural-network.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 8.2.7 | Spectrograms and mfccs | [colab](https://colab.research.google.com/github/edgeimpulse/courseware-embedded-machine-learning/blob/main/Module%208%20-%20Keyword%20Spotting/8.2.7.spectrograms-and-mfccs.3.ipynb) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.2.8 | Example assessment questions | [doc](Module%208%20-%20Keyword%20Spotting/8.2.8.example-assessment-questions.3.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

#### Section 3: Deploying a Keyword Spotting System

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 8.3.1 | Deploy audio classifier | [slides](Module%208%20-%20Keyword%20Spotting/8.3.1.deploy-audio-classifier.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.3.2 | Implementation strategies | [slides](Module%208%20-%20Keyword%20Spotting/8.3.2.implementation-strategies.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.3.3 | Metrics for keyword spotting | [slides](Module%208%20-%20Keyword%20Spotting/8.3.3.metrics-for-keyword-spotting.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.3.4 | Streaming audio | [slides](Module%208%20-%20Keyword%20Spotting/8.3.4.streaming-audio.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.3.5 | Cascade architectures | [slides](Module%208%20-%20Keyword%20Spotting/8.3.5.cascade-architectures.3.pptx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |
| 8.3.6 | Keyword spotting in the big picture | [doc](Module%208%20-%20Keyword%20Spotting/8.3.6.keyword-spotting-in-the-big-picture.3.docx?raw=true) | [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

##### Exercises and Problems

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 8.3.7 | Project: Sound classification | [doc](Module%208%20-%20Keyword%20Spotting/8.3.7.project-sound-classification.1.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 8.3.8 | Example assessment questions | [doc](Module%208%20-%20Keyword%20Spotting/8.3.8.example-assessment-questions.3.docx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) |

<!-- omit in toc -->
## Attribution

<!-- omit in toc -->
### [1] Slides and written material for "[Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<!-- omit in toc -->
### [2] Slides and written material for "[Computer Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)" by [Edge Impulse](https://edgeimpulse.com/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<!-- omit in toc -->
### [3] Slides and written material for "[TinyML Courseware](https://github.com/tinyMLx/courseware)" by Prof. Vijay Janapa Reddi of [Harvard University](http://tinyml.seas.harvard.edu/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
