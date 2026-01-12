# Self-Study Preparation Guide

**⏳ Estimated Prep Time:** 45–60 minutes

Welcome to our flipped-classroom session, where you'll review foundational concepts beforehand to maximize our time for hands-on coding and debugging. This pre-study focuses on demystifying the architecture of Neural Networks and the syntax of PyTorch. By familiarizing yourself with these core concepts now, you will be better positioned to build, tune, and troubleshoot deep learning models for real-world applications—from computer vision to natural language processing.

## ⚡ Your Self-Study Tasks

Please complete the following activities before our session using the provided **`neural_network_deep_learning_lesson.ipynb`** notebook.

### 📝 Task 1: The "What" and "Why" of Neural Networks

**Activity:** Read the **"Introduction to Neural Networks,"** **"Deep Learning,"** and **"Perceptron"** sections. Focus on the transition from simple linear models to complex deep learning architectures.

**Guiding Questions:**

* **Evolution:** How does a "Deep Learning" model differ structurally from a simple Perceptron?
* **Application:** The text lists several applications (e.g., Computer Vision, NLP). Which of these applications is most relevant to your current industry or personal interests, and why?
* **The Math:** Look at the Perceptron formula . In your own words, what role does the "weight" () play in making a decision?

### 📝 Task 2: The Mechanics of Learning

**Activity:** Review the **"Multi-Layer Perceptron (MLP)"** section, paying specific attention to **Activation Functions** and **Backpropagation**. **You do not need to memorize the calculus equations.** Focus on the logical flow of how data moves forward and how errors move backward.

**Focus your attention on these key components:**

1. **Non-linearity:** Why we need functions like ReLU or Sigmoid.
2. **Loss Function:** How the network measures its own mistake.
3. **Optimizer:** How the network adjusts its weights (Gradient Descent).

**Guiding Questions:**

* If you removed the non-linear activation function (like ReLU) from a Deep Neural Network, what would happen to its ability to learn complex patterns?
* In the context of the "Backpropagation" diagram, what information is being sent back through the network to update the weights?

### 📝 Task 3: Visualizing PyTorch Structure

**Activity:** Skim the **"Introduction to PyTorch"** and **"Building a MLP... with PyTorch"** sections. Don't worry about running the code yet. Instead, look at the Python class structure used to define the network (specifically the `Net` class).

**Guiding Questions:**

* In the `Net` class, what is the difference between the `__init__` method and the `forward` method?
* The tutorial mentions "Dynamic Computation Graph." Based on the text, what advantage does this offer during the development process?

## 🙌🏻 Active Engagement Strategies

To deepen your retention, try one of the following while you review:

* **Analogy Generation:** Try to explain *Gradient Descent* to a non-technical colleague using a real-world analogy (e.g., walking down a misty mountain).
* **Diagramming:** Sketch the flow of data through a single neuron: Inputs  Weights  Summation  Activation  Output.
* **Code Mapping:** Look at the PyTorch code block defining `class Net(nn.Module)`. Annotate (mentally or on paper) which lines correspond to the "Hidden Layers" and which correspond to the "Output Layer."

## 📖 Additional Reading Material

If you wish to explore further (optional), these resources provide excellent visual context:

* [But what is a neural network? (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk) - A highly recommended visual introduction.
* [PyTorch "Get Started" Documentation](https://pytorch.org/get-started/locally/) - Official setup and basics guide.

### 🙋🏻‍♂️ See you in the session!