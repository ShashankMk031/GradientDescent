# Gradient Descent Algorithm

## Introduction

Gradient Descent is one of the most important and widely used algorithms in machine learning. It helps machines learn patterns by minimizing error.It is used to build many systems that greatly impact us like recommendation systems or voice assitants.

---

## What is Gradient Descent?

Gradient Descent is an optimization algorithm used to minimize the error(cost) in machine learning models. In the context of **Linear Regression**, it helps find the best-fitting straight line by updating the slope and intercept to reduce the difference between actual and predicted values.

In simple words, Gradient Descent tries to move downhill on a curve — adjusting parameters step by step in the direction where the cost is reducing fastest — until it reaches the minimum.

---

## Why Use Gradient Descent for Linear Regression?

In Linear Regression,as we know we try to find a line that best fits the data. This is done by minimizing the **Mean Squared Error (MSE)** between predicted and actual values. Gradient Descent provides an efficient way to do this, even for very large datasets.

We aim to find parameters (slope `m` and intercept `b`) that minimize the cost function.

---

## The Math Behind It

### Hypothesis Function (Linear Model):

$$ h_\theta(x) = \theta^Tx = \theta_0 + \theta_1x_1$$

### Cost Function (Mean Squared Error):

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 $$

Where:
- m is the number of data points
- h($\theta$) is the predicted value
- $y^{(i)}$ is the actual value

---

## Gradient Descent Update Rule :

To minimize J($\theta$), we update each parameter of $\theta_j$ using:

$$ \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} $$ 

- $\alpha$ is the **learning rate** it can have value 0< $\alpha$ < 1
- Update happens **simultaneously** for all parameters in each epoch
- Repeats until convergence (i.e., cost stops decreasing)

---

### Learning Rate needs to be balanced just right:

- If $\alpha$ is **too small**: slow convergence and will increase the computation of the model.
- If $\alpha$  is **too large**: may overshoot and never converge.

![Learning Rate](https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png)

---

## Types of Gradient Descent

### 1. **Batch Gradient Descent**
- Uses **entire** dataset per update
- Accurate but slow on large datasets

### 2. **Stochastic Gradient Descent (SGD)**
- Updates parameters using **one example at a time**
- Faster but more noisy

### 3. **Mini-batch Gradient Descent**
- Uses a **small batch** per update
- A good tradeoff between speed and accuracy

In simple worlds:
Say you have 10,000 training samples.

Batch GD: Uses all 10,000 for each update

SGD: Updates weights 10,000 times per epoch, once per sample

Mini-Batch GD (batch size 100): Updates weights 100 times per epoch

---

## Common Challenges

###  Local Minima
In gradient descent, we aim to find the global minimum — the absolute lowest point of the cost function.But some functions (especially in deep neural networks) have multiple dips, or local minima — points where the gradient is zero, but they are not the lowest point overall.
Local minimum: A low point, but not the lowest

Global minimum: The actual lowest point of the function

Saddle point: Flat region with zero gradient but not a minimum

Why It’s a Problem?
If gradient descent gets stuck in a local minimum or saddle point, training halts without reaching the best solution (poor model performance).

#### How to avoid local minima: 

There are many ways to avoid a local minima 
- Using ADAM optimizer.
- Starting with higher learning rate and then rescheduling it to reach the global minima.
- Training model multiple times with diffrent initializations.
---

## Visualizing Gradient Descent

This animation shows how Gradient Descent finds the minimum of a function:

![Visualizing Gradient Descent](https://iq.opengenus.org/content/images/2020/04/1-1.gif)

---

## Real-World Applications

- Linear & Logistic Regression
- Neural Networks (deep learning)
- Recommendation Systems
- Natural Language Processing (word embeddings, etc.)

---
Thank You.
