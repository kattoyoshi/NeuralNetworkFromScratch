## 5. Very simplified explanation about the computation
In this section, I will summarize the computation of the neural network.
For simplification, the network structure for the description is much simpler than the network described in README.

### 5.1 Example Structure
Two layer perceptron: [INPUT] - [FC(HIDDEN1)] - [FC(OUTPUT)]  
Input: $X=(x_1, x_2)$  
Output: $\hat{y} = (\hat{y_1}, \hat{y_2})$  

The detailed structure is shown in the figure below.  

![](images/fig2.png)

Definition:
- $z_j^{(l)}$: input to the neuron $j$ of the layer $l$
- $\phi^{(l)}$: activation function of the layer $l$
- $a_j^{(l)}$: output of the nuuron $j$ of the layer $l$  
- $\bf{W}$: weight matrix
- $\bf{B}$: bias vector

### 5.2 Forward pass
```python
# Input -> hidden
z_1 = np.dot(X, W_1) + b_1
a_1 = phi_1(z_1)

# hidden -> Output
z_2 = np.dot(X, W_1) + b_2
y_hat = phi_2(z_2)  # y_hat = a_2 = phi_2(z_2)
```

### 5.3 Parameter updates (SGD)
- Update weights and bias to minimize the prediction error.
- Prediction error is calculated using loss function (error function).

In case of using SGD, the parameter update rules are following, where $\eta$ is the learning rate and $L$ is the loss function.

#### - Weight
$$\bf{W} \to \bf{W}+\Delta{\bf{W}}$$
$$\Delta{\bf{W}} = -\eta\frac{\partial L}{\partial \bf{W}}$$

#### - Bias
$$\bf{B} \to \bf{B}+\Delta{\bf{B}}$$
$$\Delta{\bf{B}} = -\eta\frac{\partial L}{\partial \bf{B}}$$

### 5.4 Backpropagation
#### 5.4.1 Objective
To calculate the partial derivative of the loss function using chain rule.

#### 5.4.2 Details
![](images/fig3.png)

##### 1) Output Layer (layer 2)
__\- Error term of the output Layer__
- Subscript "j" indicates the neuron number of the layer (2).
$$
\begin{align*}
  \delta_j^{(2)}
  &= \frac{\partial L}{\partial z_j^{(2)}}
  = \frac{\partial L}{\partial \hat{y_j}} \frac{\partial \hat{y_j}}{\partial z_j^{(2)}}
  = \frac{\partial L}{\partial \hat{y_j}} \frac{\partial a_j^{(2)}}{\partial z_j^{(2)}} \\
  &= \frac{\partial L}{\partial \hat{y_j}} \frac{\partial \phi^{(2)}(z_j^{(2)})}{\partial z_j^{(2)}}
  = \frac{\partial L}{\partial \hat{y_j}} \phi'^{(2)}(z_j^{(2)})
\end{align*}
$$

__\- Partial derivative of the weights/bias in the output Layer__
- Weights
  - Subscript "j" indicates the neuron number of the layer (2).
  - Subscript "k" indicates the neuron number of the layer (1).
$$
\begin{align*}
  \frac{\partial L}{\partial \omega^{(2)}_{k,j}}
  &= \frac{\partial L}{\partial \hat{y_j}} \frac{\partial \hat{y_j}}{\partial \omega^{(2)}_{k,j}}
  = \frac{\partial L}{\partial \hat{y_j}} \frac{\partial a_j^{(2)}}{\partial \omega^{(2)}_{k,j}}
  = \frac{\partial L}{\partial \hat{y_j}} \frac{\partial  \phi^{(2)}(z_j^{(2)})}{\partial \omega^{(2)}_{k,j}} \\
  &= \frac{\partial L}{\partial \hat{y_j}} \frac{\partial  \phi^{(2)}(z_j^{(2)})}{\partial z_j^{(2)}} \frac{\partial z_j^{(2)}}{\partial \omega^{(2)}_{k,j}}
  = \frac{\partial L}{\partial \hat{y_j}} \phi'^{(2)}(z_j^{(2)}) \cdot a^{(1)}_k \\
  &= \delta_j^{(2)} a^{(1)}_k
\end{align*}
$$

- Bias
  - Subscript "j" indicates the neuron number of the layer (2).
$$
\begin{align*}
  \frac{\partial L}{\partial b^{(2)}_j}
  &= \frac{\partial L}{\partial \hat{y_j}} \frac{\partial \hat{y_j}}{\partial b^{(2)}_j}
  =  \frac{\partial L}{\partial \hat{y_j}} \frac{\partial a_j^{(2)}}{\partial b^{(2)}_j} \\
  &=  \frac{\partial L}{\partial \hat{y_j}} \frac{\partial \phi^{(2)}(z_j^{(2)})}{\partial b^{(2)}_j}
  =  \frac{\partial L}{\partial \hat{y_j}} \frac{\partial \phi^{(2)}(z_j^{(2)})}{\partial z_j^{(2)}} \frac{\partial z_j^{(2)}}{\partial b^{(2)}_j} \\
  &= \frac{\partial L}{\partial \hat{y_j}} \phi'^{(2)}(z_j^{(2)}) \cdot 1
  = \delta_j^{(2)}
\end{align*}  
$$

where
$$z_j^{(2)}=\sum_k \omega_{k, j}^{(2)}a^{(1)}_k+b_j^{(2)}=\sum_k \omega_{k, j}^{(2)}\phi^{(1)}(z_k^{(1)})+b_j^{(2)}$$

##### 2) Hidden Layer (layer 1)

__\- Error term of the hidden Layer__
- Subscript "j" indicates the neuron number of the layer (1).
- Subscript "k" indicates the neuron number of the layer (2).
$$
\begin{align*}
  \delta_j^{(1)}
  &= \frac{\partial L}{\partial z_j^{(1)}}
  = \sum_k \frac{\partial L}{\partial z_k ^ {(2)}} \frac{\partial z_k ^ {(2)}}{\partial z_j ^ {(1)}}
  = \sum_k \frac{\partial z_k ^ {(2)}}{\partial z_j ^ {(1)}} \delta_k^{(2)}
  = \sum_k \omega_{j,k}^{(2)}\phi'(z_j^{(1)}) \delta_k^{(2)}
\end{align*}
$$

where

$$
z_k^{(2)}=\sum_j \omega_{j,k}^{(2)}a_j^{(1)}+b_k^{(2)} = \sum_j \omega_{j,k}^{(2)} \phi^{(1)}(z_j^{(1)})+b_k^{(2)}
$$

$$
\frac{\partial z_k^{(2)}}{\partial z_j^{(1)}}
=\frac{\partial}{\partial z_j^{(1)}}
\left( \sum_j \omega_{j,k}^{(2)} \phi^{(1)}(z_j^{(1)})+b_k^{(2)} \right)
=\omega_{j,k}^{(2)}\phi'(z_j^{(1)})
$$

__\- Partial derivative of the weights/bias in the hidden Layer__
- Weight
  - Subscript "j" indicates the neuron number of the layer (1).
  - Subscript "k" indicates the neuron number of the layer (0) = (Input features).
$$\frac{\partial L}{\partial \omega_{k,j}^{(1)}}=\delta_j^{(1)} a_k^{(0)}=\delta_j^{(1)} x_k$$

- Bias
  - Subscript "j" indicates the neuron number of the layer (1).
$$\frac{\partial L}{\partial b_j^{(1)}}=\delta_j^{(1)}$$

## 6. Generalized explanation about the Backpropagation
### 6.1 Error term of the output layer
#### - Scalar form
- Subscript "j" indicates the neuron index.
- Subscript "o" indicates the output layer.
$$\delta_j^{o}=\frac{\partial L}{\partial a_j^{o}} \phi'^{o}(z_j^{o})$$

#### - Matrix form
- Subscript "o" indicates the output layer.
$$\delta^{o}=\nabla_a L \odot \phi'^{o}(z^{o})$$

### 6.2 Error term of the "l-th" layer
#### - Scalar form
- Subscripts "l", "l+1" indicate the l-th layer and (l+1)-th layer respectively.
- Subscript "j" indicates the neuron number of the l-th layer.
- Subscript "k" indicates the neuron number of the (l+1)-th layer.
$$
\begin{align*}
  \delta_j^{l}
  = \sum_k \omega_{j,k}^{l+1}\phi'(z_j^{l}) \delta_k^{l+1}
\end{align*}
$$

#### - Matrix form
$$
\begin{align*}
  \delta^{(l)}
  = \left( \delta^{l+1} (\omega^{l+1})^T  \right) \odot \phi'(z^{l})
\end{align*}
$$
### 6.3 Derivative of the bias
#### Scalar form
$$\frac{\partial L}{\partial b_j^l} = \delta_j^l$$
#### Matrix form
$$\frac{\partial L}{\partial {\bf B}^l} = \bf{\delta^l}$$

### 6.4 Derivative of the weight
#### Scalar form
$$\frac{\partial L}{\partial \omega_{jk}^l} = a_k^{l-1} \delta_j^l$$
#### Matrix form
$$\frac{\partial L}{\partial {\bf W}^l} = ({\bf A}^{l-1})^T \delta^l$$

## 7. Overall description using computation graph
![](images/fig4.png)

[Notifications]:

\*1: In the bias term, back propagated gradient matrix has N rows.
This "N" corresponds to the data number of the batch.
Therefore, the matrix shape of back propagated gradient and bias term does not match the same.
So, to update the bias term, we have to calculate the average gradient of the N data.

\*2: In the weight term, the matrix shape of the back propagated gradient coincides to the shape of the weight matrix.
However, each element of the back propagated gradient has already summed up by N data.
So, to update the weight term, we have to calculate the average gradient of the N data.

## 8. Output error term
We have to calculate the error term of the output layer first to backpropagate the loss. In this section, I will introduce you the famous pattern of the output error term.

### 8.1 Mean squared loss with identity function
#### - Loss function
$$L = \sum_j \frac{1}{2}(y_j - \hat{y_j})^2$$

#### - Activation function
$$\phi^{o}(z)=z$$

#### - Output error term
$$
\delta_j^{o}
=\frac{\partial L}{\partial a_j^{o}} \phi'^{o}(z_j^{o})
=\frac{\partial L}{\partial \hat{y_j}} \phi'^{o}(z_j^{o})
=\frac{\partial L}{\partial \hat{y_j}} \cdot 1
=\frac{\partial}{\partial \hat{y_j}} \left( \sum_j \frac{1}{2}(y_j - \hat{y_j})^2 \right)
= -(y_j - \hat{y_j})
$$

### 8.2 Cross entropy loss (Binary Classification) with sigmoid function
- Two class classification
- $y_i=0$ or $1$

#### - Loss function
$$L = -\sum_j y_j \log{\hat{y_j}}=-\sum_j \left( y_i \log{\hat{y_j} + (1-y_i)\log(1-\hat{y_i})}\right)$$

#### - Activation function
$$\phi^{o}(z)=\frac{1}{1+e^{-z}}$$

#### - Derivative of the activation function
$$\phi'^{o}(z)=\phi^{o}(z)(1-\phi^{o}(z))$$

#### - Output error term
$$
\begin{align*}
\delta_j^{o}
&=\frac{\partial L}{\partial a_j^{o}} \phi'^{o}(z_j^{o})
=\frac{\partial L}{\partial \hat{y_j}} \phi'^{o}(z_j^{o})
=\frac{\partial L}{\partial \hat{y_j}} \phi^{o}(z_j^o)(1-\phi^{o}(z_j^o)) \\
&=\frac{\partial}{\partial \hat{y_j}} \left(-\sum_j \left( y_i \log{\hat{y_j} + (1-y_i)\log(1-\hat{y_i})}\right) \right)\hat{y_j} (1-\hat{y_j}) \\
&= \left( -y_j\frac{1}{\hat{y_j}} -(1-y_j)\frac{1}{1-\hat{y_j}} \right)\hat{y_j} (1-\hat{y_j}) = \left(- \frac{y_j-\hat{y_j}}{\hat{y_j}(1-\hat{y_j})} \right) \hat{y_j} (1-\hat{y_j}) \\
&= -(y_j - \hat{y_j})
\end{align*}
$$
