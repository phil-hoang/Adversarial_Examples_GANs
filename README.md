# Adversarial Examples and Generative Adversarial Networks
Recent advances in neural networks for example for image recognition or text recognition have lead to broader adoptions. However, it has been shown that most of these networks are vulnurable to attacks with adversarial examples [1]. These are inputs, for example an image, which has been modified in such a way that the classifier predicts a wrong class. Attacks with adversarial exampels even work through sensors like cameras [2] which makes it important for example for autonomous vehicles.

In addition to adversarial examples there are GANs [3].

The goal of this repository is to explore the state-of-the-art in adversarial examples, defense mechanisms and GANs. The repository is split into two parts. The first section focuses on example generation, attacks and defense mechanisms. In the second part we deal with GANs


Two datasets are used:

1. MNIST dev set with 10,000 examples [4]
2. Dataset with 1000 examples from [NIPS 2017: Adversarial Learning Development Set](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set#categories.csv). Examples are similar to the ImageNet dataset [5].


## 1. Adversarial Examples
Neural networks consistently misclassify intentionally perturbed inputs [3]. The perturbations can be applied in such a way that the network predicts a different class with high confidence. To the human eye however, the perturbations are, up to a certrain level, barely perceptible [3]. Inputs manipulated with the goal of being classified as a different class are called *adversarial examples* [???]. As shown in [1, 3], the same example gets falsely classified as the same class by different networks. This is called transferability.

[2] show that the perturbations persist even through a camera. The noise introduced by using a phone camera does not destroy the effect.

### 1.1 Types of attacks
Since the discovery of the existence of adversarial examples in [1] methods have been developed which can be grouped into the following categories [6].

**White box**
Attacker has full access to the model with all it's parameters.

**Black box with probing**
Attacker has no access do model's parameters. However, the model can be querried to approximate the gradients.

**Black box without probing**
Here, the attacker has neither access nor can he querry the model.

**Digital attack**
Attacker has direct access to digital data fed into the model.

Moreover, attacks can be **targeted** or **untargeted**. In the latter scenario the attack is succesfull if any wrong class is predicted. For the fromer attack, a specific class is predicted.



### 1.2 How to generate adversarial examples
For this part there are two notebooks. Both implement the attacks described below. The first one `1_1-Adversarial_Examples_LeNet_MNIST.ipynb` uses the small network on the `MNIST` dataset. The second `1_2-Adversarial_Examples_googleNet_ImageNet.ipynb` uses a pretrained `googleNet` on `ImageNet` data. Both notebooks include detailed explanations about the algorithms and their implementation.


To generate adversarial examples there are different algorithms. For this repository we chose the following:

**Fast Gradient Sign Method (FGSM)**

This untargeted white box attack introduced in [3] generates examples quickly by linearily approximating the optimal perturbation. It performs one step into the direction of increased loss.

\begin{equation}
\tag{1.1}
\widetilde{x} = x + \eta
\end{equation}

\begin{equation}
\tag{1.2}
\eta = \epsilon \cdot sign(\nabla_{x} J(\Theta, x, y))
\end{equation}



**Basic Iterative Method (BIM)** 

This extension to the FGSM applies it multiple times and clips values after each step. This ensures the result remains similar to the origial image [5].



**Iterative Least-Likely Class Method**



**DeepFool**


A comparison of these methods on the two datasets shows:

....

Shows that while FGSM achieves good results on the smaller, black-and-white MNIST dataset its performance is low on the 1000 class dataset with RGB colour images, as mentioned in [5].


The following plot gives an overview of the average accuracy and confidence for FGSM and BIM in MNIST:

 ![Average accuracy amd confidene on MNIST](/plots/LeNet_MNIST/Adversarial_Examples_MNIST_all.png "Title")



### 1.3 Defenses against attacks

Adversarial training -> Chose examples wisely. Using overly perturbed examples can lead to a decrease of robustness [10]



## 2. Generative Adversarial Networks (GANs)






## References

[1]  Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. ArXiv:1312.6199 [Cs]. http://arxiv.org/abs/1312.6199

[2] Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. ArXiv:1607.02533 [Cs, Stat]. http://arxiv.org/abs/1607.02533

[3] Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. ArXiv:1412.6572 [Cs, Stat]. http://arxiv.org/abs/1412.6572

[4] &emsp; LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit database. ATT Labs [Online]. Available: Http://Yann. Lecun. Com/Exdb/Mnist, 2.

[5] &emsp; Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., & Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3), 211–252. https://doi.org/10.1007/s11263-015-0816-y

---------------

[2] Engstrom, L., Ilyas, A., & Athalye, A. (2018). Evaluating and Understanding the Robustness of Adversarial Logit Pairing. ArXiv:1807.10272 [Cs, Stat]. http://arxiv.org/abs/1807.10272


[4] Kannan, H., Kurakin, A., & Goodfellow, I. (2018). Adversarial Logit Pairing. ArXiv:1803.06373 [Cs, Stat]. http://arxiv.org/abs/1803.06373




[6] Kurakin, A., Goodfellow, I., Bengio, S., Dong, Y., Liao, F., Liang, M., Pang, T., Zhu, J., Hu, X., Xie, C., Wang, J., Zhang, Z., Ren, Z., Yuille, A., Huang, S., Zhao, Y., Zhao, Y., Han, Z., Long, J., … Abe, M. (2018). Adversarial Attacks and Defences Competition. ArXiv:1804.00097 [Cs, Stat]. http://arxiv.org/abs/1804.00097

[7] Lu, J., Issaranon, T., & Forsyth, D. (2017). SafetyNet: Detecting and Rejecting Adversarial Examples Robustly. ArXiv:1704.00103 [Cs]. http://arxiv.org/abs/1704.00103

[8] Neekhara, P., Hussain, S., Jere, M., Koushanfar, F., & McAuley, J. (2020). Adversarial Deepfakes: Evaluating Vulnerability of Deepfake Detectors to Adversarial Examples. ArXiv:2002.12749 [Cs]. http://arxiv.org/abs/2002.12749

[9] Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). Practical Black-Box Attacks against Machine Learning. ArXiv:1602.02697 [Cs]. http://arxiv.org/abs/1602.02697

[10] Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2016.282

