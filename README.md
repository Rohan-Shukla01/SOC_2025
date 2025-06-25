# SOC_2025
## Topics I Have Learned and Applied in This Course

1. **Understanding the Basics of NLP through Various Resources**
   - String pattern matching  
   - One-hot vectors and encodings  
   - Tokenization and stop word removal  
   - Stemming and lemmatization (using nltk package in python)

2. **Basic Implementation of Neural Network Components Using PyTorch**
   - Linear regression and logistic regression  
   - Activation functions  
   - Fully functional multi-layer perceptrons (MLPs)  
   - Convolutional neural networks (CNNs)  
   - Recurrent neural networks (RNNs)  
   - Basic classification models (e.g., MNIST, CIFAR-10)  

3. **Beginner Tutorials and Projects for Better Understanding**
   - Emoji prediction model

## Vanilla RNN Experiment

This experiment tests a simple vanilla RNN to predict the next character in a synthetically generated sequence. The sequences are built using random choices from a rule-based grammar. The model tries to learn these transition rules.

**Note**: This is a toy problem — the ground truth is based on `random.choice()`, so there is inherent unpredictability in the data. The RNN is being used more to demonstrate training mechanics and sequence modeling, rather than to predict deterministic outcomes.

The model is implemented from scratch using PyTorch.

File: `Vanilla_RNN.py`
