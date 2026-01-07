# FraudNet
A deep-learning network built in Rust for detecting Fraud Waste and Abuse. Compiled to the onnx runtime for use client-side.

Implementation:

This is a arbitrarily-scalable neural network capable of creating a classification network of N-inputs, should look something like the following:

```
Input Layer (3 features after UMAP) 
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (52 neurons, ReLU) 
    ↓
Hidden Layer 3 (42 neurons, ReLU)
    ↓
Hidden Layer 4 (32 neurons, ReLU)
    ↓
Hidden Layer 5 (26 neurons, ReLU)
    ↓
Hidden Layer 6 (22 neurons, ReLU)
    ↓
Hidden Layer 7 (20 neurons, ReLU)
    ↓
Hidden Layer 8 (16 neurons, ReLU)
    ↓
Hidden Layer 9 (8 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

Finally, the network, once trained, should be compiled to be compatible with the ONNX runtime, to be used client side in a web-browser.
