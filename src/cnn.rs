use crate::matrix::Matrix;

/// A 4D tensor representation for CNN operations
/// Shape: (batch_size, channels, height, width)
#[derive(Debug, Clone)]
pub struct Tensor4D {
    pub batch_size: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f64>,
}

impl Tensor4D {
    /// Create a new tensor filled with zeros
    pub fn zeros(batch_size: usize, channels: usize, height: usize, width: usize) -> Self {
        let size = batch_size * channels * height * width;
        Tensor4D {
            batch_size,
            channels,
            height,
            width,
            data: vec![0.0; size],
        }
    }

    /// Create a new tensor from a flat vector
    pub fn from_vec(
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f64>,
    ) -> Self {
        assert_eq!(data.len(), batch_size * channels * height * width);
        Tensor4D {
            batch_size,
            channels,
            height,
            width,
            data,
        }
    }

    /// Get element at (b, c, h, w)
    pub fn get(&self, b: usize, c: usize, h: usize, w: usize) -> f64 {
        let idx = ((b * self.channels + c) * self.height + h) * self.width + w;
        self.data[idx]
    }

    /// Set element at (b, c, h, w)
    pub fn set(&mut self, b: usize, c: usize, h: usize, w: usize, value: f64) {
        let idx = ((b * self.channels + c) * self.height + h) * self.width + w;
        self.data[idx] = value;
    }

    /// Apply a function to each element
    pub fn map<F>(&self, f: F) -> Tensor4D
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Tensor4D::from_vec(self.batch_size, self.channels, self.height, self.width, data)
    }
}

/// Convolutional layer
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Vec<f64>, // Shape: (out_channels, in_channels, kernel_size, kernel_size)
    pub bias: Vec<f64>,    // Shape: (out_channels,)
}

impl Conv2d {
    /// Create a new convolutional layer with random weights
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Self {
        use crate::utils::SimpleRng;

        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let mut rng = SimpleRng::new(seed);

        // Xavier initialization
        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f64).sqrt();
        let weights: Vec<f64> = (0..weight_count)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale)
            .collect();

        let bias = vec![0.0; out_channels];

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
        }
    }

    /// Get weight for a specific filter and position
    fn get_weight(&self, out_c: usize, in_c: usize, kh: usize, kw: usize) -> f64 {
        let idx = ((out_c * self.in_channels + in_c) * self.kernel_size + kh) * self.kernel_size + kw;
        self.weights[idx]
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        assert_eq!(input.channels, self.in_channels);

        let out_h = (input.height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (input.width + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = Tensor4D::zeros(input.batch_size, self.out_channels, out_h, out_w);

        for b in 0..input.batch_size {
            for out_c in 0..self.out_channels {
                for h in 0..out_h {
                    for w in 0..out_w {
                        let mut sum = self.bias[out_c];

                        for in_c in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let h_idx = h * self.stride + kh;
                                    let w_idx = w * self.stride + kw;

                                    // Apply padding
                                    if h_idx >= self.padding
                                        && h_idx < input.height + self.padding
                                        && w_idx >= self.padding
                                        && w_idx < input.width + self.padding
                                    {
                                        let input_h = h_idx - self.padding;
                                        let input_w = w_idx - self.padding;

                                        if input_h < input.height && input_w < input.width {
                                            let input_val = input.get(b, in_c, input_h, input_w);
                                            let weight_val = self.get_weight(out_c, in_c, kh, kw);
                                            sum += input_val * weight_val;
                                        }
                                    }
                                }
                            }
                        }

                        output.set(b, out_c, h, w, sum);
                    }
                }
            }
        }

        output
    }
}

/// Batch Normalization layer
pub struct BatchNorm2d {
    pub num_features: usize,
    pub gamma: Vec<f64>, // Scale parameter
    pub beta: Vec<f64>,  // Shift parameter
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub momentum: f64,
    pub epsilon: f64,
    pub training: bool,
}

impl BatchNorm2d {
    /// Create a new batch normalization layer
    pub fn new(num_features: usize) -> Self {
        BatchNorm2d {
            num_features,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            momentum: 0.1,
            epsilon: 1e-5,
            training: true,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Tensor4D) -> Tensor4D {
        assert_eq!(input.channels, self.num_features);

        let mut output = input.clone();

        if self.training {
            // Calculate mean and variance for each channel
            for c in 0..self.num_features {
                let mut sum = 0.0;
                let mut count = 0;

                for b in 0..input.batch_size {
                    for h in 0..input.height {
                        for w in 0..input.width {
                            sum += input.get(b, c, h, w);
                            count += 1;
                        }
                    }
                }

                let mean = sum / count as f64;

                let mut var_sum = 0.0;
                for b in 0..input.batch_size {
                    for h in 0..input.height {
                        for w in 0..input.width {
                            let diff = input.get(b, c, h, w) - mean;
                            var_sum += diff * diff;
                        }
                    }
                }

                let variance = var_sum / count as f64;

                // Update running statistics
                self.running_mean[c] =
                    (1.0 - self.momentum) * self.running_mean[c] + self.momentum * mean;
                self.running_var[c] =
                    (1.0 - self.momentum) * self.running_var[c] + self.momentum * variance;

                // Normalize
                let std = (variance + self.epsilon).sqrt();
                for b in 0..input.batch_size {
                    for h in 0..input.height {
                        for w in 0..input.width {
                            let normalized = (input.get(b, c, h, w) - mean) / std;
                            let scaled = self.gamma[c] * normalized + self.beta[c];
                            output.set(b, c, h, w, scaled);
                        }
                    }
                }
            }
        } else {
            // Use running statistics for inference
            for c in 0..self.num_features {
                let std = (self.running_var[c] + self.epsilon).sqrt();
                for b in 0..input.batch_size {
                    for h in 0..input.height {
                        for w in 0..input.width {
                            let normalized = (input.get(b, c, h, w) - self.running_mean[c]) / std;
                            let scaled = self.gamma[c] * normalized + self.beta[c];
                            output.set(b, c, h, w, scaled);
                        }
                    }
                }
            }
        }

        output
    }
}

/// Max Pooling layer
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    /// Create a new max pooling layer
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let out_h = (input.height - self.kernel_size) / self.stride + 1;
        let out_w = (input.width - self.kernel_size) / self.stride + 1;

        let mut output = Tensor4D::zeros(input.batch_size, input.channels, out_h, out_w);

        for b in 0..input.batch_size {
            for c in 0..input.channels {
                for h in 0..out_h {
                    for w in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let h_idx = h * self.stride + kh;
                                let w_idx = w * self.stride + kw;

                                if h_idx < input.height && w_idx < input.width {
                                    let val = input.get(b, c, h_idx, w_idx);
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        output.set(b, c, h, w, max_val);
                    }
                }
            }
        }

        output
    }
}

/// Zero Padding layer
pub fn zero_pad_2d(input: &Tensor4D, padding: usize) -> Tensor4D {
    let new_h = input.height + 2 * padding;
    let new_w = input.width + 2 * padding;

    let mut output = Tensor4D::zeros(input.batch_size, input.channels, new_h, new_w);

    for b in 0..input.batch_size {
        for c in 0..input.channels {
            for h in 0..input.height {
                for w in 0..input.width {
                    output.set(b, c, h + padding, w + padding, input.get(b, c, h, w));
                }
            }
        }
    }

    output
}

/// Flatten a 4D tensor to a 2D matrix (batch_size, features)
pub fn flatten(input: &Tensor4D) -> Matrix {
    let features = input.channels * input.height * input.width;
    let mut data = Vec::with_capacity(input.batch_size * features);

    for b in 0..input.batch_size {
        for c in 0..input.channels {
            for h in 0..input.height {
                for w in 0..input.width {
                    data.push(input.get(b, c, h, w));
                }
            }
        }
    }

    Matrix::from_vec(input.batch_size, features, data)
}

/// CNN for MNIST digit classification
pub struct MnistCNN {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub pool: MaxPool2d,
    pub fc_weights: Matrix,
    pub fc_bias: Matrix,
    pub learning_rate: f64,
}

impl MnistCNN {
    /// Create a new MNIST CNN
    /// Architecture:
    /// - ZeroPad2d: 28x28 -> 32x32
    /// - Conv2d: 16 filters, 5x5 kernel
    /// - BatchNorm2d: 16 features
    /// - ReLU
    /// - MaxPool2d: 2x2 kernel
    /// - Flatten: 16*14*14 = 3136 -> 10
    /// - Linear: 3136 -> 10
    /// - Softmax
    pub fn new(learning_rate: f64, seed: u64) -> Self {
        let conv1 = Conv2d::new(1, 16, 5, 1, 0, seed);
        let bn1 = BatchNorm2d::new(16);
        let pool = MaxPool2d::new(2, 2);

        // FC layer: 3136 -> 10
        let mut rng = crate::utils::SimpleRng::new(seed + 1);
        let scale = (2.0 / 3136.0_f64).sqrt();
        let fc_weights_data: Vec<f64> = (0..10 * 3136)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale)
            .collect();
        let fc_weights = Matrix::from_vec(10, 3136, fc_weights_data);
        let fc_bias = Matrix::zeros(10, 1);

        MnistCNN {
            conv1,
            bn1,
            pool,
            fc_weights,
            fc_bias,
            learning_rate,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        assert_eq!(input.rows, 784);
        assert_eq!(input.cols, 1);

        // Reshape input to 28x28 image
        let mut img_data = Vec::with_capacity(784);
        for i in 0..784 {
            img_data.push(input.data[i]);
        }
        let img = Tensor4D::from_vec(1, 1, 28, 28, img_data);

        // Zero padding: 28x28 -> 32x32
        let padded = zero_pad_2d(&img, 2);

        // Conv2d: 32x32x1 -> 28x28x16 (with 5x5 kernel, no padding)
        let conv_out = self.conv1.forward(&padded);

        // BatchNorm
        let bn_out = self.bn1.forward(&conv_out);

        // ReLU
        let relu_out = bn_out.map(|x| if x > 0.0 { x } else { 0.0 });

        // MaxPool: 28x28x16 -> 14x14x16
        let pool_out = self.pool.forward(&relu_out);

        // Flatten: 14x14x16 = 3136
        let flat = flatten(&pool_out);

        // Extract single batch
        let mut flat_vec = Vec::with_capacity(3136);
        for i in 0..3136 {
            flat_vec.push(flat.get(0, i));
        }
        let flat_matrix = Matrix::from_vec(3136, 1, flat_vec);

        // Linear: 3136 -> 10
        let fc_out = self.fc_weights.dot(&flat_matrix).add(&self.fc_bias);

        // Softmax
        self.softmax(&fc_out)
    }

    /// Predict output for given input
    pub fn predict(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
    }

    /// Softmax activation
    fn softmax(&self, x: &Matrix) -> Matrix {
        let max_val = x.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f64 = x.data.iter().map(|&v| (v - max_val).exp()).sum();

        let data: Vec<f64> = x
            .data
            .iter()
            .map(|&v| (v - max_val).exp() / exp_sum)
            .collect();

        Matrix::from_vec(x.rows, x.cols, data)
    }

    /// Train on a single example (simplified - no backprop through CNN)
    pub fn train_step(&mut self, input: &Matrix, target: &Matrix) -> f64 {
        // For now, we'll just do forward pass and calculate loss
        // Full backprop through CNN would require storing intermediate values
        let prediction = self.forward(input);

        // Calculate cross-entropy loss
        let mut loss = 0.0;
        for i in 0..10 {
            let pred = prediction.get(i, 0).max(1e-10); // Avoid log(0)
            let tgt = target.get(i, 0);
            loss -= tgt * pred.ln();
        }

        // Simple gradient descent on FC layer only
        let error = prediction.sub(target);

        // Get flattened features for this input
        assert_eq!(input.rows, 784);
        assert_eq!(input.cols, 1);

        // Reshape input to 28x28 image
        let mut img_data = Vec::with_capacity(784);
        for i in 0..784 {
            img_data.push(input.data[i]);
        }
        let img = Tensor4D::from_vec(1, 1, 28, 28, img_data);

        // Forward through CNN layers to get features
        let padded = zero_pad_2d(&img, 2);
        let conv_out = self.conv1.forward(&padded);
        let bn_out = self.bn1.forward(&conv_out);
        let relu_out = bn_out.map(|x| if x > 0.0 { x } else { 0.0 });
        let pool_out = self.pool.forward(&relu_out);
        let flat = flatten(&pool_out);

        // Extract single batch
        let mut flat_vec = Vec::with_capacity(3136);
        for i in 0..3136 {
            flat_vec.push(flat.get(0, i));
        }
        let features = Matrix::from_vec(3136, 1, flat_vec);

        // Update FC weights
        let gradient = error.dot(&features.transpose());
        let weight_update = gradient.scale(self.learning_rate);
        self.fc_weights = self.fc_weights.sub(&weight_update);

        // Update FC bias
        let bias_update = error.scale(self.learning_rate);
        self.fc_bias = self.fc_bias.sub(&bias_update);

        loss
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.bn1.training = training;
    }
}
