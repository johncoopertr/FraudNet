/**
 * FraudNet - Client-Side Neural Network Implementation
 * Pure JavaScript implementation of the neural network for browser execution
 */

class Matrix {
    constructor(rows, cols, data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Array(rows * cols).fill(0);
    }
    
    static fromArray(arr) {
        return new Matrix(arr.length, 1, arr);
    }
    
    get(row, col) {
        return this.data[row * this.cols + col];
    }
    
    set(row, col, value) {
        this.data[row * this.cols + col] = value;
    }
    
    // Matrix multiplication
    dot(other) {
        if (this.cols !== other.rows) {
            throw new Error(`Matrix dimension mismatch: ${this.cols} != ${other.rows}`);
        }
        
        const result = new Matrix(this.rows, other.cols);
        
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        
        return result;
    }
    
    // Element-wise addition
    add(other) {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error('Matrix dimensions must match for addition');
        }
        
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        
        return result;
    }
    
    // Apply a function to each element
    map(func) {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = func(this.data[i]);
        }
        return result;
    }
    
    toArray() {
        return Array.from(this.data);
    }
}

// Activation functions
function relu(x) {
    return Math.max(0, x);
}

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Neural Network class for client-side inference
 */
class NeuralNetwork {
    constructor(modelData) {
        this.layerSizes = modelData.layer_sizes;
        this.learningRate = modelData.learning_rate;
        
        // Convert serialized weights and biases to Matrix objects
        this.weights = modelData.weights.map(w => 
            new Matrix(w.rows, w.cols, w.data)
        );
        
        this.biases = modelData.biases.map(b => 
            new Matrix(b.rows, b.cols, b.data)
        );
    }
    
    /**
     * Perform forward pass through the network
     * @param {Array} inputArray - Input features as array
     * @returns {Array} - Output predictions as array
     */
    predict(inputArray) {
        // Convert input array to column vector matrix
        let current = Matrix.fromArray(inputArray);
        
        // Forward pass through each layer
        for (let i = 0; i < this.weights.length; i++) {
            // z = W * a + b
            current = this.weights[i].dot(current).add(this.biases[i]);
            
            // Apply activation function
            if (i < this.weights.length - 1) {
                // ReLU for hidden layers
                current = current.map(relu);
            } else {
                // Sigmoid for output layer
                current = current.map(sigmoid);
            }
        }
        
        return current.toArray();
    }
    
    /**
     * Get network architecture as string
     */
    getArchitecture() {
        return this.layerSizes.join(' → ');
    }
    
    /**
     * Get total number of parameters (weights + biases)
     */
    getParameterCount() {
        let count = 0;
        for (let i = 0; i < this.weights.length; i++) {
            count += this.weights[i].data.length;
            count += this.biases[i].data.length;
        }
        return count;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeuralNetwork, Matrix };
}
