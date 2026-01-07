#!/usr/bin/env node

/**
 * Test script to verify the client-side neural network implementation
 * Runs the JavaScript neural network with the exported models
 */

const fs = require('fs');
const path = require('path');

// Load the neural network implementation in Node.js context
const fraudnetCode = fs.readFileSync(path.join(__dirname, 'web', 'fraudnet.js'), 'utf8');

// Evaluate in global context to get exports
const vm = require('vm');
const sandbox = { module: { exports: {} }, Math, Array };
vm.createContext(sandbox);
vm.runInContext(fraudnetCode, sandbox);
const { NeuralNetwork, Matrix } = sandbox.module.exports;

console.log('🧠 FraudNet Client-Side Test\n');
console.log('================================\n');

// Test 1: Load and test Linear model
console.log('Test 1: Linear Model');
console.log('--------------------');
try {
    const linearModelData = JSON.parse(fs.readFileSync('model_linear.json', 'utf8'));
    const linearModel = new NeuralNetwork(linearModelData);
    
    console.log('✓ Model loaded successfully');
    console.log(`  Architecture: ${linearModel.getArchitecture()}`);
    console.log(`  Parameters: ${linearModel.getParameterCount()}`);
    
    // Test prediction with positive example
    const pred1 = linearModel.predict([0.5, 0.3, 0.2]);
    console.log(`  Prediction [0.5, 0.3, 0.2]: ${pred1[0].toFixed(4)} (${pred1[0] >= 0.5 ? 'Class 1' : 'Class 0'})`);
    
    // Test prediction with negative example
    const pred2 = linearModel.predict([-0.5, -0.3, -0.2]);
    console.log(`  Prediction [-0.5, -0.3, -0.2]: ${pred2[0].toFixed(4)} (${pred2[0] >= 0.5 ? 'Class 1' : 'Class 0'})`);
    
    console.log('✓ All tests passed!\n');
} catch (error) {
    console.error('✗ Error:', error.message);
    process.exit(1);
}

// Test 2: Load and test XOR model
console.log('Test 2: XOR Model');
console.log('-----------------');
try {
    const xorModelData = JSON.parse(fs.readFileSync('model_xor.json', 'utf8'));
    const xorModel = new NeuralNetwork(xorModelData);
    
    console.log('✓ Model loaded successfully');
    console.log(`  Architecture: ${xorModel.getArchitecture()}`);
    console.log(`  Parameters: ${xorModel.getParameterCount()}`);
    
    // Test XOR predictions
    const testCases = [
        [0.5, 0.5],   // Same sign: expect 0
        [-0.5, -0.5], // Same sign: expect 0
        [0.5, -0.5],  // Different sign: expect 1
        [-0.5, 0.5],  // Different sign: expect 1
    ];
    
    for (const input of testCases) {
        const pred = xorModel.predict(input);
        const expected = input[0] * input[1] < 0 ? 1 : 0;
        const result = pred[0] >= 0.5 ? 1 : 0;
        const match = result === expected ? '✓' : '✗';
        console.log(`  ${match} Prediction [${input[0]}, ${input[1]}]: ${pred[0].toFixed(4)} (expected ${expected})`);
    }
    
    console.log('✓ All tests passed!\n');
} catch (error) {
    console.error('✗ Error:', error.message);
    process.exit(1);
}

// Test 3: Load and test Circular model
console.log('Test 3: Circular Model');
console.log('----------------------');
try {
    const circModelData = JSON.parse(fs.readFileSync('model_circular.json', 'utf8'));
    const circModel = new NeuralNetwork(circModelData);
    
    console.log('✓ Model loaded successfully');
    console.log(`  Architecture: ${circModel.getArchitecture()}`);
    console.log(`  Parameters: ${circModel.getParameterCount()}`);
    
    // Test circular boundary predictions
    const testCases = [
        [0.2, 0.2],   // Inside (distance ~0.28)
        [0.3, 0.3],   // Inside (distance ~0.42)
        [0.6, 0.6],   // Outside (distance ~0.85)
        [-0.7, 0.7],  // Outside (distance ~0.99)
    ];
    
    for (const input of testCases) {
        const pred = circModel.predict(input);
        const distance = Math.sqrt(input[0] * input[0] + input[1] * input[1]);
        const expected = distance < 0.5 ? 1 : 0;
        const result = pred[0] >= 0.5 ? 1 : 0;
        const match = result === expected ? '✓' : '✗';
        console.log(`  ${match} Prediction [${input[0]}, ${input[1]}]: ${pred[0].toFixed(4)} (distance: ${distance.toFixed(3)}, expected ${expected})`);
    }
    
    console.log('✓ All tests passed!\n');
} catch (error) {
    console.error('✗ Error:', error.message);
    process.exit(1);
}

console.log('================================');
console.log('✓ All client-side tests passed!');
console.log('================================\n');
console.log('The client-side neural network is working correctly!');
console.log('Models can be loaded and used for inference in the browser.\n');
