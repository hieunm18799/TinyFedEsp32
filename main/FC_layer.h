#ifndef FC_LAYER
#define FC_LAYER

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

// #include <iostream>
#include <cmath>
#include <cstring>
// #include <Arduino.h>

// Define a simple neural network layer structure
class FCLayer {
public:
    std::function<void(FCLayer&, float*, unsigned int)> activationFunc;
    std::function<void(FCLayer&, float*, float*, unsigned int)> derivationFunc;

    float *outputData;
    float *weights, *grad;

    ~FCLayer() {
        delete[] weights;
        delete[] grad;
    }

    FCLayer (unsigned int inputSize, unsigned int outputSize, unsigned char batchSize, std::function<void(FCLayer&, float*, unsigned int)> activationFunc, std::function<void(FCLayer&, float*, float*, unsigned int)> derivationFunc)
        : activationFunc(activationFunc), derivationFunc(derivationFunc) {
        outputData = (float*)calloc(batchSize * outputSize, sizeof(float));
        grad = (float*)calloc(batchSize * outputSize, sizeof(float));
        weights = (float *)calloc((inputSize + 1) * outputSize, sizeof(float));
    }

    // Sigmoid
    void sigmoid(float* arr, unsigned int size) {
        for (unsigned int node = 0; node < size; ++node) {
            arr[node] = 1.0f / (1.0f + expf(-arr[node]));
        }
    }

    void sigmoidDerivative (float* output, float* grad, unsigned int size) {
        for (unsigned int node = 0; node < size; ++node) {
            grad[node] *= output[node] * (1.0f - output[node]);
        }
    }

    //Softmax
    void softmax(float* arr, unsigned int size) {
        float sum = 0.0f;
        for (unsigned int node = 0; node < size; ++node) {
            sum += exp(arr[node]);
        }
        for (unsigned int node = 0; node < size; ++node) {
            arr[node] = expf(arr[node]) / sum;
        }
    }

    void softmaxDerivative(float* output, float* grad, unsigned int size) {
        float * grad_temp = (float*)calloc(size, sizeof(float));
        std::fill(grad_temp, grad_temp + size, 0.f);
        for (unsigned int i = 0; i < size; ++i) {
            for (unsigned int j = 0; j < size; ++j) {
                grad_temp[i] += grad[j] * output[j] * ((i == j) - output[i]);
            }
        }
        std::transform(grad_temp, grad_temp + size, grad, [](float value) { return value; });
        free(grad_temp);
    }

    // Forward pass
    void forward (float *input, unsigned int inputSize, unsigned int outputSize, unsigned char batchSize) {
        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            for (unsigned int outputNode = 0; outputNode < outputSize; ++outputNode) {
                outputData[batch * outputSize + outputNode] = weights[outputNode];
                for (unsigned int inputNode = 0; inputNode < inputSize; ++inputNode) {
                    outputData[batch * outputSize + outputNode] += input[batch * inputSize + inputNode] * weights[(inputNode + 1) * outputSize + outputNode];
                }
            }
            activationFunc(*this, outputData + batch * outputSize, outputSize);
        }
    }

    // Backward pass to update weights
    void backward (float *input, unsigned int inputSize, unsigned int outputSize, float* gradBack, float learningRate, float lambdaValue, unsigned char batchSize) {
        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            derivationFunc(*this, outputData + batch * outputSize, grad + batch * outputSize, outputSize);
        }
        
        if (gradBack != NULL) {
            for (unsigned char batch = 0; batch < batchSize; ++batch) {
                // unsigned int outRow = batch * outputSize, inRow = batch * inputSize;
                for (unsigned int inputNode = 0; inputNode < inputSize; ++inputNode) {
                    // unsigned int weiRow = (inputNode + 1) * outputSize;
                    for (unsigned int outputNode = 0; outputNode < outputSize; ++outputNode) {
                        gradBack[batch * inputSize + inputNode] += grad[batch * outputSize + outputNode] * weights[(inputNode + 1) * outputSize + outputNode]  / batchSize;
                    }
                }
            }
        }

        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            // unsigned int outRow = batch * outputSize;
            for (unsigned int outputNode = 0; outputNode < outputSize; ++outputNode) {
                weights[outputNode] -= learningRate * grad[batch * outputSize + outputNode] / batchSize;
            }
        }

        for (unsigned int inputNode = 0; inputNode < inputSize; ++inputNode) {
            for (unsigned int outputNode = 0; outputNode < outputSize; ++outputNode) {
                float temp = 0.f;
                for (unsigned char batch = 0; batch < batchSize; ++batch) {
                    temp += input[batch * inputSize + inputNode] * grad[batch * outputSize + outputNode];
                }
                // weights[(inputNode + 1) * outputSize + outputNode] += learningRate * temp / batchSize;
                weights[(inputNode + 1) * outputSize + outputNode] = weights[(inputNode + 1) * outputSize + outputNode] * (1 - learningRate * lambdaValue / batchSize) - learningRate * temp / batchSize;
            }
        }
    }
};


#endif
