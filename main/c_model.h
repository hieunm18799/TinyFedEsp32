#ifndef C_MODEL
#define C_MODEL

#include <cmath>
#include "FC_layer.h"

class CModel {
public:
    unsigned char length;
    unsigned int *nodesLength;
    unsigned char batchSize;
    float learningRate;
    float lambdaValue;

    FCLayer *layers;

    //Category cross entropy
    float cross_entropy_loss(unsigned char *groundTruth) {
        float result = 0.0f;

        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            for (unsigned int node = 0; node < nodesLength[length]; ++node) {
                float temp = layers[length - 1].outputData[batch * nodesLength[length] + node];
                // if (temp < epsilon) temp = epsilon;
                if (groundTruth[batch * nodesLength[length] + node]) result -= logf(temp);
            }
        }
        return result / float(batchSize);
    }

    void cross_entropy_loss_gradient(unsigned char *groundTruth) {
        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            for (unsigned int node = 0; node < nodesLength[length]; ++node) {
                unsigned int temp = batch * nodesLength[length] + node;
                layers[length - 1].grad[temp] = - groundTruth[temp] / layers[length - 1].outputData[temp];
            }
        }
    }

    float accuracy(unsigned char *groundTruth) {
        unsigned char correct = 0;

        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            unsigned int maxNode = 0;
            for (unsigned int node = 1; node < nodesLength[length]; ++node) {
                if (layers[length - 1].outputData[batch * nodesLength[length] + node] > layers[length - 1].outputData[batch * nodesLength[length] + maxNode]) maxNode = node;
            }
            if (groundTruth[batch * nodesLength[length] + maxNode] == 1) correct++;
        }
        return float(correct) / float(batchSize);
    }

    ~CModel() {
        delete[] layers;
    }

    CModel (unsigned char length, unsigned int* nodesLength, unsigned char batchSize, float learningRate, float lambdaValue)
    : length(length), nodesLength(nodesLength), batchSize(batchSize), learningRate(learningRate), lambdaValue(lambdaValue) {
        layers = (FCLayer*)calloc(length, sizeof(FCLayer));
        for (unsigned int layer = 0; layer < length - 1; ++layer){
            layers[layer] = *new FCLayer((unsigned int)nodesLength[layer], (unsigned int)nodesLength[layer + 1], batchSize, &FCLayer::sigmoid, &FCLayer::sigmoidDerivative);
        }
        layers[length - 1] = *new FCLayer((unsigned int)nodesLength[length - 1], (unsigned int)nodesLength[length], batchSize, &FCLayer::softmax, &FCLayer::softmaxDerivative);
    }

    // Forward pass
    void forward (float *input) {
        float* input_temp = input;
        for (unsigned char layer = 0; layer < length ; ++layer) {
            layers[layer].forward(input_temp, nodesLength[layer], nodesLength[layer + 1], batchSize);
            input_temp = layers[layer].outputData;
        }
    }

    // Backward pass to update weights
    void backward (float *input, unsigned char *groundTruth) {
        for (unsigned char layer = 0; layer < length; ++layer) {
            std::fill(layers[layer].grad, layers[layer].grad + batchSize * nodesLength[layer + 1], 0.f);
        }
        cross_entropy_loss_gradient(groundTruth);
        for (unsigned char layer = length - 1; layer != 0; --layer) {
            layers[layer].backward(layers[layer - 1].outputData, nodesLength[layer], nodesLength[layer + 1], layers[layer - 1].grad, learningRate, lambdaValue, batchSize);
        }
        layers[0].backward(input, nodesLength[0], nodesLength[1], NULL, learningRate, lambdaValue, batchSize);
    }
};


#endif
