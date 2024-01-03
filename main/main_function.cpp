#include "main_function.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    // static std::vector<FCLayer> devices;
    // const uint32_t kTensorArenaSize = 626688; // tf-mobilenet
    const uint32_t kTensorArenaSize = 320000; // perf-mobilenet
    static unsigned char *tensor_arena;
}  // namespace

// Variables
CModel *cModel;
unsigned char batchSize;
unsigned char round_num = 0;
unsigned char localEpoch;
unsigned char *groundTruth;
unsigned long int time_saved;
float time_forward, time_backward;
float *inputDatas;

void init_network_model();
void run(bool only_forward);
float readFloat32();
void readByte(unsigned char *temp);
void sendFloat(float arg);

/**
 * @brief      Arduino setup function
 */
void setup() {
    uart_init();

    init_network_model();

    groundTruth = (unsigned char*)calloc(batchSize * OutputNode, sizeof(unsigned char));
    inputDatas = (float*)calloc(batchSize * FeatureNodes, sizeof(float));

    round_num = 0;

    std::cout << "Received new model.\n";
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    unsigned char read;
    readByte(&read);
    if (read == 'r') { // Train with a sample
        std::fill(groundTruth, groundTruth + batchSize * OutputNode, 0);
        std::fill(inputDatas, inputDatas + batchSize * FeatureNodes, 0.f);
        time_forward = 0.f;
        time_backward = 0.f;

        unsigned char only_forward;
        readByte(&only_forward);
        std::cout << static_cast<unsigned int>(only_forward) << "\n";
        
        for (unsigned char batch = 0; batch < batchSize; ++batch) {
            std::cout << "ok\n";

            unsigned char label;
            readByte(&label);
            std::cout << static_cast<unsigned int>(label) << "\n";
            groundTruth[batch * OutputNode + label] = 1;
            
            for(unsigned int i = 0; i < InputNode; ++i) {
                unsigned char temp;
                readByte(&temp);
                // input->data.int8[i] = temp - 128;
                input->data.uint8[i] = temp;
            }
            time_saved = esp_timer_get_time();

            // Run the model on this input and make sure it succeeds.
            if (kTfLiteOk != interpreter->Invoke()) {
                std::cout << "Invoke failed.\n";
            }
            time_forward += (esp_timer_get_time() - time_saved) / 1000;
            TfLiteTensor* output = interpreter->output(0);

            // std::transform(output->data.int8, output->data.int8 + FeatureNodes, inputDatas + batch * FeatureNodes, [](int value) { return value / 128; });
            std::transform(output->data.f, output->data.f + FeatureNodes, inputDatas + batch * FeatureNodes, [](float value) { return value; });
        }

        run(only_forward);
    } else if (read == '>') { // s -> FEDERATED LEARNING
        /***********************
         * Federate Learning
         ***********************/
        std::cout << '<';
        unsigned char conf;
        readByte(&conf);
        if (conf == 'f') {
            std::cout << "start\n";
            std::cout << static_cast<unsigned int>(round_num) << "\n";
            round_num = 0;

            /*******
             * Sending model
             *******/
            for (unsigned char layer = 0; layer < LayerLength; ++layer) {
                for (unsigned int inputNode = 0; inputNode < cModel->nodesLength[layer] + 1; ++inputNode) {
                    for (unsigned int outputNode = 0; outputNode < cModel->nodesLength[layer + 1]; ++outputNode) {
                        sendFloat(cModel->layers[layer].weights[inputNode * cModel->nodesLength[layer + 1] + outputNode]);
                    }
                }
            }

            /*****
             * Receiving model
             *****/
            for (unsigned char layer = 0; layer < LayerLength; ++layer) {
                for (unsigned int inputNode = 0; inputNode < cModel->nodesLength[layer] + 1; ++inputNode) {
                    for (unsigned int outputNode = 0; outputNode < cModel->nodesLength[layer + 1]; ++outputNode) {
                        cModel->layers[layer].weights[inputNode * cModel->nodesLength[layer + 1] + outputNode] = readFloat32();
                    }
                }
            }

            std::cout << "done fl\n";
        }

    }
}

void init_network_model() {
    unsigned char startChar;
    do {
        std::cout << "Waiting for new model...\n";
        readByte(&startChar);
    } while(startChar != 's'); // s -> START

    std::cout << "start\n";
    float learningRate = readFloat32();
    float lambdaVal = readFloat32();
    readByte(&localEpoch);
    readByte(&batchSize);

    cModel = new CModel(LayerLength, (unsigned int *)Nodes, batchSize, learningRate, lambdaVal);

    for (unsigned char layer = 0; layer < LayerLength; ++layer) {
        for (unsigned int inputNode = 0; inputNode < cModel->nodesLength[layer] + 1; ++inputNode) {
            for (unsigned int outputNode = 0; outputNode < cModel->nodesLength[layer + 1]; ++outputNode) {
                cModel->layers[layer].weights[inputNode * cModel->nodesLength[layer + 1] + outputNode] = readFloat32();
            }
        }
    }

    //Preruned model
    //Mobilenet unsigned char
    tensor_arena = (unsigned char*)calloc(kTensorArenaSize, sizeof(unsigned char));

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(model_data_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cout << "Model provided is schema version " << model->version() << " not equal to supported version" << TFLITE_SCHEMA_VERSION << ".\n";
        return;
    }
    // TF-Mobilenet
    // static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    // micro_op_resolver.AddDequantize();
    // micro_op_resolver.AddQuantize();
    // micro_op_resolver.AddMul();
    // micro_op_resolver.AddSub();
    // micro_op_resolver.AddAdd();
    // micro_op_resolver.AddDepthwiseConv2D();
    // micro_op_resolver.AddConv2D();
    // micro_op_resolver.AddAveragePool2D();
    // micro_op_resolver.AddReshape();
    // micro_op_resolver.AddRelu6();

    // Perf-Mobilenet
    static tflite::MicroMutableOpResolver<7> micro_op_resolver;
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddRelu6();
    
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        std::cout << "AllocateTensors() failed\n";
        return;
    }

    input = interpreter->input(0);
}

void run(bool only_forward) {
    float temp_time = 0.f;
    if (!only_forward) {
        for (unsigned char e = 0; e < localEpoch; ++e) {
            time_saved = esp_timer_get_time();
            cModel->forward(inputDatas);
            temp_time += (esp_timer_get_time() - time_saved) / 1000;
            time_saved = esp_timer_get_time();
            cModel->backward(inputDatas, groundTruth);
            time_backward += (esp_timer_get_time() - time_saved) / 1000;
            // std::cout << static_cast<int>(e) << '\n';
            // for (unsigned char batch = 0; batch < batchSize; ++batch) {
            //     for (unsigned int node = 0; node < OutputNode; ++node) {
            //         std::cout << cModel->layers[LayerLength - 1].outputData[batch * OutputNode + node] << ' ' << static_cast<int>(groundTruth[batch * OutputNode + node]) << ' ';
            //     }
            //     std::cout << "\n";
            // }
        }
        ++round_num;
    } else {
        cModel->forward(inputDatas);
    }
    time_forward = time_forward / batchSize + temp_time / localEpoch / batchSize;

    std::cout << "graph\n";
    sendFloat(time_forward);
    sendFloat(time_backward);
    sendFloat(cModel->accuracy(groundTruth));
    sendFloat(cModel->cross_entropy_loss(groundTruth));
    return;
}

float readFloat32() {
    unsigned char res[4];
    for (unsigned char n = 0; n < sizeof(float); ++n) {
        readByte(&res[n]);
    }
    return *(float *)&res;
}

void readByte(unsigned char *temp)
{
    unsigned char check;
    do {
        check = uart_read_bytes(EX_UART_NUM, temp, 1, 20 / portTICK_PERIOD_MS);
    } while (check <= 0);
    return;
}

void sendFloat(float arg)
{
    unsigned char *bytePointer = reinterpret_cast<unsigned char*>(&arg);

    for (unsigned char i = 0; i < sizeof(float); ++i) {
        uart_write_bytes(EX_UART_NUM, bytePointer + i, 1);
    }

    return;
}