# python3 -m venv my_env
# source my_env/bin/activate
# sudo apt-get install python3-tk
# pip3 install numpy matplotlib pyserial

import numpy as np
import tensorflow as tf

import serial
from serial.tools.list_ports import comports

import struct
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
# import json
import random
from PIL import Image
import pickle
import signal
import sys
import datetime
from math import sqrt

random.seed(4321)
np.random.seed(4321)

model_type = 'perf-mobilenet'
baudrate = 460800
samples_per_device = 5000
batch_size = 20
# size_input_nodes = 1280 # tf-mobilenet
size_input_nodes = 256 # perf-mobilenet
layers_size_nodes = [64, 2]
# layers_size_nodes = [2]
layers_weight = []
learningRate = 0.1
lambda_value = 0.001
local_epoch = 5

trainingDatas = []
trainingLabels = []
testDatas = []
testLabels = []

fileDataFile = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
# fileDataFile = ['data_batch_1']
testDataFile = 'test_batch'
colors = ['g', 'b', 'y', 'r']
markers = ['--', ':', '-.', '-']

saved_history = {
    'round': 0,
    'device': []
}

#____________________________________ SIGNAL HANDLE ___________________________
def save_data_and_exit(signal, frame):
    print('Terminattion\'s signal! Saving result!')
    round = range(0, saved_history['round'])
    if round == 0: sys.exit()
    
    plt.figure()  # Create a new figure
    
    for idx in range(len(devices)):
        plt.plot(round, saved_history['device'][idx]['acc'], colors[idx] + markers[idx], label=f'Device {idx}')
    plt.plot(round, saved_history['device'][len(devices)]['acc'], colors[-1] + markers[-1], label=f'Server')

    plt.title('Validation Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    # plt.xlim(left=0)
    # plt.ylim(0.0, 1.0)
    plt.legend(loc=0)
    plt.savefig(f"result/Acc/{datetime.datetime.now()}-MODEL{model_type}-BS{batch_size}-R{saved_history['round']}-LE{local_epoch}.png", format='png')
    
    plt.figure()  # Create a new figure
    for idx in range(len(devices)):
        plt.plot(round, saved_history['device'][idx]['loss'], colors[idx] + markers[idx], label=f'Device {idx}')
    plt.plot(round, saved_history['device'][len(devices)]['loss'], colors[-1] + markers[-1], label=f'Server')

    plt.title('Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    # plt.xlim(left=0)
    # plt.ylim(bottom=0)
    plt.legend(loc=0)
    plt.savefig(f"result/Loss/{datetime.datetime.now()}-MODEL{model_type}-BS{batch_size}-R{saved_history['round']}-LE{local_epoch}.png", format='png')

    with open(f"result/History/{datetime.datetime.now()}-MODEL{model_type}-BS{batch_size}-R{saved_history['round']}-training_history.txt", 'w') as f:
        f.write(str(saved_history))

    sys.exit(0)

signal.signal(signal.SIGTERM, save_data_and_exit)
signal.signal(signal.SIGINT, save_data_and_exit)

#____________________________________ FUNCTION ___________________________
def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        # print(msg[:-2])
        if msg[:-2] == keyword:
            break
        else:
            print(f'({arduino.port}):', msg, end='')

def init_network(device):
    device.reset_input_buffer()
    device.write(b's')
    print_until_keyword('start', device)
    print(f"Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', lambda_value))
    device.write(struct.pack('b', local_epoch))
    device.write(struct.pack('b', batch_size))

    device.write([item for layer in layers_weight for num in layer for item in struct.pack('f', num)])

    print_until_keyword('Received new model.', device)

def sendBatch(device, data, start, forward):
    ini_time = time.time() * 1000
    img, target = data
    device.write(b'r')
    device.write(struct.pack('b', forward))
    device.readline().decode() # Backward conf
    # print(f"[{device.port}] Only forward confirmation: {device.readline().decode()}", end='') # Label confirmation

    for i in range(start, start + batch_size):
        print(f"[{device.port}] Sending image {i + 1}")

        # device.readline().decode()
        print(f"[{device.port}] Train start confirmation: {device.readline().decode()}", end='')
        # print_until_keyword('ok', device)

        device.write(struct.pack('b', target[i]))
        device.readline().decode() # Label confirmation
        # print(f"[{device.port}] Label received confirmation: ", device.readline().decode(), end='')

        device.write(img[i])
        # print(len(img), img)
        # device.readline().decode()

    # Accept 'graph' command
    print_until_keyword('graph', device)

    [time_forward] = struct.unpack('f', device.read(4))
    [time_backward] = struct.unpack('f', device.read(4))
    [accuracy] = struct.unpack('f', device.read(4))
    [error] = struct.unpack('f', device.read(4))
    print(f"[{device.port}] The time to forward an image is {time_forward}ms and the backward of all batchs and epochs is {time_backward}ms")
    print(f"[{device.port}] Batch generated an error of {error} and accuracy of {accuracy}")
    print(f'[{device.port}] Batch done in: {(time.time() * 1000) - ini_time} milliseconds)')
    return accuracy, error

def sendSamples(device, saved, data, start, vali):
    val_acc, val_err = 0, 0
    step = samples_per_device*0.2 / batch_size
    if vali == 0: sendBatch(device, data, start, 0)
    # for batch in range(int(samples_per_device*0.2/batch_size)):
    #     accuracy, error = sendBatch(device, data, int(samples_per_device*0.8) + batch * batch_size, 1)
    #     val_acc += accuracy / step
    #     val_err += error / step
    # saved_history['device'][saved]['acc'].append(val_acc)
    # saved_history['device'][saved]['loss'].append(val_err)


def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except:
            print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            dev =  serial.Serial(None, baudrate)
            dev.port = port
            dev.rts = False
            dev.dtr = False
            # dev.rts = True
            # dev.dtr = True
            dev.open()
            time.sleep(1)
            dev.rts = False
            dev.dtr = False
            return dev
        except:
            print(f"ERROR: Wrong port connection ({port})")

def getDevices():
    global devices
    num_devices = read_number("Number of devices: ")

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]

def FlGetModel(d, device_index, devices_layers_weight, devices_round_num):
    print(f'Starting connection to {d.port} ...') # Hanshake
    d.write(b'>') # Python --> SYN --> Arduino
    if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
        d.write(b'f') # Python --> ACK --> Arduino
        
        print('Connection accepted.')
        d.timeout = None

        print_until_keyword('start', d)
        devices_round_num.append(int(d.readline()[:-2]))

        print(f'Receiving model from {d.port} ...')
        ini_time = time.time()

        for layer in range(len(layers_size_nodes)): # output layer
            for i in range(layers_weight[layer].size):
                [devices_layers_weight[layer][device_index][i]] = struct.unpack('f', d.read(4))

        print(f'Model received from {d.port} ({time.time() - ini_time} seconds)')

    else:
        print(f'Connection timed out. Skipping {d.port}.')

def sendModel(d):
    ini_time = time.time()

    d.write([item for layer in layers_weight for num in layer for item in struct.pack('f', num)])

    print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')
    print_until_keyword('done fl', d)

def startFL():
    print('Starting Federated Learning')
    devices_layers_weight = [np.empty((len(devices), layers_weight[layer].size), dtype='float32') for layer in range(len(layers_size_nodes))]
    devices_round_num = []

    # Receiving models
    threads = []
    for i, d in enumerate(devices):
        thread = threading.Thread(target=FlGetModel, args=(d, i, devices_layers_weight, devices_round_num))
        thread.daemon = True
        thread.start()
        threads.append(thread)
 
    for thread in threads: thread.join() # Wait for all the threads to end

    if sum(devices_round_num) > 0:
        ini_time = time.time() * 1000
        for layer in range(len(layers_size_nodes)):
            layers_weight[layer] = np.average(devices_layers_weight[layer], axis=0, weights=devices_round_num)
        print(f'Average time: {(time.time()*1000)-ini_time} milliseconds)')

    # Test for each device in server
    for deviceIdx in range(len(devices)):
        (origin_datas, labels) = iid_datasets[deviceIndex]
        origin_datas = origin_datas[int(samples_per_device*0.8):]
        labels = labels[int(samples_per_device*0.8):]
        features_datas = []

        labels = [np.array([0.0, 1.0]) if label == 1 else np.array([1.0, 0.0]) for label in labels]
        # testLabels.extend(dict[b'labels'])
        for data in origin_datas:
            interpreter.set_tensor(input_details[0]['index'], data.reshape(1, 96, 96, 3))
            interpreter.invoke()
            features_datas.append(interpreter.get_tensor(output_details[0]['index']).flatten())
        
        input_nodes = size_input_nodes
        for idx, devices_layer_weight in enumerate(devices_layers_weight):
            model.layers[idx].set_weights([
                devices_layer_weight[deviceIdx][layers_size_nodes[idx]:].reshape(input_nodes, layers_size_nodes[idx]),
                devices_layer_weight[deviceIdx][:layers_size_nodes[idx]],
            ])
            input_nodes = layers_size_nodes[idx]
        results = model.evaluate(np.array(features_datas), np.array(labels))
        print(results)
        saved_history['device'][deviceIdx]['acc'].append(results[1])
        saved_history['device'][deviceIdx]['loss'].append(results[0])

    # Sending models
    threads = []
    for d in devices:
        print(f'Sending model to {d.port} ...')
        thread = threading.Thread(target=sendModel, args=(d, ))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end


#____________________________________ START ___________________________
input_nodes = size_input_nodes
for nodes in layers_size_nodes:
    limit = sqrt(6 / (input_nodes + nodes))
    weight = np.random.uniform(-limit, limit, (input_nodes + 1) * nodes).astype('float32')
    layers_weight.append(weight)
    input_nodes = nodes

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_data.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(layers_size_nodes[0], input_shape=(size_input_nodes,), activation='sigmoid'),
    tf.keras.layers.Dense(layers_size_nodes[1], activation='softmax'),
    # tf.keras.layers.Dense(layers_size_nodes[0], input_shape=(size_input_nodes,), activation='softmax'),
])
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#____________________________________ GET DATASET ___________________________
for file in fileDataFile:
    with open('datasets/cifar-10-python/cifar-10-batches-py/{file}'.format(file=file), 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

        trainingLabels.extend([1 if label == 3 else 0 for label in dict[b'labels']])
        # trainingLabels.extend(dict[b'labels'])
        for imgBuf in dict[b'data']:
            # print(type(imgBuf), imgBuf.shape)
            imgBuf = np.reshape(imgBuf, (3,32,32)).transpose((1,2,0))
            image = Image.fromarray(imgBuf)
            resized_image_array = np.array(image.resize((96, 96))).flatten()
            trainingDatas.append(resized_image_array)

with open(f'datasets/cifar-10-python/cifar-10-batches-py/{testDataFile}', 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

        testLabels.extend([1 if label == 3 else 0 for label in dict[b'labels']])
        # testLabels.extend(dict[b'labels'])
        for idx, imgBuf in enumerate(dict[b'data']):
            # print(type(imgBuf), imgBuf.shape)
            imgBuf = np.reshape(imgBuf, (3,32,32)).transpose((1,2,0))
            image = Image.fromarray(imgBuf)
            resized_image_array = np.array(image.resize((96, 96))).reshape(1, 96, 96, 3)
            interpreter.set_tensor(input_details[0]['index'], resized_image_array)
            interpreter.invoke()
            testDatas.append(interpreter.get_tensor(output_details[0]['index']).flatten())
            # testDatasOfTuple.append((resized_image_array, dict[b'labels'][idx]))

combined_lists = list(zip(trainingLabels, trainingDatas))
sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0], reverse=True)

cat_size = 0
for index, tup in enumerate(sorted_combined_lists):
    if tup[0] == 0:
        cat_size = index
        break
print(f'Cat\'s datas: {cat_size}')

combined_list = [item for pair in zip(sorted_combined_lists[: cat_size], sorted_combined_lists[cat_size : 2 * cat_size]) for item in pair]
trainingLabels, trainingDatas = zip(*combined_list)

combined_lists = list(zip(testLabels, testDatas))
sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0], reverse=True)

cat_size = 0
for index, tup in enumerate(sorted_combined_lists):
    if tup[0] == 0:
        cat_size = index
        break
print(f'Cat\'s datas: {cat_size}')

combined_list = [item for pair in zip(sorted_combined_lists[: cat_size], sorted_combined_lists[cat_size : 2 * cat_size]) for item in pair]
testLabels, testDatas = zip(*combined_list)
testLabels = [np.array([0.0, 1.0]) if label == 1 else np.array([1.0, 0.0]) for label in testLabels]

getDevices()

for _ in range(len(devices) + 1):
    saved_history['device'].extend([{
        'loss': [],
        'acc': [],
    }])

#____________________________________IID DATASET___________________________
iid_datasets = []
for deviceIdx in range(len(devices)):
    iid_datasets.append((trainingDatas[deviceIdx * samples_per_device: (deviceIdx + 1) * samples_per_device], trainingLabels[deviceIdx * samples_per_device: (deviceIdx + 1) * samples_per_device]))

#____________________________________NON-IID DATASET___________________________
# unique_labels = np.array(list(set(label for label in trainingLabels)))
# random.shuffle(unique_labels)
# non_iid_partitions = np.split(unique_labels, len(devices))

# data_dict = defaultdict(lambda :[])
# for idx, label in enumerate(trainingLabels):
#     data_dict[label].append(trainingDatas[idx])

# non_iid_dataset = []
# for indices in non_iid_partitions:
#     indices = indices.flatten()
#     temp = [(item, label) for label in indices for item in data_dict[label]]
#     print(len(temp))
#     np.random.shuffle(temp)
#     non_iid_dataset.append(temp)

#____________________________________INIT___________________________

threads = []
for device in devices:
    thread = threading.Thread(target=init_network, args=(device, ))
    thread.daemon = True
    thread.start()
    threads.append(thread)
for thread in threads: thread.join() # Wait for all the threads to end

#____________________________________TRAINNING -> FL -> TESTING___________________________
train_ini_time = time.time()
for batch in range(int(samples_per_device*0.8/batch_size)):
    print("----------------------------------------------------------------------------------")
    print(f'<Round>: {batch + 1}/{int(samples_per_device*0.8/batch_size)}!')
    batch_ini_time = time.time()
    for deviceIndex, device in enumerate(devices):
        thread = threading.Thread(target=sendSamples, args=(device, deviceIndex, iid_datasets[deviceIndex], batch * batch_size, 0))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end
    print(f'Round time: {time.time() - batch_ini_time} seconds)')
    # Federated Learning
    fl_ini_time = time.time()
    startFL()
    print(f'FL time: {time.time() - fl_ini_time} seconds)')
    # Testing the model after FL
    print(f'<Testing>!')
    # sendSamples(devices[0], len(devices), testDatasOfTuple, batch * batch_size, 1)
    input_nodes = size_input_nodes
    for idx, layer_weight in enumerate(layers_weight):
        model.layers[idx].set_weights([
            layer_weight[layers_size_nodes[idx]:].reshape(input_nodes, layers_size_nodes[idx]),
            layer_weight[:layers_size_nodes[idx]],
        ])
        input_nodes = layers_size_nodes[idx]
    results = model.evaluate(np.array(testDatas), np.array(testLabels))
    print(results)
    saved_history['device'][len(devices)]['acc'].append(results[1])
    saved_history['device'][len(devices)]['loss'].append(results[0])
    saved_history['round'] = batch + 1

train_time = time.time()-train_ini_time
# np.savetxt(f"result/Weight/{datetime.datetime.now()}-MODEL{model_type}-BS{batch_size}-R{saved_history['round']}.txt", layers_weight, fmt='%f')

signal.raise_signal(signal.SIGTERM)