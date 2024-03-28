# TinyFedEsp32

This project is learned from an existed paper [TinyFedTL](https://arxiv.org/abs/2110.01107). Project use ESP32-CAMs (instead of Arduino) so that we can upload much more memory to tiny devices. Moreover, the esp-idf provided a strong DL library esp-dl ([link](https://docs.espressif.com/projects/esp-dl/en/latest/esp32s3/introduction.html)) help the process of DL faster when use ESP32.

- Federated learning is implemented using 1 server and multiple (2 in test) client as ESP32 
- The server preprocesses datas, gives the mini-batch data to the ESP32s, aggregates the models and update model to all clients
- The client - ESP32 don't have SD card, use pre-trained model with some output layer, get model's parameters and datas to train the model

## Dependencies

1. Server:
- tensorflow: ML's library for run test on server per round (after update model)
- python3-tk: use matplotlib.pyplot to drawing the result of FL
- threading: need to read, write datas, parameters to multiple ESP32s
- pyserial: connect, communicate with ESP32s throught cable
- other: numpy, pickle ...

2. Client:
- esp-nn: the main-used library inside esp-dl
- esp-tflite-micro: for using pre-trained model

## Installation

1. Server
````
sudo apt-get install python3-tk
pip3 install numpy pyserial pickle
````

2. Client

For the client, i follow the instruction of esp-idf, not using arduino ide, because it's the only ways to use esp-nn and esp-tflite-micro. The config of esp-idf is store in the [partition.csv](https://github.com/hieunm18799/TinyFedEsp32/blob/master/partitions.csv) and [sdkconfig](https://github.com/hieunm18799/TinyFedEsp32/blob/master/sdkconfig), the re-main thing todo is upload code to the ESP32-CAM throught serial port connected.

## Development

After the code has been uploaded to ESP32s, devices will run and wait signal from server if not crash or have some issues. Run the server's python file [fl_server.py](https://github.com/hieunm18799/TinyFedEsp32/blob/master/server/fl_server.py), wait server for read datas and type the number of devices and its port the connected to.
