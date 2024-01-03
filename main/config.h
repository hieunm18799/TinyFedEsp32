#ifndef CONFIG
#define CONFIG
// #define ESP_NN true

static const unsigned char ImgWidth = 96;
static const unsigned char ImgHeight = 96;
static const unsigned char ImgChannel = 3;

static const unsigned int InputNode = ImgWidth * ImgHeight * ImgChannel;
static const unsigned char LayerLength = 2;
// static const unsigned int FeatureNodes = 1280; //Mobilenet v2 output
static const unsigned int FeatureNodes = 256; //perf mobilenet output
static const unsigned char HiddenNode1 = 64;
static const unsigned char OutputNode = 2; // Cat vs no car
// static const unsigned char OutputNode = 10;    // Cifar-10 problem
// static const unsigned int Nodes[] = {FeatureNodes, OutputNode};
static const unsigned int Nodes[] = {FeatureNodes, HiddenNode1, OutputNode};

static const unsigned long int BAUDRATE = 460800;

#endif