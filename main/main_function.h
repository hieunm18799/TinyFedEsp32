#ifndef MAIN_FUNCTION
#define MAIN_FUNCTION

/* Includes ---------------------------------------------------------------- */
#include <esp_timer.h>
#include <esp_log.h>
#include <iostream>


#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "config.h"
#include "c_model.h"
#include "model_data.h"
#include "uart_init.h"
// #include "image_provider.h"

extern "C" {
void setup();
void loop();
}
#endif