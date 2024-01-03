#include "main_function.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void tf_main(void) {
  setup();
  while (true) {
    loop();
  }
}

extern "C" void app_main(void)
{
  xTaskCreate((TaskFunction_t)&tf_main, "tf_main", 4 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}
