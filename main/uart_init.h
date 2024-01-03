#ifndef UART_INIT
#define UART_INIT

#include <stdio.h>
#include <string.h>
#include "esp_system.h"
#include "esp_console.h"
#include "esp_vfs_dev.h"
#include "esp_vfs_fat.h"
#include "driver/uart.h"
#include "freertos/queue.h"

#include "config.h"

#define EX_UART_NUM UART_NUM_0
#define BUF_SIZE (1024)
extern "C" {
void uart_init();
}

#endif