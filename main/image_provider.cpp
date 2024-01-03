/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "image_provider.h"

static uint8_t IMG_SIZE = 96 * 96 * 3;

// Get the camera module ready
TfLiteStatus InitCamera() {
  MicroPrintf("Attempting to start Camera", "");
  
  // initialize the camera OV2640
  static camera_config_t config = {
    .pin_pwdn = 32,
    .pin_reset = -1,
    .pin_xclk = 0,
    .pin_sscb_sda = 26,
    .pin_sscb_scl = 27,

    .pin_d7 = 35,
    .pin_d6 = 34,
    .pin_d5 = 39,
    .pin_d4 = 36,
    .pin_d3 = 21,
    .pin_d2 = 19,
    .pin_d1 = 18,
    .pin_d0 = 5,
    .pin_vsync = 25,
    .pin_href = 23,
    .pin_pclk = 22,

    // XCLK 20MHz or 10MHz for OV2640 float_t FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG, // YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_QQVGA,   // QQVGA-UXGA Do not use sizes above QVGA when not JPEG

    .jpeg_quality = 10, // 0-63 lower number means higher quality
    .fb_count = 2,      // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
  };
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
      MicroPrintf("Camera init failed with error 0x%x\n", err);
      return kTfLiteError;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_gain_ctrl(s, 1);     // auto gain on
  s->set_exposure_ctrl(s, 1); // auto exposure on
  s->set_awb_gain(s, 1);      // Auto White Balance enable (0 or 1)
  s->set_brightness(s, 1);    // up the brightness just a bit
  s->set_contrast(s, 1);      // -2 to 2
  s->set_saturation(s, -1);   // lower the saturation

  delay(1000);

  return kTfLiteOk;
}

// Decode the JPEG image, crop it, and convert it to greyscale
TfLiteStatus DecodeAndProcessImage(int image_width, int image_height, int8_t* image_data) {
  MicroPrintf("Decoding JPEG and converting to 888rgb");

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
      MicroPrintf("Camera capture failed\n");
      return kTfLiteError;
  }

  JpegDec.decodeArray(fb->buf, fb->len);

  // Crop the image by keeping a certain number of MCUs in each dimension
  const int keep_x_mcus = image_width / JpegDec.MCUWidth;
  const int keep_y_mcus = image_height / JpegDec.MCUHeight;

  // Calculate how many MCUs we will throw away on the x axis
  const int skip_x_mcus = JpegDec.MCUSPerRow - keep_x_mcus;
  // Roughly center the crop by skipping half the throwaway MCUs at the
  // beginning of each row
  const int skip_start_x_mcus = skip_x_mcus / 2;
  // Index where we will start throwing away MCUs after the data
  const int skip_end_x_mcu_index = skip_start_x_mcus + keep_x_mcus;
  // Same approach for the columns
  const int skip_y_mcus = JpegDec.MCUSPerCol - keep_y_mcus;
  const int skip_start_y_mcus = skip_y_mcus / 2;
  const int skip_end_y_mcu_index = skip_start_y_mcus + keep_y_mcus;

  // Pointer to the current pixel
  uint16_t* pImg;
  // Color of the current pixel
  uint16_t color;

  // Loop over the MCUs
  while (JpegDec.read()) {
    // Skip over the initial set of rows
    if (JpegDec.MCUy < skip_start_y_mcus) {
      continue;
    }
    // Skip if we're on a column that we don't want
    if (JpegDec.MCUx < skip_start_x_mcus ||
        JpegDec.MCUx >= skip_end_x_mcu_index) {
      continue;
    }
    // Skip if we've got all the rows we want
    if (JpegDec.MCUy >= skip_end_y_mcu_index) {
      continue;
    }
    // Pointer to the current pixel
    pImg = JpegDec.pImage;

    // The x and y indexes of the current MCU, ignoring the MCUs we skip
    int relative_mcu_x = JpegDec.MCUx - skip_start_x_mcus;
    int relative_mcu_y = JpegDec.MCUy - skip_start_y_mcus;

    // The coordinates of the top left of this MCU when applied to the output
    // image
    int x_origin = relative_mcu_x * JpegDec.MCUWidth;
    int y_origin = relative_mcu_y * JpegDec.MCUHeight;

    for(int rgb_color = 0; rgb_color < 3; rgb_color++){
      // Loop through the MCU's rows and columns
      for (int mcu_row = 0; mcu_row < JpegDec.MCUHeight; mcu_row++) {
        // The y coordinate of this pixel in the output index
        int current_y = y_origin + mcu_row;
        for (int mcu_col = 0; mcu_col < JpegDec.MCUWidth; mcu_col++) {
          // Read the color of the pixel as 16-bit integer
          color = *pImg++;
          // Extract the color values (5 red bits, 6 green, 5 blue)
          uint8_t r, g, b;
          r = ((color & 0xF800) >> 11) * 8;
          g = ((color & 0x07E0) >> 5) * 4;
          b = ((color & 0x001F) >> 0) * 8;
          // Convert to grayscale by calculating luminance
          // See https://en.wikipedia.org/wiki/Grayscale for magic numbers
          //float gray_value = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
          float gray_value = 0;
          if(rgb_color == 0){
            gray_value = r;
          }
          else if(rgb_color == 1){
            gray_value = g;
          }
          else{
            gray_value = b;
          }

          // Convert to signed 8-bit integer by subtracting 128.
          gray_value -= 128;

          // The x coordinate of this pixel in the output image
          int current_x = x_origin + mcu_col;
          // The index of this pixel in our flat output buffer
          int index = ((current_y * image_width) + current_x)*rgb_color;
          image_data[index] = static_cast<int8_t>(gray_value); 
        }
      }
    }
  }
  esp_camera_fb_return(fb);
  MicroPrintf("Image decoded and processed");
  return kTfLiteOk;
}

// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, int8_t* image_data) {
  TfLiteStatus decode_status = DecodeAndProcessImage(image_width, image_height, image_data);
  
  if (decode_status != kTfLiteOk) {
    MicroPrintf("DecodeAndProcessImage failed");
    return decode_status;
  }

  return kTfLiteOk;
}