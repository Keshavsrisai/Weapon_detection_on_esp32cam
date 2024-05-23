#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "weaponDmodel.h"
#include "model_settings.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <esp_heap_caps.h>

#include "esp_camera.h"

#include "img_converters.h"

#include "soc/soc.h" // Disable brownout problems
#include "soc/rtc_cntl_reg.h" // Disable brownout problems

// Select camera model
// #define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE  // Has PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM

#include "camera_pins.h"
#include "downsample.h"

unsigned long tm_to_disp;

// #include <fb_gfx.h>
const char* kCategoryLabels[kCategoryCount] = {
    "Knife", "Gun", "Neg"
};




#define TEXT "starting app..."

camera_fb_t * fb = NULL;
uint16_t *buffer;

size_t _jpg_buf_len = 0;
uint8_t * _jpg_buf = NULL;


//tflite ptrs
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 39 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize; //81 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;


void init_camera(){
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;//PIXFORMAT_GRAYSCALE;//PIXFORMAT_JPEG;//PIXFORMAT_RGB565;// 
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.frame_size = FRAMESIZE_96X96;//FRAMESIZE_QVGA;//FRAMESIZE_96X96;//
  
  
  if(psramFound()){
    
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    
    config.jpeg_quality = 12;
    config.fb_count = 2;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    delay(1000);
    ESP.restart();
  }
#if defined(CAMERA_MODEL_M5STACK_WIDE)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif
}


void setup() {
  
  // put your setup code here, to run once:
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);//disable brownout detector

  Serial.begin(115200);
  init_camera();
  
  dstImage = (uint16_t *) malloc(DST_WIDTH * DST_HEIGHT*2);
  img192x192 = (uint16_t *) malloc(192*192*2);
  delay(200);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(WD5000_lite_save_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  if (tensor_arena == NULL) {
    tensor_arena =  (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }
  // tflite::AllOpsResolver resolver;
  /*
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
      //
      */
  // NOLINTNEXTLINE(runtime-global-variables)
  //
  
  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  // micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  
  
  

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  
      ////
  
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  pinMode(4, OUTPUT);


}

void loop() {
  for (int i = 0; i<1; i++){
  fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
    }
    if(fb){
      esp_camera_fb_return(fb);
      fb=NULL;
    }
  delay(1);
  }
  tm_to_disp=millis();
  while(millis()-tm_to_disp<5000){ 

  Serial.println("in display");
  fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
    }
  uint16_t * tmp = (uint16_t *) fb->buf;

    downsampleImage((uint16_t *) fb->buf, fb->width, fb->height);
    bool jpeg_converted = frame2jpg(fb, 90, &_jpg_buf, &_jpg_buf_len);

    for (int y = 0; y < DST_HEIGHT; y++) {
      for (int x = 0; x < DST_WIDTH; x++) {
        tmp[y*(fb->width) + x] = (uint16_t) dstImage[y*DST_WIDTH +x];

      }
    }
    upsample((uint16_t *) fb->buf);
    delay(15);
  

     
    if(fb){
      esp_camera_fb_return(fb);
      if(_jpg_buf){
      free(_jpg_buf);}
      fb = NULL;
      _jpg_buf = NULL;
    }
    delay(15);
  }
    int8_t * image_data = input->data.int8;

    for (int i = 0; i < kNumRows; i++) {
      for (int j = 0; j < kNumCols; j++) {
        uint16_t pixel = ((uint16_t *) (dstImage))[i * kNumCols + j];

        // for inference
        uint8_t hb = pixel & 0xFF;
        uint8_t lb = pixel >> 8;
        uint8_t r = (lb & 0x1F) << 3;
        uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
        uint8_t b = (hb & 0xF8);

        int8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;

        // int8_t grey_pixel = (int8_t) ((int) tst_img1[i * kNumCols + j]-128);

        image_data[i * kNumCols + j] = grey_pixel;
      }
    }

    

  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  

  TfLiteTensor* output = interpreter->output(0);
  
  int idx = 0;
  int8_t max_confidence = output->data.uint8[idx];
  int8_t cur_confidence;
  float max_tmp = -10000.0;
  for(int i = 0; i < kCategoryCount; i++){
    float tmp=output->data.f[i];
    cur_confidence = output->data.uint8[i];
    
    if(max_confidence < cur_confidence){
      idx = i;
      max_confidence = cur_confidence;
    }
    if (tmp > max_tmp){
      max_tmp = tmp;
    }
  }
  Serial.println(idx);
  Serial.println(max_confidence);
  String detected= String(kCategoryLabels[idx]);
  Serial.println(detected);
  
  delay(2000);

}
