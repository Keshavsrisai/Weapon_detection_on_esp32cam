#ifndef _DOWN_SAMPLE_H
#define _DOWN_SAMPLE_H

#define DST_WIDTH 28          
#define DST_HEIGHT 28         

#define DST2_WIDTH 32         
#define DST2_HEIGHT 32         
uint16_t *img192x192;
uint16_t *dstImage; //[DST_WIDTH * DST_HEIGHT];   // Destination gray level image data
void downsampleImage(uint16_t *srcImage, uint16_t SRC_WIDTH, uint16_t SRC_HEIGHT);
void upsample(uint16_t *srcImage);
void averageResampleImage(uint16_t *srcImage, uint16_t SRC_WIDTH, uint16_t SRC_HEIGHT);

void downsampleImage(uint16_t *srcImage, uint16_t SRC_WIDTH, uint16_t SRC_HEIGHT) {
  // float scaleX = (float)SRC_WIDTH / DST_WIDTH;    // Calculate the X scaling factor
  float scaleY = (float)SRC_HEIGHT / DST_HEIGHT;  // Calculate the Y scaling factor
  float scaleX = scaleY;
  for (int y = 0; y < DST_HEIGHT; y++) {
    for (int x = 0; x < DST_WIDTH; x++) {
      int srcX = x * scaleX;//+40;
      int srcY = y * scaleY;

      uint16_t pixel = srcImage[srcY * SRC_WIDTH + srcX];

      dstImage[y * DST_WIDTH + x] = pixel;
    }
  }
}

void downsampleImage2(uint16_t *srcImage, uint16_t SRC_WIDTH, uint16_t SRC_HEIGHT) {
  // float scaleX = (float)SRC_WIDTH / DST_WIDTH;    // Calculate the X scaling factor
  float scaleY = (float)SRC_HEIGHT / DST2_HEIGHT;  // Calculate the Y scaling factor
  float scaleX = scaleY;
  for (int y = 0; y < DST2_HEIGHT; y++) {
    for (int x = 0; x < DST2_WIDTH; x++) {
      int srcX = x * scaleX;//+40;
      int srcY = y * scaleY;
      uint16_t pixel = srcImage[srcY * SRC_WIDTH + srcX];

      dstImage[y * DST_WIDTH + x] = pixel;
    }
  }
}

void upsample(uint16_t *srcImage) {
  
  
  for (int y = 0; y < 96; y++) {
    for (int x = 0; x < 96; x++) {
      uint16_t pixel = ((uint16_t *) srcImage)[y*96+x];
      img192x192[2*y*96*2+2*x] = pixel;
        img192x192[2*y*96*2+2*x+1] = pixel;
        img192x192[(2*y+1)*96*2+2*x] = pixel;
        img192x192[(2*y+1)*96*2+2*x+1] = pixel;
      
    }
  }
}

void averageResampleImage(uint16_t *srcImage, uint16_t SRC_WIDTH, uint16_t SRC_HEIGHT) {
  float scaleX = (float)SRC_WIDTH / DST_WIDTH;
  float scaleY = (float)SRC_HEIGHT / DST_HEIGHT;

  for (int y = 0; y < DST_HEIGHT; y++) {
    for (int x = 0; x < DST_WIDTH; x++) {
      // Calculate the corresponding position in the source image
      int srcXStart = scaleX * x;
      int srcYStart = scaleY * y;

      // Calculate the sum of pixel values in the source area
      int pixelSum = 0;
      for (int sy = 0; sy < scaleY; sy++) {
        for (int sx = 0; sx < scaleX; sx++) {
          pixelSum += srcImage[(srcYStart + sy) * SRC_WIDTH + (srcXStart + sx)];
        }
      }

      uint16_t averagePixelValue = pixelSum / (scaleX * scaleY);
      dstImage[y * DST_WIDTH + x] = averagePixelValue;
    }
  }
}

#endif
