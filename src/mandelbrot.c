#define RAYGUI_IMPLEMENTATION
/*#include "mandelbrot.cu"*/
#include "raygui.h"
#include "raylib.h"
#include <complex.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_ITERATIONS 300
#define PALETTE_SIZE 5
#define NUM_THREADS 64
#define SCREENWIDTH 1920
#define SCREENHEIGHT 1080
#define GPU 1

float exponent = 2.0;
typedef struct {
  int start_row;
  int end_row;
  int width;
  int height;
  double (*complex_plane)[SCREENHEIGHT][2];
  int (*grid)[SCREENHEIGHT];
  int max_iterations;
} ThreadData;

int iterate_mandelbrot(double complex start_c) {
  float bound = 2;
  double complex z = start_c;
  int iter;

  // main cardiod checking
  if ((exponent == 2.0 && pow(creall(z) + 1.0, 2) + pow(cimagl(z), 2)) <
      1.0 / 16.0)
    return MAX_ITERATIONS;
  for (iter = 0; (iter < MAX_ITERATIONS); iter++) {
    z = cpow(z, exponent) + start_c;
    if (cabs(z) > 2.0) {
      return iter;
    }
  };
  return MAX_ITERATIONS;
}

void *parallelIterations(void *args) {
  ThreadData *data = (ThreadData *)args;
  for (int i = data->start_row; i < data->end_row; i++) {
    for (int j = 0; j < data->height; j++) {
      double complex c = (*data->complex_plane)[i * data->height + j][0] +
                         (*data->complex_plane)[i * data->height + j][1] * I;
      int result = iterate_mandelbrot(c);
      (*data->grid)[i * data->height + j] = result;
    }
  }
  return NULL;
}

// Define the color palette
Color palette[PALETTE_SIZE] = {
    (Color){139, 191, 159, 255}, // green
    (Color){131, 188, 255, 255}, // blue
    (Color){18, 69, 89, 255},    // midnight
    (Color){89, 52, 79, 255},    // violet
    (Color){238, 32, 77, 255}    // crayola
};

Color getColor(int iterations) {
  if (iterations == MAX_ITERATIONS) {
    return palette[2]; // midnight for points inside the set
  }

  // Split the remaining colors into 4 sections
  int section = iterations / 25;
  Color color;

  switch (section) {
  case 0:
    color = palette[0]; // green
    break;
  case 1:
    color = palette[1]; // blue
    break;
  case 2:
    color = palette[3]; // violet
    break;
  case 3:
    color = palette[4]; // crayola
    break;
  };
  float factor = (float)(iterations % 25) / 25.0f;
  return ColorBrightness(color, factor - 0.5f);
}

void pixelsToCoords(int x, int y, float x_scale, float y_scale, float x_start,
                    float y_start, double dest[x][y][2]) {
  float x_width = 4.0 * x_scale;
  float y_width = 4.0 * y_scale;
  double step_x = x_width / x;
  double step_y = y_width / y;
  double x_coord = x_start, y_coord = y_start;

  for (int i = 0; i < x; i++) {
    y_coord = y_start;

    for (int j = 0; j < y; j++) {
      dest[i][j][0] = x_coord;
      dest[i][j][1] = y_coord;

      y_coord = y_coord + step_y;
    };
    x_coord = x_coord + step_x;
  };
}

void iterateOverGrid(int width, int height,
                     double (*complex_plane)[SCREENHEIGHT][2],
                     int (*grid)[SCREENHEIGHT], int max_iterations) {
  pthread_t threads[NUM_THREADS];
  ThreadData thread_data[NUM_THREADS];

  int rows_per_thread = width / NUM_THREADS;
  int extra_rows = width % NUM_THREADS;
  int start_row = 0;
  for (int i = 0; i < NUM_THREADS; i++) {
    thread_data[i].start_row = start_row;
    // bit magic??
    thread_data[i].end_row =
        start_row + rows_per_thread + (i < extra_rows ? 1 : 0);
    thread_data[i].width = width;
    thread_data[i].height = height;
    thread_data[i].complex_plane = complex_plane;
    thread_data[i].grid = grid;
    thread_data[i].max_iterations = max_iterations;

    pthread_create(&threads[i], NULL, parallelIterations, &thread_data[i]);
    start_row = thread_data[i].end_row;
  };

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
}

void selection(int *origin_x, int *origin_y, float *min_real,
               float *min_imaginary, int *destination_x, int *destination_y,
               float *zoom_x, float *zoom_y,
               double grid[SCREENWIDTH][SCREENHEIGHT][2],
               int mandelbrotSet[SCREENWIDTH][SCREENHEIGHT]) {

  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    *origin_x = GetMouseX();
    *origin_y = GetMouseY();
  };

  if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && GetMouseX() > 120 &&
      GetMouseY() > 40) {
    *destination_x = GetMouseX();
    *destination_y = GetMouseY();
    DrawRectangleLines(*origin_x, *origin_y, *destination_x - *origin_x,
                       (*destination_y - *origin_y), RAYWHITE);
  }

  if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON) && GetMouseX() > 120 &&
      GetMouseY() > 30) {
    *min_real = grid[*origin_x][*origin_y][0];
    *min_imaginary = grid[*origin_x][*origin_y][1];
    *zoom_x *= ((*destination_x - *origin_x) / (float)SCREENWIDTH);
    *zoom_y *= ((*destination_y - *origin_y) / (float)SCREENHEIGHT);
    pixelsToCoords(SCREENWIDTH, SCREENHEIGHT, *zoom_x, *zoom_y, *min_real,
                   *min_imaginary, grid);
    iterateOverGrid(SCREENWIDTH, SCREENHEIGHT, grid, mandelbrotSet,
                    MAX_ITERATIONS);
  }
};

void scroll(int *origin_x, int *origin_y, float *min_real, float *min_imaginary,
            float *zoom_x, float *zoom_y,
            double grid[SCREENWIDTH][SCREENHEIGHT][2],
            int mandelbrotSet[SCREENWIDTH][SCREENHEIGHT]) {
  int dest_x;
  int dest_y;
  if (GetMouseWheelMove() != 0) {
    dest_x = GetMouseX();
    dest_y = GetMouseY();
    float direction = -GetMouseWheelMove();
    if (*zoom_x < 1.0 || direction == -1.0) {
      *zoom_x *= 1.0 + 0.05 * direction;
      *zoom_y *= 1.0 + 0.05 * direction;

      *min_real = grid[*origin_x][*origin_y][0] -
                  (0.05 * direction) * grid[*origin_x][*origin_y][0];
      *min_imaginary = grid[*origin_x][*origin_y][1] -
                       (0.05 * direction) * grid[*origin_x][*origin_y][1];

      pixelsToCoords(SCREENWIDTH, SCREENHEIGHT, *zoom_x, *zoom_x, *min_real,
                     *min_imaginary, grid);
      iterateOverGrid(SCREENWIDTH, SCREENHEIGHT, grid, mandelbrotSet,
                      MAX_ITERATIONS);
    };
  }
};

int main(void) {

  const int screenWidth = SCREENWIDTH;
  const int screenHeight = SCREENHEIGHT;
  double(*grid)[screenHeight][2] =
      malloc(sizeof(double[screenWidth][screenHeight][2]));
  int(*mandelbrotSet)[screenHeight] =
      malloc(sizeof(int[screenWidth][screenHeight]));
  float x_scale = 1.0;
  float y_scale = 1.0;
  float real_start = -2.0;
  float imag_start = -2.0;
  pixelsToCoords(screenWidth, screenHeight, x_scale, y_scale, real_start,
                 imag_start, grid);
  iterateOverGrid(screenWidth, screenHeight, grid, mandelbrotSet,
                  MAX_ITERATIONS);

  InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

  SetTargetFPS(10); // Set our game to run at 60 frames-per-second
  //--------------------------------------------------------------------------------------

  // Main game loop
  int origin_x = 0;
  int origin_y = 0;
  int dest_x = 0;
  int dest_y = 0;
  Rectangle slider_box = (Rectangle){24, 24, 120, 30};
  float start = 2.0, end = 7.0;
  float last_exponent = exponent;
  Image mandelbrotImage = GenImageColor(screenWidth, screenHeight, BLACK);
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    if (last_exponent != exponent) {
      last_exponent = exponent;
      iterateOverGrid(screenWidth, screenHeight, grid, mandelbrotSet,
                      MAX_ITERATIONS);
      /*GPUIterations(grid, *mandelbrotSet, SCREENWIDTH, SCREENHEIGHT,*/
      /*              MAX_ITERATIONS);*/
    };

    //----------------------------------------------------------------------------------
    BeginDrawing();
    ClearBackground(BLACK);
    // Fill the image with Mandelbrot set data
    for (int i = 0; i < screenWidth; i++) {
      for (int j = 0; j < screenHeight; j++) {
        Color pixelColor = getColor(mandelbrotSet[i][j]);
        ImageDrawPixel(&mandelbrotImage, i, j, pixelColor);
      }
    }

    Texture2D mandelbrotTexture = LoadTextureFromImage(mandelbrotImage);

    DrawTexture(mandelbrotTexture, 0, 0, WHITE);
    selection(&origin_x, &origin_y, &real_start, &imag_start, &dest_x, &dest_y,
              &x_scale, &y_scale, grid, mandelbrotSet);
    scroll(&origin_x, &origin_y, &real_start, &imag_start, &x_scale, &y_scale,
           grid, mandelbrotSet);

    GuiSliderBar(slider_box, "2", "7", &exponent, start, end);
    char slider_val[10];
    sprintf(slider_val, "%.2f", exponent);
    DrawText(slider_val, 160, 30, 18, RAYWHITE);
    EndDrawing();

    //----------------------------------------------------------------------------------
  }

  UnloadImage(mandelbrotImage);
  // De-Initialization
  CloseWindow(); // Close window and OpenGL context
  //--------------------------------------------------------------------------------------

  return 0;
}
