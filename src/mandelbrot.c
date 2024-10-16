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
#define SCREENWIDTH 1920 / 2
#define SCREENHEIGHT 1080 / 2
#define GPU 1

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
  float power = 3.8;

  // main cardiod checking
  /*if ((power == 2.0 && pow(creall(z) + 1.0, 2) + pow(cimagl(z), 2)) < 1.0
   * / 16.0)*/
  /*  return MAX_ITERATIONS;*/
  for (iter = 0; (iter < MAX_ITERATIONS); iter++) {
    z = cpow(z, power) + start_c;
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

int main(void) {

  const int screenWidth = SCREENWIDTH;
  const int screenHeight = SCREENHEIGHT;
  double(*grid)[screenHeight][2] =
      malloc(sizeof(double[screenWidth][screenHeight][2]));
  int(*mandelbrotSet)[screenHeight] =
      malloc(sizeof(int[screenWidth][screenHeight]));
  float x_scale = 1.0;
  float y_scale = 1.0;
  float x_start = -2.0;
  float y_start = -2.0;
  pixelsToCoords(screenWidth, screenHeight, x_scale, y_scale, x_start, y_start,
                 grid);
  iterateOverGrid(screenWidth, screenHeight, grid, mandelbrotSet,
                  MAX_ITERATIONS);

  InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

  SetTargetFPS(20); // Set our game to run at 60 frames-per-second
  //--------------------------------------------------------------------------------------

  // Main game loop
  int start_x = 0;
  int start_y = 0;
  int curr_x = 0;
  int curr_y = 0;
  Image mandelbrotImage = GenImageColor(screenWidth, screenHeight, BLACK);
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    // Update
    // Draw
    //----------------------------------------------------------------------------------
    BeginDrawing();
    ClearBackground(BLACK);
    Rectangle slider_box = (Rectangle){24, 24, 120, 30};
    float start = 2.0, val = 2.0, end = 7.0;
    GuiSliderBar(slider_box, "2", "9", &val, start, end);
    // Fill the image with Mandelbrot set data
    for (int i = 0; i < screenWidth; i++) {
      for (int j = 0; j < screenHeight; j++) {
        Color pixelColor = getColor(mandelbrotSet[i][j]);
        ImageDrawPixel(&mandelbrotImage, i, j, pixelColor);
      }
    }

    Texture2D mandelbrotTexture = LoadTextureFromImage(mandelbrotImage);

    BeginDrawing();
    DrawTexture(mandelbrotTexture, 0, 0, WHITE);

    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
      start_x = GetMouseX();
      start_y = GetMouseY();
    };

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
      curr_x = GetMouseX();
      curr_y = GetMouseY();
      DrawRectangleLines(start_x, start_y, curr_x - start_x, (curr_y - start_y),
                         RAYWHITE);
    }

    if (GetMouseWheelMove() != 0) {
      curr_x = GetMouseX();
      curr_y = GetMouseY();
      float direction = -GetMouseWheelMove();
      if (x_scale < 1.0 || direction == -1.0) {
        x_scale *= 1.0 + 0.05 * direction;
        y_scale *= 1.0 + 0.05 * direction;
        x_start = grid[start_x][start_y][0] -
                  (0.05 * direction) * grid[start_x][start_y][0];
        y_start = grid[start_x][start_y][1] -
                  (0.05 * direction) * grid[start_x][start_y][1];

        pixelsToCoords(screenWidth, screenHeight, x_scale, y_scale, x_start,
                       y_start, grid);
        iterateOverGrid(screenWidth, screenHeight, grid, mandelbrotSet,
                        MAX_ITERATIONS);
      };
    }
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
      x_start = grid[start_x][start_y][0];
      y_start = grid[start_x][start_y][1];
      x_scale = x_scale * ((curr_x - start_x) / (float)screenWidth);
      y_scale = y_scale * ((curr_y - start_y) / (float)screenHeight);
      pixelsToCoords(screenWidth, screenHeight, x_scale, y_scale, x_start,
                     y_start, grid);
      iterateOverGrid(screenWidth, screenHeight, grid, mandelbrotSet,
                      MAX_ITERATIONS);
    }
    EndDrawing();

    //----------------------------------------------------------------------------------
  }

  UnloadImage(mandelbrotImage);
  // De-Initialization
  CloseWindow(); // Close window and OpenGL context
  //--------------------------------------------------------------------------------------

  return 0;
}
