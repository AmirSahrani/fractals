#include "raylib.h"
#include <complex.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>

int iterate_mandelbrot(double complex start_c, int max_iterations) {
  float bound = 2;
  double complex z = start_c;
  int iter;

  for (iter = 0; (iter < max_iterations); iter++) {
    z = cpow(z, 3.0) + start_c;
    if (cabs(z) > 2.0) {
      return iter;
    }
  };
  return max_iterations;
}

Color getColor(int iterations, int max_iterations) {
  if (iterations == max_iterations)
    return WHITE;
  float t = (float)iterations / max_iterations;
  return ColorFromHSV(170 * t, 1.0f, pow(t, 2.0));
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

// Convert this to a cuda function, for loops should go over blocks and strides, also for simplicity make this one long array
// you can do some modulo arithmatic to make sense of it :)
void iterateOverGrid(int width, int height,
                     double complex_plane[width][height][2],
                     int grid[width][height], int max_iterations) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      double complex c = complex_plane[i][j][0] + complex_plane[i][j][1] * I;
      grid[i][j] = iterate_mandelbrot(c, max_iterations);
    };
  };
}

int main(void) {

  const int screenWidth = 1000;
  const int screenHeight = 1000;
  const int maxIterations = 100;
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
                  maxIterations);

  InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

  SetTargetFPS(60); // Set our game to run at 60 frames-per-second
  //--------------------------------------------------------------------------------------

  // Main game loop
  int start_x = 0;
  int start_y = 0;
  int curr_x = 0;
  int curr_y = 0;
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    // Update
    // Draw
    //----------------------------------------------------------------------------------
    BeginDrawing();
    ClearBackground(RAYWHITE);
    for (int i = 0; i < screenWidth; i++) {

      for (int j = 0; j < screenHeight; j++) {
        DrawPixel(i, j, getColor(mandelbrotSet[i][j], maxIterations));
      };
    };
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
      start_x = GetMouseX();
      start_y = GetMouseY();
    };

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
      curr_x = GetMouseX();
      curr_y = GetMouseY();
      DrawRectangleLines(start_x, start_y, curr_x - start_x, curr_y - start_y,
                         RAYWHITE);
    }
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
      x_start = grid[start_x][start_y][0];
      y_start = grid[start_x][start_y][1];
      x_scale = x_scale * ((curr_x - start_x) / (float)screenWidth);
      y_scale = y_scale * ((curr_y - start_y) / (float)screenHeight);
      pixelsToCoords(screenWidth, screenHeight, x_scale, y_scale, x_start,
                     y_start, grid);
      iterateOverGrid(screenWidth, screenHeight, grid, mandelbrotSet,
                      maxIterations);

      /*printf("scale x: %f\n", x_scale);*/
      /*printf("scale y: %f\n", y_scale);*/
    }
    EndDrawing();

    //----------------------------------------------------------------------------------
  }

  // De-Initialization
  //--------------------------------------------------------------------------------------
  CloseWindow(); // Close window and OpenGL context
  //--------------------------------------------------------------------------------------

  return 0;
}
