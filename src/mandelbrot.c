#include "raylib.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

int iterate_mandelbrot(double complex start_c, int max_iterations) {
  float bound = 2;
  double complex z = start_c;
  int iter;

  for (iter = 0; (iter < max_iterations); iter++) {
    z = cpow(z, 2.0) + start_c;
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
  return ColorFromHSV(170 * t, 1.0f,  t);
}

void pixelsToCoords(int x, int y, double dest[x][y][2]) {
  double step_x = 4.0 / x;
  double step_y = 4.0 / y;
  double x_coord = -2.0, y_coord = -2.0;

  for (int i = 0; i < x; i++) {

    y_coord = -2.0;
    for (int j = 0; j < y; j++) {

      dest[i][j][0] = x_coord;
      dest[i][j][1] = y_coord;

      y_coord = y_coord + step_y;
    };
    x_coord = x_coord + step_x;
  };
}

void iterateOverGrid(double complex_plane[600][600][2], int grid[600][600],
                     int max_iterations) {
  for (int i = 0; i < 600; i++) {

    for (int j = 0; j < 600; j++) {
      double complex c = complex_plane[i][j][0] + complex_plane[i][j][1] * I;
      grid[i][j] = iterate_mandelbrot(c, max_iterations);
    };
  };
}

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(void) {
  // Initialization
  //--------------------------------------------------------------------------------------
  const int screenWidth = 600;
  const int screenHeight = 600;
  const int maxIterations = 800;
  double grid[screenWidth][screenHeight][2];
  int mandelbrotSet[screenWidth][screenHeight];
  pixelsToCoords(screenWidth, screenHeight, grid);
  iterateOverGrid(grid, mandelbrotSet, maxIterations);

  InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

  SetTargetFPS(60); // Set our game to run at 60 frames-per-second
  //--------------------------------------------------------------------------------------

  // Main game loop
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    // Update
    //----------------------------------------------------------------------------------
    // TODO: Update your variables here
    //----------------------------------------------------------------------------------

    // Draw
    //----------------------------------------------------------------------------------
    BeginDrawing();

    ClearBackground(RAYWHITE);
    for (int i = 0; i < 600; i++) {

      for (int j = 0; j < 600; j++) {
        DrawPixel(i, j, getColor(mandelbrotSet[i][j], maxIterations));
      };
    };
    EndDrawing();
    //----------------------------------------------------------------------------------
  }

  // De-Initialization
  //--------------------------------------------------------------------------------------
  CloseWindow(); // Close window and OpenGL context
  //--------------------------------------------------------------------------------------

  return 0;
}
