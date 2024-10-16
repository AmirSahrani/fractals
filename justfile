alias b := build

build:
  cc src/mandelbrot.c src/raygui.h -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -o tgt/mandelbrot

run:
  cc src/mandelbrot.c src/raygui.h -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -o tgt/mandelbrot
  ./tgt/mandelbrot
gpu:
 nvcc src/mandelbrot.c -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -o tgt/mandelbrot
 ./tgt/mandelbrot

debug: 
 cc src/mandelbrot.c -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -o tgt/mandelbrot -g
 gdb tgt/mandelbrot

