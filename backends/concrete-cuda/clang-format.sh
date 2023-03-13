#!/bin/bash

clang-format-11 -i implementation/include/* && 
  clang-format-11 -i implementation/src/*.cu && 
  clang-format-11 -i implementation/src/*.cuh && 
  clang-format-11 -i implementation/src/*/*.* &&
  clang-format-11 -i implementation/test/*.h &&
  clang-format-11 -i implementation/test/*.cpp
