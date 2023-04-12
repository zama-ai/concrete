#!/bin/env bash

BUILD_TYPE=$1

if [[ ${BUILD_TYPE,,} = "release" ]]; then
  echo "build"
else
  echo "build_${BUILD_TYPE}"
fi

