#!/bin/bash

make

mkdir -p imagenes/BOVW_descriptors_50
./build/extractor 50

mkdir -p imagenes/BOVW_descriptors_75
./build/extractor 75   
   
mkdir -p imagenes/BOVW_descriptors_90
./build/extractor 90

mkdir -p imagenes/BOVW_descriptors_100
./build/extractor 100

mkdir -p imagenes/BOVW_descriptors_200
./build/extractor 200
   
mkdir -p imagenes/BOVW_descriptors_400
./build/extractor 400
   
   