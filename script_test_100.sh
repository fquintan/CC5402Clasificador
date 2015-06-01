#!/bin/bash

make

echo "Extrayendo descriptores locales"
mkdir -p imagenes/descriptores_locales
./build/local_extractor

echo "----------------------------------------------------"
echo "Extrayendo descriptores globales BOVW"
mkdir -p imagenes/BOVW_descriptors_100
./build/extractor 100 

echo "Clasificando imagenes"
echo "----------------------------------------------------"
mkdir -p imagenes/resultados_clasificador
./build/classifier 100 imagenes/BOVW_descriptors_100 imagenes/resultados_clasificador/resultados_100.txt
