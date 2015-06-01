#!/bin/bash

# make clean
make

mkdir -p imagenes/resultados_clasificador


echo "-----------------------------------------------------------------"
echo "classifier 50"
./build/classifier 50 imagenes/BOVW_descriptors_50 imagenes/resultados_clasificador/resultados_50.txt
echo "-----------------------------------------------------------------"
echo "classifier 75"
./build/classifier 75 imagenes/BOVW_descriptors_75 imagenes/resultados_clasificador/resultados_75.txt
echo "-----------------------------------------------------------------"
echo "classifier 90"
./build/classifier 90 imagenes/BOVW_descriptors_90 imagenes/resultados_clasificador/resultados_90.txt
echo "-----------------------------------------------------------------"
echo "classifier 100"
./build/classifier 100 imagenes/BOVW_descriptors_100 imagenes/resultados_clasificador/resultados_100.txt
echo "-----------------------------------------------------------------"
echo "classifier 200"
./build/classifier 200 imagenes/BOVW_descriptors_200 imagenes/resultados_clasificador/resultados_200.txt
echo "-----------------------------------------------------------------"
echo "classifier 400"
./build/classifier 400 imagenes/BOVW_descriptors_400 imagenes/resultados_clasificador/resultados_400.txt
