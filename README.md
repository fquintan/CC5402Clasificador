# CC5402Clasificador
Tarea 3 del curso Búsqueda por Contenido de Imágenes y Videos
consiste en un clasificador de imágenes

Para compilar simplemente ejecutar make desde el directorio principal

Para ejecutar debe existir el directorio imagenes con las imagenes correspondientes 
a las categorías de auto, gato y pajaro, la estructura del directorio debe ser como sigue:

imagenes
├── bird_test
├── bird_train
├── bird_val
├── car_test
├── car_train
├── car_val
├── cat_test
├── cat_train
├── cat_val

Una vez que se tiene esta estructura se puede ejecutar el script "script_test_100.sh"
para realizar una prueba del programa completo usando 100 centros para el 
algoritmo de clustering.