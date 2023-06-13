## Arquitectura elegida

Para la generación de la arquitectura se ha decidio usar una arquitectura de tipo _**Encoder-Decoder**_, pues se pensó
que sería la más adecuada para la tarea en cuestión. La epxplicación de ello reside en el hecho que se quiere hacer
aprender a la red a realizar un movmimento. Para ello se debe enseñar a la red a cambiar de estado a partir de la
imagen. Aquí es donde entra la arquitectura Encoder-Decoder, pues se puede usar la parte de codificación para extraer
características del estado que representa la imagen y usar el decoder para que sea capaz de reconstruir un nuevo estado
y su imagen.

Más especificamente se ha decidido usar una arquitectura de tipo U-Net. Esta arquitectura se ha elegido por su buen
rendimiento en la tarea de segmentación de imágenes médica, ya que se puede enteder que la red entiende donde está la
zona de interés de la imagen y puede aprender que es en esa zona de interés donde se debe aplicar un movmiento. La
arquitectura se compone de dos partes principales, la parte de codificación y la parte de decodificación. La parte de
codificación se encarga de extraer características de la imagen de entrada y la parte de decodificación se encarga de
reconstruir la imagen de salida a partir de las características extraídas. Hasta aquí se puede ver que es de tipo
Encoder-Decoder, pero la arquitectura U-Net añade una conexión entre la parte de codificación y la parte de
decodificación. Esta conexión se hace a través de un skip connection, que es una conexión directa entre la parte de
codificación y la parte de decodificación.

La forma en la que se configura una U-Net peude variar según las necesidades de la tarea. En el presente proyecto se ha
decidio probar varias arquitecturas U-Net, pero todas ellas comparten la misma estructura básica. La primera de ellas es
una U-Net cláscia con capas convolucionales. La segunda es similar a la primera, pero en lugar de usar capas
convolucionales se usa capas tipo ResNet. La tercera es una modifcación que en lugar de usar capas convolucionales
usa capas tipo Transformer.

Todas ellas tienen una subred que se encarga de predecir el movimiento que se debe aplicar a la imagen de entrada y es
una red tipo ResNet que se conecta a la parte de codificación.

### U-Net

Esta arquitectura se compone de 3 bloques de codificación y 3 bloques de decodificación. Un diagrama de la arquitectura
se puede ver en la siguiente imagen:

IMAGEN

Esta red, aunque al principio parecía que estaba aprendiendo, acababa por dar como resultado de la función de pérdida
un NaN. Después de varios intentos de fine-tuning de los hiperparámetros no se ha conseguido que la red aprenda. Por lo
que se decidió modificar la arquitectura y el entrenamiento para añadir las capas ResNet y gradient clipping.

### U-Net+

Esta arquitectura se compone de 3 bloques de codificación y 3 bloques de decodificación. En lugar de usar capas
convolucionales se usa capas tipo ResNet. Un diagrama de la arquitectura se puede ver en la siguiente imagen:

IMAGEN

Esta red ya empezaba a dar resultados mucho mejores que la anterior, aunque se vió que no era suficiente. Pues la red
que predecia los movimientos no aprendía a predecir los movimientos, simplemente etiquetaba todas las muestras como
un único movimiento. Por lo que se decidió modificar la arquitectura para añadir las capas Transformer, que estaban
dando buenos resultados en la mayoría de los proyectos de Deep Learning.

### U-Net++

Esta arquitectura se compone de 3 bloques de codificación y 3 bloques de decodificación. En lugar de usar capas
convolucionales se usa capas tipo Transformer. Un diagrama de la arquitectura se puede ver en la siguiente imagen:

IMAGEN

## Resumen de la generación del dataset

En esta sección se va a proceder a hablar de como se generan los datasets usados para los experimentos. Cada muestra de
cada dataset se representará cómo una 3-tupla (x, y, m) : X, Y -> D, m -> M, donde D se define como el conjunto
D = {i / 255 | 0 <= i <= 255} que repsenta una imagen de un sólo canal (éscala de grises) normalizada diviendo por 255.
Se ha elegido esta forma de normalizar y no usando la media y la varianza porque es la manera más extendida en los
proyectos de Visión por Computador usando redes neuronales. El conjunto M se repsenta cómo un conjunto de tensores
unidiminsonales defninidos cómo one-hot encoding.

Con todo esto se puede dar una explicación a cada elemento de la 3-tupla (x, y, m):

 - x -> Imagen de entrada que representa un estado del problema.
 - y -> Imagen de salida que representa un nuevo estado procedente del estado x (al cuál se ha aplicado una acción).
 - m -> Movimiento aplicado al estado x para que se transforme en el estado y.
 
 
### Dataset 1: 8-puzzle

Para el dataset 1 se ha definido el 8-puzzle. Este problema consta de un tablero en dividido en 9 celdas, en 8 de las
cuales hay definidos los números del 1 al 8. En el tablero siempre hay una celda que queda sin número (que de ahora en
adelante se le denominará 0). Este 0 puede intercambiar su posición con los números que tenga arriba, abajo, a la
derecha o a la izquierda. El problema consiste en ordenar estos números moviendo el 0.

Este problema se puede representar de varias formas cómo un vector de 9 elementos o una matriz A3x3, pero en el presente
proyecto se va a usar la elección del vector por un mejor uso de la memória principal e indexación. Como se ha definido
anteriormente, hay 4 posibles movmientos que definen una permutación entre dos elementos del vector, concretamnte el 0 y
un número colidante a él. Ésto se puede aplicar al vector sumando o sustrayendo al índice en el que esté alojado el 0:

 - Mover hacia arriba: Restar al índice del 0 un 3.
 - Mover hacia abajo: Sumar al índice del 0 un 3.
 - Mover hacia la izquierda: Restar al índice del 0 un 1.
 - Mocer hacia la derecha: Sumar al índice del 0 un 1.
 
A partir de las definiciones anteriores se puede definir cuando un movmiento es válido o no usando los límites del
tablero y los números modulares:
 
 - Si se produce un movimiento hacia arriba y el resultado es negativo el movmiento es inválido.
 - Si se produce un movimiento hacia abajo y el resultado es mayor a 8 el movmiento es inválido.
 - Si se produce un movimiento hacia la izquierda y el resultado módulo 3 es 2 el movmiento es inválido.
 - Si se produce un movimiento hacia la derecha y el resultado módulo 3 es 0 el movmiento es inválido.
 
Cada movmiento se representará como la operación aplicada al 0 "+3 -3 +1 -1" y su codificación one hot
"1000 0100 0010 0001".Cada número del tablero será representado cómo una imagen del dataset MNIST (incluido el 0 como
casilla vacía), por lo que cada imagen usada por este dataset seran los números del 0 al 8 pertenecientes a la tarea de
MNIST.

Con todo esto se puede definir cómo será la 3-tupla de este dataset:

 - x -> Un estado del tablero.
 - y -> Un estado con el movmiento m aplciado (es decir, con el 0 desplazado).
 - m -> Movimento aplicado en formato one hot.
 
Las muestras se generan de manera online, es decir, en cada paso de entrenamiento se generan las imágenes y los
movimentos que se van a usar. El proceso consiste en generar una serie de movmientos aleatorios a partir de un tablero
ordenado. EL total de movmimentos aplicados es el tamaño de batch elegido multiplicado por 4. Acto seguido se eligen de
forma aleatorio tantas imágenes como el tamaño de batch indique y esto es la muestra que se devuelve, pues para cada
movmiento aplicado se obtiene la 3-tupla anteriormente indicada: La imagen antes del movmiento, la imagen después del
movmiento y el movimiento.