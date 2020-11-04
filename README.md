# **<center> 💻 G3 2020-I (ALGORITMOS PARALELOS Y DISTRIBUIDOS) 💻 </center>**

---

### Datos Academicos 📖

- **Institucion:** Universidad Nacional de San Antonio Abad del Cusco
- **Facultad:** Facultad de ingenieria electrica, electronica, informatica y mecanica
- **Escuela Prof:** Ingenieria Informatica y de Sistemas

#### Docente:
- **Quintanilla Portugal Roxana Lisette** - _Docente_ - [Concytec](http://directorio.concytec.gob.pe/appDirectorioCTI/VerDatosInvestigador.do?id_investigador=40930).

#### Trabajo:

- Documentar y entender un programa paralelizable.
- Elegimos un programa y lo ejecutamos en google colaboratory

#### Autores:✒️

- **Calderon Tintaya Fallcha Xiomara** - _GitHub Account_ - [Fallcha](https://github.com/Fallcha)
- **Deza Condori Rosmel Uriel** - _GitHub Account_ - [RosmelGDeza](https://github.com/RosmelGDeza)
- **Guevara Ferro Cristian** - _GitHub Account_ - [Cristian1938](https://github.com/cristian1938)
- **Holguin Josh** - _GitHub Account_ - [Josh](https://github.com/JoshYts)
- **Ore Gamarra Abraham Benjamin** - _GitHub Account_ - [benjaminx321](https://github.com/benjaminx321)
- **Rojas Garay Jafet Caleb** - _GitHub Account_ - [JafetCaleb](https://github.com/JafetCaleb)
- **Sullca Peralta Melanie Indira** - _GitHub Account_ - [Melanie279](https://github.com/Melanie279)

---
## Empezamos... 🚀

.Programa que cuenta los numeros impares generados dentro de un arreglo 

### Construido con... 🛠️

- Lenguaje: [Dev-C++](https://bloodshed-dev-c.softonic.com/)
- Servicio Cloud: [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)
- Diagrama de flujo: [Visual Paradigm](https://www.visual-paradigm.com/)
- Editor: [Visual Studio Code](https://code.visualstudio.com/)

## Codificacion del programa 📄
```py
%%cu
//cuda_runtime_api.h define los funciones y tipos de host públicos para la API CUDA tiempo de ejecución
//cuda_runtime.h define todo cuda_runtime_api.h hace, asi como de tipo incorporado definiciones y superposiciones de funciones para las extensiones de lenguaje CUDA y funciones intrínsecas dispositivo.
#include "cuda_runtime.h"
//Importa / Incluye las funciones, constantes y macros desde la librería de Entrada / Salida estándard (standar input/output
//los periféricos estándar son, monitor, teclado, discoduro, impresora … y esta librería te permite leer desde o enviar información hacia estos periféricos
//como leer variables de ese teclado, o enviar una texto a un archivo en disco.
#include <stdio.h>
//Es el archivo de cabecera de la biblioteca estándar de propósito general de el lenguaje de programación C. Contiene los prototipos de funciones de C para gestión de memoria dinámica, control de procesos y otras. Es compatible con C++ donde se conoce como cstdlib.
#include <stdlib.h>
//cuda.h define el host público funciones y tipos para la API de controlador CUDA
#include <cuda.h>

#define MAXVALUE 10000

//------------------------------------
void numberGen(int N, int max, int* store) //Funcion generar numeros
{
    //Declarando variable entero i
    int i;
    //Tiempo para generar numeros aleatorios
    srand(time(0));
    for (i = 0; i < N; i++)//For para recorrer los numeros
        store[i] = rand() % max; //Almacena en el arreglo
}
//------------------------------------
//Funcion __global__ que se puede llama r desde el host, y se ejecuta en el dispositivo
__global__ void countOdds(int* d, int N, int* odds)
{   
    //Asignamos memoria compartida dinamicamente(extern significa que la matriz hace referencia a la memoria compartida declarada en otro lugar)
    extern __shared__ int count[];

    int myID = blockIdx.x * blockDim.x + threadIdx.x;//Añadiendo el tamaño del grid y del bloque
    int localID = threadIdx.x; // Ponemos el id en cada hilo
    count[localID] = 0;
    if (myID < N)
        count[localID] = (d[myID] % 2); //Un modulo paran detectar si es par o impar
    __syncthreads(); //Sincroniza los subprocesos locales escribiendo en la memoria caché

    // fase de reducción
    int step = 1;
    while (((localID | step) < blockDim.x) && ((localID & step) == 0))
    {
        count[localID] += count[localID | step];
        step <<= 1;
        __syncthreads();
    }

   
    // adiciona al contador global
    if (localID == 0)
        atomicAdd(odds, count[0]);
}

//------------------------------------
int sharedSize(int b)
{
    return b * sizeof(int);
}

//------------------------------------

int main(int argc, char** argv)
{
    //Definimos la cantidad de numeros que se genera aleatoriamente
    int N = 50000;
    //Define punteros host(h*) y device (d*)
    int* ha, * hres, * da, * dres;   // punteros host (h*) y device (d*)

    ha = new int[N];//Declaramos un arreglo en la variable ha
    hres = new int[1];//Declaramos un arreglo en la variable hres
 
    //Almacena un espacio en la memoria da
    cudaMalloc((void**)&da, sizeof(int) * N);
    //Almacena un espacio en la memoria para dres
    cudaMalloc((void**)&dres, sizeof(int) * 1);
 
    //Llama a la funcion numbeGen()
    numberGen(N, MAXVALUE, ha);

    cudaMemcpy(da, ha, sizeof(int) * N, cudaMemcpyHostToDevice);//Copia datos a la memoria del dispositivo
    cudaMemset(dres, 0, sizeof(int));//llenamos los primeros bytes de recuento del área de memoria apuntada por dres con el valor de byte valor constante.

    int blockSize, gridSize;
 
    //Devuelve el tamaño de cuadrícula y bloque que logra la máxima ocupación potencial para una función de dispositivo.
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&gridSize, &blockSize, (void*)countOdds, sharedSize, N);

    gridSize = ceil(1.0 * N / blockSize);
    printf("Grid : %i    Block : %i\n", gridSize, blockSize);
    countOdds << < gridSize, blockSize, blockSize * sizeof(int) >> > (da, N, dres);

    //Copia datos a la memoria del dispostivo
    cudaMemcpy(hres, dres, sizeof(int), cudaMemcpyDeviceToHost);

   // chequeo
    int oc = 0;
    for (int i = 0; i < N; i++)//Hcae un recorrido
        if (ha[i] % 2){
            printf("%i ",ha[i]);
            oc++;
        }
    printf("");
    //Imprime cantidad de numeros imapres en el arreglo
    printf("\nCantidad de numeros impares - Chequeo\n");
    printf("%i %i\n", *hres, oc);
    cudaFree((void*)da);//Libera memoria de da
    cudaFree((void*)dres);//Libera memoria de dres
 
    delete[]ha; //Operacion eliminar ha
    delete[]hres;//Operacion eliminar hres
    cudaDeviceReset();//Libreria contexto de cuda, quita las asignaciones de todo los dispositivos
    getchar();
    return 0;
}

```


## Colaboratory 📖

No tenemos un wiki :( ... pero puedes ver mas sobre el proyecto en el archivo .ipynb, puedes acceder haciendo click [aqui Colab](https://colab.research.google.com/drive/1pC42W8I2eGt8ZEss9sjWhpHDtlFpy_k9#scrollTo=HDsNV972EYdX).


## Despliegue 📦

_Para poder programar en cuda primero tenemos que ejecutar una serie de instrucciones en nuestro mismo colab haciendo un copia y pega .

```py
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update

!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2
!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin

```



