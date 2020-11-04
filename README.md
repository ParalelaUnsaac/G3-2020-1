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

## Wiki 📖

No tenemos un wiki :( ... pero puedes ver mas sobre el proyecto en el archivo .ipynb, puedes acceder haciendo click [aqui Wiki](https://colab.research.google.com/drive/1pC42W8I2eGt8ZEss9sjWhpHDtlFpy_k9#scrollTo=HDsNV972EYdX).


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



