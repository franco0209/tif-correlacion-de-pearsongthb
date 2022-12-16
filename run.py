
#!/usr/bin/env python
# coding: utf-8


#Importamos librerias para Ciencia de Datos y Machine Learning
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

def caratula():
    st.write('''# Universidad Nacional de San Agustín de Arequipa \n ## Escuela Profesional de Ingeniería de Telecomunicaciones \n\n''')
    st.image('https://www.unsa.edu.pe/wp-content/uploads/sites/3/2018/05/Logo-UNSA.png')
    st.write('''\n### Ingeniero Renzo Bolivar - Docente DAIE\n ### Curso : Computación 1\n #### Alumnos:\n\n ''')
    st.write('- Jean Carlos Rodriguez Luna')
    st.write('- Gianfranco Melgar Tejada')
    st.write('- Antony Yampier García Torres')
    st.write('- Bruno David Alarcón Aquise')
    st.write('- Erick Raúl Carrillo Barrera')
    st.write('- Franco Jesus Cahua Soto\n\n')
    st.write('''### INVESTIGACIÓN FORMATIVA\n ### PROYECTO FINAL\n ### PYTHON - Inteligencia Artificial''')
    st.write('''### OBJETIVOS''')
    st.write('''Los Objetivos de la investigación formativa son:
    
- **Competencia Comunicativa** Presentación de sus resultados con lenguaje de programación Python utilizando los archivos Jupyter Notebook.
- **Competencia Aprendizaje**: con las aptitudes en **Descomposición** (desarticular el problema en pequeñas series de soluciones), **Reconocimiento de Patrones** (encontrar simulitud al momento de resolver problemas), **Abstracción** (omitir información relevante), **Algoritmos** (pasos para resolución de un problema).
- **Competencia de Trabajo en Equipo**: exige habilidades individuales y grupales orientadas a la cooperación, planificación, coordinación, asignación de tareas, cumplimiento de tareas y solución de conflictos en pro de un trabajo colectivo, utilizando los archivos Jupyter Notebook los cuales se sincronizan en el servidor Gitlab con comandos Git.\n''')

def teoria():
    st.write('''### Aplicación en IA\n ### Sistema Recomendador\n''')
    st.write('''El Sistema recomendador deberá encontrar la compatibilidad o similitud entre un grupo de personas encuestadas, en las áreas de:

   -Musica
   
   -Peliculas
    
   -Comida
    
   -Lugares que desean Conocer
    
   -Obras de Arte
   
   La compatibilidad o similitud será encontrada con el algoritmo de Correlación de Pearson y será verificada con  La Matrix de Correlación de Pearson con una librería de Python y utilizando una función personal''')
    st.write('''### Base teórica''')

    st.write('''**Análisis de Correlación**
El análisis de correlación es el primer paso para construir modelos explicativos y predictivos más complejos.

A menudo nos interesa observar y medir la relación entre 2 variables numéricas mediante el análisis de correlación. Se trata de una de las técnicas más habituales en análisis de datos y el primer paso necesario antes de construir cualquier modelo explicativo o predictivo más complejo. Para poder tener el Datset hay que recolectar información a travez de encuentas.
**¿Qué es la correlación?**
La correlación es un tipo de asociación entre dos variables numéricas, específicamente evalúa la tendencia (creciente o decreciente) en los datos.

Dos variables están asociadas cuando una variable nos da información acerca de la otra. Por el contrario, cuando no existe asociación, el aumento o disminución de una variable no nos dice nada sobre el comportamiento de la otra variable.

Dos variables se correlacionan cuando muestran una tendencia creciente o decreciente.

**¿Cómo se mide la correlación?**
Tenemos el coeficiente de correlación lineal de Pearson que se sirve para cuantificar tendencias lineales, y el coeficiente de correlación de Spearman que se utiliza para tendencias de aumento o disminución, no necesariamente lineales pero sí monótonas.

**Correlación de Pearson**
El coeficiente de correlación lineal de Pearson mide una tendencia lineal entre dos variables numéricas.
Es el método de correlación más utilizado, pero asume que:
La tendencia debe ser de tipo lineal.
No existen valores atípicos (outliers).
Las variables deben ser numéricas.
Tenemos suficientes datos (algunos autores recomiendan tener más de 30 puntos u observaciones).
Los dos primeros supuestos se pueden evaluar simplemente con un diagrama de dispersión, mientras que para los últimos basta con mirar los datos y evaluar el diseño que tenemos.

**Cómo se interpreta la correlación**
El signo nos indica la dirección de la relación, como hemos visto en el diagrama de dispersión.

un valor positivo indica una relación directa o positiva,
un valor negativo indica relación indirecta, inversa o negativa,
un valor nulo indica que no existe una tendencia entre ambas variables (puede ocurrir que no exista relación o que la relación sea más compleja que una tendencia, por ejemplo, una relación en forma de U).
La magnitud nos indica la fuerza de la relación, y toma valores entre −1 a 1. Cuanto más cercano sea el valor a los extremos del intervalo (1 o −1) más fuerte será la tendencia de las variables, o será menor la dispersión que existe en los puntos alrededor de dicha tendencia. Cuanto más cerca del cero esté el coeficiente de correlación, más débil será la tendencia, es decir, habrá más dispersión en la nube de puntos.

si la correlación vale 1 o −1 diremos que la correlación es “perfecta”,
si la correlación vale 0 diremos que las variables no están correlacionadas.

**Varianza**
La varianza de una muestra o de un conjunto de valores, es la sumatoria de las desviaciones al cuadrado con respecto al promedio o a la media, todo esto dividido entre el número total de observaciones menos 1. De manera muy general se puede decir que la varianza es la desviación estándar elevada al cuadrado.

La varianza de una muestra presenta la siguiente fórmula:
(Σ x²) / N ) - μ²

En la mayoría de los casos es muy difícil, por no decir imposible obtener un N total de datos, por ejemplo, al hablar de individuos de una población, no es posible muestrear a todos estos individuos, ya que existe un factor de tiempo y recursos limitante. Es por esto que se suele utilizar los estadísticos para estimar los parámetros de una población. De acuerdo a la manera en que se encuentra escrita esta fórmula, las unidades de la varianza presenta las mismas unidades de la variable, pero elevada al cuadrado.También, vemos que la varianza no puede ser negativa, por lo que el mínimo valor que se puede obtener en esta es cero.

**Covarianza**
La covarianza mide la relación direccional entre los rendimientos de dos activos. Una covarianza positiva significa que los rendimientos de los activos se mueven juntos, mientras que una covarianza negativa significa que se mueven a la inversa. La covarianza se calcula analizando las sorpresas de retorno (desviaciones estándar del rendimiento esperado) o multiplicando la correlación entre las dos variables aleatorias por la desviación estándar de cada variable. La covarianza evalúa cómo se mueven juntos los valores medios de dos variables aleatorias. Si el rendimiento de la acción A aumenta cada vez que aumenta el rendimiento de la acción B y se encuentra la misma relación cuando disminuye el rendimiento de cada acción, se dice que estas acciones tienen una covarianza positiva. En finanzas, las covarianzas se calculan para ayudar a diversificar las tenencias de valores.

Cov (X, Y) = E(X x Y) – E(X) x (E(Y).

**Covarianza positiva**
Una covarianza positiva entre dos variables indica que estas variables tienden a ser más grandes o más pequeñas al mismo tiempo. En otras palabras, una covarianza positiva entre las variables X y y indica que X es más alto que el promedio al mismo tiempo y es superior a la media y viceversa. Cuando se representan en un gráfico bidimensional, los puntos de datos tenderán a inclinarse hacia arriba.

**Covarianza negativa**
Cuando la covarianza calculada es menor que cero, indica que las dos variables tienen una relación inversa. En otras palabras, año X un valor que es menor que el promedio tiende a estar asociado con una y que es superior a la media y viceversa.

**Media**
La media o promedio, se usa en matemáticas, economía o estadística y es el resultado de un conjunto de operaciones realizadas a varios números para que ese resultado pueda representar a todo un conjunto.

**Formas de calcular la media**
Existen diferentes formas de calcular la media, aunque la más conocida es la media aritmética, hay otras formas de calcular la media muy útiles en determinadas ocasiones y que se utilizan de forma muy frecuente en economía.

**Media aritmética**
Es la forma más conocida y la que todos hemos utilizado en muchas ocasiones. Para calcularla sumamos todos los valores y dividimos el resultado entre el número de valores que tenemos. Por ejemplo, si en una asignatura hemos obtenido 5 calificaciones y el peso de todas ellas para la nota final es el mismo y estas calificaciones son 6, 8 10 y 9, nuestra nota final será: (6+8+10+9)/4=8,25''')


def propuesta():
    st.write('''\n### PROPUESTA\n''')
    st.write('''\n#### 1. Dataset\n''')

    st.write('''Para poder tener el Datset hay que recolectar información con una encuenta elaborada por nosotros.

Encuesta :
La encuesta la realizamos en Google-Form donde se solicitara escoger una comida.

Donde si escoge 1 es el que menos le gusta hasta 5 que es el que mas le gusta (escala de liker)
Formulario de Google (Preguntas)
Comidas que mas le gusta al encuestado. Ejemplo: Ceviche, arroz con pato, etc.

Formulario de Google (Imagenes)
\n''')
    st.image('https://i.ibb.co/SND173C/ceviche.png')
    st.image('https://i.ibb.co/qyXHKSV/cg.png')

def conclusiones():
    st.write('''\n### CONCLUSIONES\n''')
    st.write('''- ¿Se valido o no los resultados?
   - Los resultados Validados son:
   - ¿Es efectivo el metodo de correlación de pearson?
   - Correlación de Pearson y Regresión Lineal, ¿cual es su relación?
\n**¿Se valido o no los resultados?**

-Si, los resultados obtenidos en la matriz de validación coinciden exactamente con la que elaboró pandas; esto lo podemos verificar tanto en la gráfica de calor como en la búsqueda de los mayores coeficientes de correlación.

Los resultados Validados son:

¿Es efectivo el metodo de correlación de pearson?

Si, puesto que a traves de una valor aritmético muestra valiosa información acerca de las similitudes en algun aspecto a comparar que pueden ser útiles a muchas empresas.

Correlación de Pearson y Regresión Lineal, ¿cual es su relación?

La correlación de Pearson cuantifica como de relacionadas están dos variables, mientras que la regresión lineal consiste en generar una ecuación que, basándose en la relación existente entre ambas variables, permita predecir el valor de una a partir de la otra.

**Referencias**

Profesor de Matematicas: John Gabriel Muñoz Cruz https://www.linkedin.com/in/jgmc

Profesor de Computo: Canal:Código Maquina https://youtu.be/IBMrXyTR6CU

Profesor de Computo: Manuel Gonzáles https://www.youtube.com/@manuelgonzalez1644

Correlación lineal y Regresión lineal simple: Joaquín Amat Rodrigo https://www.cienciadedatos.net/documentos/24_correlacion_y_regresion_lineal''')


data= pd.read_csv('COMIDAS 13.csv')


for varf in data.columns[1:]:
    data[varf].fillna(data[varf].mode()[0],inplace=True)

n = data[data.columns[1:]].to_numpy()
m = data[data.columns[0]].to_numpy()

df = pd.DataFrame(n.T, columns = m)
m_corr_PANDAS = df.corr()
m_corr_PANDAS = np.round(m_corr_PANDAS, 
                       decimals = 2) 

def grafcalor():
    calor=m_corr_PANDAS
    plt.matshow(calor,cmap="bwr",vmin=-1,vmax=+1)
    plt.xticks(range(31),rotation=90)
    plt.yticks(range(31))
    plt.colorbar()
    st.pyplot()

co = m_corr_PANDAS[m_corr_PANDAS.columns[1:]].to_numpy()
a = m_corr_PANDAS[m_corr_PANDAS.columns[0]].to_numpy()
b = m_corr_PANDAS[m_corr_PANDAS.columns[1]].to_numpy()
c = m_corr_PANDAS[m_corr_PANDAS.columns[2]].to_numpy()
d = m_corr_PANDAS[m_corr_PANDAS.columns[3]].to_numpy()
e = m_corr_PANDAS[m_corr_PANDAS.columns[4]].to_numpy()
f = m_corr_PANDAS[m_corr_PANDAS.columns[5]].to_numpy()
g = m_corr_PANDAS[m_corr_PANDAS.columns[6]].to_numpy()
h = m_corr_PANDAS[m_corr_PANDAS.columns[7]].to_numpy()
i = m_corr_PANDAS[m_corr_PANDAS.columns[8]].to_numpy()
j = m_corr_PANDAS[m_corr_PANDAS.columns[9]].to_numpy()
k = m_corr_PANDAS[m_corr_PANDAS.columns[10]].to_numpy()
l = m_corr_PANDAS[m_corr_PANDAS.columns[11]].to_numpy()
mu = m_corr_PANDAS[m_corr_PANDAS.columns[12]].to_numpy()
nu = m_corr_PANDAS[m_corr_PANDAS.columns[13]].to_numpy()
ñ = m_corr_PANDAS[m_corr_PANDAS.columns[14]].to_numpy()
o = m_corr_PANDAS[m_corr_PANDAS.columns[15]].to_numpy()
p = m_corr_PANDAS[m_corr_PANDAS.columns[16]].to_numpy()
q = m_corr_PANDAS[m_corr_PANDAS.columns[17]].to_numpy()
r = m_corr_PANDAS[m_corr_PANDAS.columns[18]].to_numpy()
s = m_corr_PANDAS[m_corr_PANDAS.columns[19]].to_numpy()
t = m_corr_PANDAS[m_corr_PANDAS.columns[20]].to_numpy()
u = m_corr_PANDAS[m_corr_PANDAS.columns[21]].to_numpy()
v = m_corr_PANDAS[m_corr_PANDAS.columns[22]].to_numpy()
w = m_corr_PANDAS[m_corr_PANDAS.columns[24]].to_numpy()
aa = m_corr_PANDAS[m_corr_PANDAS.columns[25]].to_numpy()
bb = m_corr_PANDAS[m_corr_PANDAS.columns[26]].to_numpy()
cc = m_corr_PANDAS[m_corr_PANDAS.columns[27]].to_numpy()
dd = m_corr_PANDAS[m_corr_PANDAS.columns[28]].to_numpy()
ee = m_corr_PANDAS[m_corr_PANDAS.columns[29]].to_numpy()

def control(z):
    lista=[]
    lista2=[]
    for x in z:
        lista.append(x)
    for y in lista:
        if y<1:
            lista2.append(y)
    return(max(lista2))

final=[control(a),control(b),control(c),control(d),control(e),control(f),control(g),control(h),control(i),control(j),control(k),control(l),control(mu),control(nu),control(ñ),control(o),control(p),control(q),control(r),control(s),control(t),control(u),control(v),control(w),control(aa),control(bb),control(cc),control(dd),control(ee)]
final.sort()

import math
def pearsonfinal(x,y):
    suma1 = 0
    suma2 = 0
    suma3=0
    c= 0
    c1=0
    c2= 0
    for n in x:
        suma1 = suma1 +((x[c]-(x.mean()))*((y[c]-(y.mean()))))
        c += 1
        primc=suma1
    for m in x:
        suma2 = suma2 +pow((x[c1]-(x.mean())),2)
        suma3 = suma3 +pow((y[c2]-(y.mean())),2)
        c1 += 1
        c2 += 1
        segundc=(math.sqrt(suma2))*(math.sqrt(suma3))
    return (round((primc/segundc),2))
    
matrizf1=([])
for nu1 in range(len(m)):
    matrizf1.append([0]*len(m) )
for nu1 in range(0,len(m)):
    for nu2 in range(0,len(m)):
        matrizf1[nu1][nu2]=(pearsonfinal(n[nu1],n[nu2]))
    print(' ')

encabezado=m

daf= pd.DataFrame(data=matrizf1,columns=encabezado)

def grafcalor2():
    calor2=daf
    plt.matshow(calor2,cmap="bwr",vmin=-1,vmax=+1)
    plt.xticks(range(31),rotation=90)
    plt.yticks(range(31))
    plt.colorbar()
    st.pyplot()

# ## Ubicar el máximo coeficiente de correlacion utilizando la matriz de validación

as2 = matrizf1[0]
bs2 = matrizf1[1]
cs2 = matrizf1[2]
ds2 = matrizf1[3]
es2 = matrizf1[4]
fs2 = matrizf1[5] 
gs2 = matrizf1[6]
hs2 = matrizf1[7]
is2 = matrizf1[8]
js2 = matrizf1[9]
ks2 = matrizf1[10]
ls2 = matrizf1[11]
mus2 = matrizf1[12]
nus2 =matrizf1[13]
ñs2 =matrizf1[14]
os2 = matrizf1[15]
ps2 = matrizf1[16]
qs2 =matrizf1[17]
rs2 =matrizf1[18]
ss2 =matrizf1[19]
ts2 =matrizf1[20]
us2 =matrizf1[21]
vs2 =matrizf1[22]
ws2 =matrizf1[23]
aas2 =matrizf1[23]
bbs2 =matrizf1[23]
ccs2 =matrizf1[23]
dds2 =matrizf1[23]
ees2 =matrizf1[23]
print('----------------------------------')

def controla2(zs2):
    listas2_=[]
    lista2s1_=[]
    for xs2 in zs2:
        listas2_.append(xs2)
    for ys2 in listas2_:
        if ys2<1:
            lista2s1_.append(ys2)
    return(max(lista2s1_))

finals2=[controla2(as2),controla2(bs2),controla2(cs2),controla2(ds2),controla2(es2),controla2(fs2),controla2(gs2),controla2(hs2),controla2(is2),controla2(js2),controla2(ks2),controla2(ls2),controla2(mus2),controla2(nus2),controla2(ñs2),controla2(os2),controla2(ps2),controla2(qs2),controla2(rs2),controla2(ss2),controla2(ts2),controla2(us2),controla2(vs2),controla2(ws2),controla2(aas2),controla2(bbs2),controla2(ccs2),controla2(dds2),controla2(ees2)]
finals2.sort()


    
st.title('P.I.F:Matriz de correlación de Pearson')
st.sidebar.title('Contenido')
options= st.sidebar.radio('Paginas',options=['Presentación','Base teórica', 'Propuesta', 'Matríz de correlación de Pearson(Pandas)','Gráfica de calor (Pandas)','Búsqueda de los coeficientes de correlación mas altos(Pandas)','Matríz de correlación de Pearson(Validación)','Gráfica de calor (Validación)','Búsqueda de los coeficientes de correlación mas altos(Validación)','Conclusiones y referencias'])
if options=='Presentación':
    caratula()
elif options=='Base teórica':
    teoria()
elif options=='Propuesta':
    propuesta()
elif options=='Matríz de correlación de Pearson(Pandas)':
    st.write(m_corr_PANDAS)
elif options=='Gráfica de calor (Pandas)':
    grafcalor()
elif options=='Búsqueda de los coeficientes de correlación mas altos(Pandas)':
    st.write('Búsqueda del valor más alto de correlación de la matriz de Pandas.')
    st.write('El primer coeficiente de correlación mas alto es:\n ')
    st.write(final[28], 'entre:', m[7] ,' y ', m[14])
    st.write('\nEl segundo coeficiente de correlación mas alto es:\n ')
    st.write(final[26], 'entre:', m[9] ,' y ', m[22])
elif options=='Matríz de correlación de Pearson(Validación)':
    daf
elif options=='Gráfica de calor (Validación)':
    grafcalor2()
elif options=='Búsqueda de los coeficientes de correlación mas altos(Validación)':
    st.write('El primer coeficiente de correlación mas alto es:\n ')
    st.write(finals2[28], 'entre:', m[7] ,' y ', m[14])
    st.write('\nEl segundo coeficiente de correlación mas alto es:\n ')
    st.write(finals2[26], 'entre:', m[9] ,' y ', m[22])

elif options=='Conclusiones y referencias':
    conclusiones() 

    