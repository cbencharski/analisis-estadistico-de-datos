# Análisis Estadístico de Datos

Trabajos prácticos y examen final del curso **Análisis Estadístico de Datos** del Instituto Sabato (CNEA/UNSAM), dictado por Diego Ravignani (ITeDA). Cursado en el 2.º cuatrimestre de 2024.

El curso recorre la estadística desde la probabilidad hasta el ajuste de modelos a datos. Casi todos los trabajos usan **simulaciones Monte Carlo**: se generan muchos datos aleatorios con Python y se comparan los resultados simulados con lo que predice la teoría.

## Contenido

| TP | Tema | Qué se hace | Notebook |
|---|---|---|---|
| 1 | Probabilidad discreta | Suma de dos variables de Poisson | [TP1](./TP1_Bencharski.ipynb) |
| 2 | Probabilidad continua | Cambio de variables y distribución χ² | [TP2](./TP2_Bencharski.ipynb) |
| 3 | Probabilidad multidimensional | Normal bivariada y elipse de covarianza | [TP3](./TP3_Bencharski.ipynb) |
| 4 | Estimación de parámetros | Sesgo de estimadores de la desviación estándar | [TP4](./TP4_Bencharski.ipynb) |
| 5 | Estimación de intervalos | Cobertura de intervalos de confianza para una proporción | [TP5](./TP5_Bencharski.ipynb) |
| 6 | Test de hipótesis | Diagnóstico de diabetes: errores tipo I y II, poder y Bayes | [TP6](./TP6_Bencharski.ipynb) |
| Final | Ajuste de datos | Ajuste de máxima verosimilitud de la eficiencia de un detector | [Final](./TP7_Bencharski_AED.ipynb) |

**Herramientas:** Python · NumPy · SciPy (`scipy.stats`) · pandas · Matplotlib · seaborn

---

## Repaso por trabajo práctico

### TP1: Probabilidad discreta

**Problema:** se simulan 1000 veces dos variables de Poisson, $X_1$ con $\mu_1 = 1$ y $X_2$ con $\mu_2 = 2$, y se estudia la suma $Y = X_1 + X_2$.

**Conceptos clave:**

- **Distribución de Poisson:** cuenta eventos raros e independientes en un intervalo (por ejemplo, partículas por segundo). Su media y su varianza son iguales: $E = Var = \mu$.
- **Suma de Poisson independientes:** la suma también es Poisson, con $\mu = \mu_1 + \mu_2$. Las medias se suman y, como las variables son independientes, las varianzas también.
- **Simulación Monte Carlo:** generar muchos valores aleatorios (`poisson.rvs`) y comparar el histograma normalizado con la distribución teórica (`poisson.pmf`).

**Resultado:** el histograma de $Y$ coincide con una Poisson de $\mu = 3$.

### TP2: Probabilidad continua

**Problema:** se toman 20 variables normales estándar $X_i$, se elevan al cuadrado ($Y_i = X_i^2$) y se suman ($Z = \sum Y_i$). Se repite 10.000 veces.

**Conceptos clave:**

- **Cambio de variables:** si $Y = g(X)$, la densidad de $Y$ se obtiene a partir de la de $X$ dividiendo por $|dg/dx|$. Así se demuestra que el cuadrado de una normal estándar sigue una **χ² con 1 grado de libertad**.
- **Distribución χ²:** la suma de $k$ normales estándar al cuadrado sigue una χ² con $k$ grados de libertad, con media $k$ y varianza $2k$.
- **Teorema central del límite:** una suma de muchas variables tiende a una normal. Con $k = 20$ la χ² todavía es algo asimétrica, por eso la normal ajusta peor.

**Resultado:** media simulada de 19,997 y varianza de 39,6, frente a los valores teóricos de 20 y 40. La χ²(20) describe el histograma mejor que una normal con la misma media y varianza.

### TP3: Probabilidad multidimensional

**Problema:** se simulan 1000 puntos de una normal bivariada ($\mu_1 = 2.3$, $\mu_2 = 1.5$, $\sigma_1 = 1.2$, $\sigma_2 = 0.5$, $\rho = 0.7$) y se cuenta qué fracción cae dentro de la elipse de 1σ.

**Conceptos clave:**

- **Matriz de covarianza:** $V = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}$. La correlación $\rho$ inclina la elipse.
- **Forma cuadrática:** $Q = (x-\mu)^T V^{-1} (x-\mu)$ mide qué tan lejos está un punto del centro, teniendo en cuenta la forma de la distribución. Un punto está dentro de la elipse $r\sigma$ si $Q \le r^2$.
- **Q sigue una χ² con 2 grados de libertad**, así que la probabilidad dentro de la elipse de 1σ es $P(\chi^2_2 \le 1) = 39{,}3\,\%$.
- **En 2D, 1σ no es el 68 %:** el 68 % vale solo en una dimensión. En dos dimensiones, la elipse de 1σ contiene apenas el 39 %.

**Resultado:** el 40,0 % de los puntos simulados cae dentro de la elipse, frente al 39,3 % teórico.

### TP4: Estimación de parámetros

**Problema:** con muestras de solo 2 valores de una normal estándar, se comparan dos estimadores de la desviación estándar ($\sigma = 1$): $t_1 = \sqrt{s^2}$ y $t_2 = \sqrt{\pi s^2/2}$. Se repite 1000 veces.

**Conceptos clave:**

- **Sesgo:** diferencia entre el valor medio del estimador y el valor verdadero del parámetro. Un estimador es **insesgado** si en promedio acierta.
- **Corrección de Bessel:** la varianza muestral $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$ divide por $n-1$ y no por $n$, y así es insesgada. En NumPy: `np.var(x, ddof=1)`.
- **¿Por qué $\sqrt{s^2}$ es sesgado aunque $s^2$ no lo sea?** Por la **desigualdad de Jensen**: la raíz cuadrada es una función cóncava, por lo que $E[\sqrt{s^2}] \le \sqrt{E[s^2]} = \sigma$. En otras palabras, la raíz de un promedio no es el promedio de las raíces, y $t_1$ subestima sistemáticamente a $\sigma$. El sesgo es grande cuando la muestra es chica.
- El factor $\sqrt{\pi/2}$ de $t_2$ corrige justamente ese sesgo para $n = 2$.

**Resultado:** el valor medio de $t_1$ fue 0,811 (sesgo ≈ 0,19) y el de $t_2$ fue 1,017 (prácticamente insesgado).

### TP5: Estimación de intervalos

**Problema:** se lanza 32 veces una moneda con $p = 0.2$ y se estima $\hat{p} = k/n$. Se simulan 10.000 tandas y se cuenta cuántas veces el intervalo de confianza del 95 % contiene al valor verdadero.

**Conceptos clave:**

- **Intervalo de confianza:** rango calculado a partir de los datos que, si se repitiera el experimento muchas veces, contendría al parámetro verdadero en un porcentaje dado de los casos (el **nivel de confianza**).
- **Cobertura:** la fracción de intervalos que realmente contienen al parámetro. Un intervalo es **exacto** si su cobertura coincide con el nivel de confianza.
- **Intervalo estándar (de Wald):** $\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n}$. Usa la aproximación normal, que falla con $n$ chico o con $p$ cerca de 0 o 1.
- **Intervalo corregido (de Agresti–Coull):** usa $t = (k+2)/(n+4)$, como si se sumaran 2 éxitos y 2 fracasos. Corre el centro hacia 0,5 y mejora mucho la cobertura.
- $\hat{p} = k/n$ es un estimador **insesgado**: su valor medio coincide con $p$ para cualquier valor de $p$.

**Resultado:** el intervalo estándar cubrió solo el 88,8 % de las veces (menos que el 95 % nominal). Con la corrección, la cobertura subió a 95,5 %.

### TP6: Test de hipótesis

**Problema:** la glucosa en sangre de personas sanas sigue una normal con $\mu_0 = 110.6$ y $\sigma_0 = 24.8$ mg/dl, y la de personas con diabetes, una normal con $\mu_1 = 142.3$ y $\sigma_1 = 29.6$. Se diagnostica diabetes si la glucosa supera 126 mg/dl.

**Conceptos clave:**

- **Hipótesis nula ($H_0$):** la persona está sana. **Hipótesis alternativa ($H_1$):** tiene diabetes. El **valor crítico** (126 mg/dl) separa las dos decisiones.
- **Error de tipo I ($\alpha$):** rechazar $H_0$ siendo cierta, es decir, diagnosticar diabetes a una persona sana (falso positivo).
- **Error de tipo II ($\beta$):** no rechazar $H_0$ siendo falsa, es decir, no detectar a una persona con diabetes (falso negativo).
- **Poder o sensibilidad ($1-\beta$)** y **especificidad ($1-\alpha$)**. Al mover el valor crítico, uno baja y el otro sube.
- **Teorema de Bayes:** la probabilidad de tener diabetes dado un test positivo depende de la **prevalencia** de la enfermedad en la población, no solo de la calidad del test.

**Resultado:** $\alpha$ = 0,27, $\beta$ = 0,29, poder = 0,71 y especificidad = 0,73. Con una prevalencia del 35 %, una persona con test positivo tiene diabetes con una probabilidad del **59 %**.

### Examen final: ajuste de datos

**Problema:** se mide la eficiencia de un detector de partículas contando cuántas de 30 partículas detecta ($k$), para 20 energías distintas. Se ajusta un modelo sigmoide: $\varepsilon(E; a, b) = \frac{1}{1+e^{-(E-a)/b}}$.

**Conceptos clave:**

- **Estimador de máxima verosimilitud (MLE):** el valor del parámetro que hace más probables los datos observados. Para una binomial, $\hat{\varepsilon} = k/n$.
- **Ajuste con función de costo binomial:** como los datos son conteos de éxitos sobre intentos, se maximiza la verosimilitud binomial en lugar de usar cuadrados mínimos.
- **Errores y correlación de los parámetros:** salen de la matriz de covarianza del ajuste. Una correlación baja indica que los parámetros miden cosas distintas.
- **Región de confianza:** zona del plano $(a, b)$ compatible con los datos a 1σ.
- **Bondad del ajuste:** se calcula un χ² y su p-valor. Un p-valor muy chico indica que el modelo no describe los datos, y uno muy cercano a 1 puede indicar errores sobreestimados.

**Resultado:**

- $a = 0.571 \pm 0.014$: la energía en la que el detector alcanza el 50 % de eficiencia.
- $b = 0.121 \pm 0.009$: qué tan rápida es la transición de 0 a 100 % de eficiencia.
- Correlación entre $a$ y $b$: 0,02.
- $\chi^2 = 24.0$ con 18 grados de libertad, p-valor = 0,155. El modelo es consistente con los datos.

---

## Cómo ejecutar los notebooks

```bash
pip install numpy scipy pandas matplotlib seaborn jupyter
jupyter notebook
```

El TP3 y el examen final usan `danatools` y `likefit`, librerías provistas por la cátedra.
