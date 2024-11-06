import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pulp

# Configuración de la página y estilo de la portada
st.set_page_config(page_title="Optimización con Programación Entera", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f7f2f2;  
    }
    .title {
        color: #5a5a5a;  
        font-size: 30px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #8b5b93;  
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo de la portada
st.markdown('<div class="title">UNIVERSIDAD NACIONAL DEL ALTIPLANO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NELIDA ARACELY QUISPE CALATAYUD</div>', unsafe_allow_html=True)

# Mostrar las imágenes
col1, col2 = st.columns([1, 1])
with col1:
    st.image("Unap.png", width=80)
with col2:
    st.image("Finesi.png", width=80)

st.markdown("<br><br>", unsafe_allow_html=True)

# ===========================
# Ejercicio 8.1: Dakin’s Branch and Bound
# ===========================
st.header("Ejercicio 8.1: Dakin’s Branch and Bound")
st.write("Maximizar P(x1, x2, x3) = 4x1 + 3x2 + 3x3 sujeto a las restricciones:")
st.latex(r"""
\begin{cases}
4x_1 + 2x_2 + x_3 \leq 10 \\ 
3x_1 + 4x_2 + 2x_3 \leq 14 \\ 
2x_1 + x_2 + 3x_3 \leq 7 \\
x_1, x_2, x_3 \geq 0 \quad \text{y enteros}
\end{cases}
""")

# Definir y resolver el problema
prob_8_1 = pulp.LpProblem("Maximization_8_1", pulp.LpMaximize)
x1 = pulp.LpVariable('x1', lowBound=0, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, cat='Integer')
prob_8_1 += 4 * x1 + 3 * x2 + 3 * x3
prob_8_1 += 4 * x1 + 2 * x2 + x3 <= 10
prob_8_1 += 3 * x1 + 4 * x2 + 2 * x3 <= 14
prob_8_1 += 2 * x1 + x2 + 3 * x3 <= 7
prob_8_1.solve()

# Mostrar solución
st.write("Estado:", pulp.LpStatus[prob_8_1.status])
st.write(f"x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
st.write("Valor máximo de P =", pulp.value(prob_8_1.objective))

# Gráfico
fig, ax = plt.subplots(figsize=(2, 1.5))  # Ajustar tamaño
x_vals = np.linspace(0, 5, 200)
ax.plot(x_vals, (10 - 4 * x_vals) / 2, label="4x1 + 2x2 + x3 <= 10")
ax.plot(x_vals, (14 - 3 * x_vals) / 4, label="3x1 + 4x2 + 2x3 <= 14")
ax.plot(x_vals, (7 - 2 * x_vals), label="2x1 + x2 + 3x3 <= 7")
ax.fill_between(x_vals, 0, np.minimum(np.minimum((10 - 4 * x_vals) / 2, (14 - 3 * x_vals) / 4), (7 - 2 * x_vals)), color="gray", alpha=0.3)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
st.pyplot(fig)

# ===========================
# Ejercicio 8.2: Resolución como LP e IP
# ===========================
st.header("Ejercicio 8.2: Resolución como LP e IP")
st.write("Resolver el Ejercicio 8.1 relajando las variables (sin enteros) y luego resolviendo con enteros:")

# Resolver como LP sin enteros
prob_8_2_lp = prob_8_1.copy()
x1.cat, x2.cat, x3.cat = 'Continuous', 'Continuous', 'Continuous'
prob_8_2_lp.solve()
st.write("Solución LP (sin enteros):")
st.write(f"x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
st.write("Valor máximo de P LP =", pulp.value(prob_8_2_lp.objective))

# Gráfico del problema LP
fig, ax = plt.subplots(figsize=(2, 1.5))  # Ajustar tamaño
x_vals = np.linspace(0, 5, 200)
ax.plot(x_vals, (10 - 4 * x_vals) / 2, label="4x1 + 2x2 + x3 <= 10")
ax.plot(x_vals, (14 - 3 * x_vals) / 4, label="3x1 + 4x2 + 2x3 <= 14")
ax.plot(x_vals, (7 - 2 * x_vals), label="2x1 + x2 + 3x3 <= 7")
ax.fill_between(x_vals, 0, (7 - 2 * x_vals), color="gray", alpha=0.3)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
st.pyplot(fig)

# ===========================
# Ejercicio 8.3: Minimización con Planos de Corte
# ===========================
st.header("Ejercicio 8.3: Minimización con Planos de Corte")
st.write("Minimizar C(x, y) = x - y sujeto a las restricciones:")
st.latex(r"""
\begin{cases}
3x + 4y \leq 6 \\
x - y \leq 1 \\
x, y \geq 0 \quad \text{y enteros}
\end{cases}
""")

# Definir y resolver el problema
x = pulp.LpVariable('x', lowBound=0, cat='Integer')
y = pulp.LpVariable('y', lowBound=0, cat='Integer')
prob_8_3 = pulp.LpProblem("Minimization_8_3", pulp.LpMinimize)
prob_8_3 += x - y
prob_8_3 += 3 * x + 4 * y <= 6
prob_8_3 += x - y <= 1
prob_8_3.solve()

st.write("Estado:", pulp.LpStatus[prob_8_3.status])
st.write(f"x = {x.varValue}, y = {y.varValue}")
st.write("Valor mínimo de C =", pulp.value(prob_8_3.objective))

# Gráfico
fig, ax = plt.subplots(figsize=(2, 1.5))  # Ajustar tamaño
x_vals = np.linspace(0, 3, 100)
ax.plot(x_vals, (6 - 3 * x_vals) / 4, label="3x + 4y <= 6")
ax.plot(x_vals, x_vals - 1, label="x - y <= 1")
ax.fill_between(x_vals, 0, np.minimum((6 - 3 * x_vals) / 4, x_vals - 1), color="gray", alpha=0.3)
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)

# ===========================
# Ejercicio 8.4: Maximización con Planos de Corte
# ===========================
st.header("Ejercicio 8.4: Maximización con Planos de Corte")
st.write("Maximizar P(x1, x2, x3) = 4x1 + 3x2 + 3x3 sujeto a restricciones de corte.")
st.write("Nota: El proceso incluye el uso de cortes de Gomory para obtener soluciones enteras.")

# Crear el problema de Programación Lineal
def crear_problema():
    prob = pulp.LpProblem("Maximization_8_4", pulp.LpMaximize)
    
    # Definir las variables (inicialmente continuas)
    x1 = pulp.LpVariable('x1', lowBound=0, cat='Continuous')
    x2 = pulp.LpVariable('x2', lowBound=0, cat='Continuous')
    x3 = pulp.LpVariable('x3', lowBound=0, cat='Continuous')

    # Función objetivo
    prob += 4 * x1 + 3 * x2 + 3 * x3

    # Restricciones
    prob += 4 * x1 + 2 * x2 + x3 <= 10
    prob += 3 * x1 + 4 * x2 + 2 * x3 <= 14
    prob += 2 * x1 + x2 + 3 * x3 <= 7
    
    return prob, x1, x2, x3

# Función para verificar si la solución es entera
def es_entera(soluciones):
    return all([v % 1 == 0 for v in soluciones])

# Función para aplicar el corte de Gomory
def aplicar_corte(prob, x1, x2, x3, iteracion):
    # Se obtiene el valor actual de las variables
    sol = [x1.varValue, x2.varValue, x3.varValue]
    
    # Verificar si alguna variable tiene valor fraccionario
    corte_agregado = False
    for i, x in enumerate([x1, x2, x3]):
        if sol[i] % 1 != 0:  # Si el valor es fraccionario
            frac = sol[i] - int(sol[i])  # Parte fraccionaria
            if frac > 0.01:  # Solo si la fracción es significativa
                # Generar el corte de Gomory
                corte = frac * x <= int(sol[i])
                prob += corte
                st.write(f"Iteración {iteracion} - Corte de Gomory agregado: {frac} * {x.name} <= {int(sol[i])}")
                corte_agregado = True
                break  # Solo se aplica un corte por iteración
    
    return prob, corte_agregado

# Resolver el problema inicial
prob_8_4, x1, x2, x3 = crear_problema()
prob_8_4.solve()

# Mostrar los resultados
st.write("### Solución Inicial (Sin Cortes de Gomory):")
st.write(f"Estado: {pulp.LpStatus[prob_8_4.status]}") 
st.write(f"x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
st.write(f"Valor máximo de P = {pulp.value(prob_8_4.objective)}")

# Verificar si la solución es entera
if es_entera([x1.varValue, x2.varValue, x3.varValue]):
    st.write("### Solución Entera")
    st.write("La solución obtenida es entera y óptima.")
else:
    st.write("### Corte de Gomory Necesario")
    st.write("La solución obtenida no es entera. Se deben aplicar cortes de Gomory para obtener una solución entera.")
    
    # Aplicar cortes de Gomory hasta que la solución sea entera
    iteracion = 1
    while not es_entera([x1.varValue, x2.varValue, x3.varValue]) and iteracion < 50:
        st.write(f"### Iteración {iteracion} - Aplicando Corte de Gomory")
        prob_8_4, corte_realizado = aplicar_corte(prob_8_4, x1, x2, x3, iteracion)
        
        if corte_realizado:
            prob_8_4.solve()  # Resolver el problema después de agregar el corte
            st.write(f"Solución después del corte: x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
            st.write(f"Valor máximo de P después del corte = {pulp.value(prob_8_4.objective)}")
        else:
            st.write("### No se requiere más corte")
            break  # Si no se requiere más corte, salir del loop
        
        iteracion += 1

# Verificación final de la solución
if es_entera([x1.varValue, x2.varValue, x3.varValue]):
    st.write("### Solución Entera Encontrada")
    st.write(f"x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
    st.write(f"Valor máximo de P = {pulp.value(prob_8_4.objective)}")

# Gráfico Ejercicio 8.4 (Región factible)
fig_8_4, ax = plt.subplots(figsize=(2, 1.5))  # Ajustar tamaño

# Valores para x1
x_vals = np.linspace(0, 5, 200)

# Graficar las restricciones en 2D (simplificación para visualizar)
ax.plot(x_vals, (10 - 4 * x_vals) / 2, label="4x1 + 2x2 + x3 <= 10")
ax.plot(x_vals, (14 - 3 * x_vals) / 4, label="3x1 + 4x2 + 2x3 <= 14")
ax.plot(x_vals, (7 - 2 * x_vals), label="2x1 + x2 + 3x3 <= 7")

# Sombrear la región factible
ax.fill_between(x_vals, 0, np.minimum(np.minimum((10 - 4 * x_vals) / 2, (14 - 3 * x_vals) / 4), (7 - 2 * x_vals)), color="gray", alpha=0.3)

# Configuración del gráfico
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
plt.title("Región factible para el ejercicio 8.4")
st.pyplot(fig_8_4)

# ===========================
# Ejercicio 8.5: Selección de Proyectos
# ===========================
st.header("Ejercicio 8.5: Selección de Proyectos")
st.write("Seleccionar proyectos para maximizar el NPV bajo restricciones de presupuesto anual")

# Definir y resolver el problema binario
prob_8_5 = pulp.LpProblem("Maximization_8_5", pulp.LpMaximize)
p1 = pulp.LpVariable("p1", 0, 1, cat="Binary")
p2 = pulp.LpVariable("p2", 0, 1, cat="Binary")
p3 = pulp.LpVariable("p3", 0, 1, cat="Binary")
p4 = pulp.LpVariable("p4", 0, 1, cat="Binary")
p5 = pulp.LpVariable("p5", 0, 1, cat="Binary")
p6 = pulp.LpVariable("p6", 0, 1, cat="Binary")
prob_8_5 += 141 * p1 + 187 * p2 + 121 * p3 + 83 * p4 + 262 * p5 + 127 * p6
prob_8_5 += 75 * p1 + 90 * p2 + 60 * p3 + 30 * p4 + 100 * p5 + 50 * p6 <= 250
prob_8_5 += 25 * p1 + 35 * p2 + 15 * p3 + 20 * p4 + 25 * p5 + 20 * p6 <= 75
prob_8_5 += 20 * p1 + 0 * p2 + 15 * p3 + 10 * p4 + 20 * p5 + 10 * p6 <= 50
prob_8_5 += 15 * p1 + 0 * p2 + 15 * p3 + 5 * p4 + 20 * p5 + 30 * p6 <= 50
prob_8_5 += 10 * p1 + 30 * p2 + 15 * p3 + 5 * p4 + 20 * p5 + 40 * p6 <= 50
prob_8_5.solve()

st.write("Proyectos seleccionados:")
for var in [p1, p2, p3, p4, p5, p6]:
    st.write(f"{var.name} = {var.varValue}")
st.write("Valor máximo de NPV =", pulp.value(prob_8_5.objective))

# Gráfico del problema de selección de proyectos
fig, ax = plt.subplots(figsize=(2, 1.5))  # Ajustar tamaño
proyectos = ["P1", "P2", "P3", "P4", "P5", "P6"]
valores = [p1.varValue, p2.varValue, p3.varValue, p4.varValue, p5.varValue, p6.varValue]
ax.bar(proyectos, valores, color='lightblue')
ax.set_title("Selección de Proyectos - Ejercicio 8.5")
ax.set_ylabel("Selección (1 = Sí, 0 = No)")
st.pyplot(fig)
