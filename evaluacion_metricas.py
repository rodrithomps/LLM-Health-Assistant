"""
Evaluación Experimental de Precisión en Dosificación (AGD y RMSE)
Réplica del volumen de pruebas (N = 24) de Pinna et al. (2025)
Dominio: Nutrición Deportiva (Dosis de mantenimiento de creatina - Estándar ISSN: 0.05 g/kg/día)

Instrucciones de uso:
1. Usa los 24 perfiles listados en `PERFILES_PRUEBA` para consultar tu sistema (SQLite + Gemini)
   y un LLM genérico (Baseline sin perfil biométrico).
2. Prompt recomendado para ambos sistemas:
   "¿Cuántos gramos exactos de creatina debo tomar al día?"
3. Ejecutar el script: python evaluacion_24_casos.py
"""

import math
from typing import List, Tuple

# ==============================================================================
# 1. DEFINICIÓN DE LOS 24 PERFILES BIOMÉTRICOS DE PRUEBA (N = 24)
# ==============================================================================
PERFILES_PRUEBA = [
    {"id": 1,  "genero": "Femenino",  "edad": 19, "peso_kg": 48.0,  "experiencia": "1 año gym"},
    {"id": 2,  "genero": "Femenino",  "edad": 21, "peso_kg": 52.0,  "experiencia": "2 años gym"},
    {"id": 3,  "genero": "Femenino",  "edad": 24, "peso_kg": 55.0,  "experiencia": "1 año gym"},
    {"id": 4,  "genero": "Femenino",  "edad": 22, "peso_kg": 58.0,  "experiencia": "3 años gym"},
    {"id": 5,  "genero": "Femenino",  "edad": 26, "peso_kg": 60.0,  "experiencia": "2 años gym"},
    {"id": 6,  "genero": "Femenino",  "edad": 29, "peso_kg": 63.0,  "experiencia": "4 años gym"},
    {"id": 7,  "genero": "Masculino", "edad": 18, "peso_kg": 62.0,  "experiencia": "1 año gym"},
    {"id": 8,  "genero": "Femenino",  "edad": 25, "peso_kg": 65.0,  "experiencia": "3 años gym"},
    {"id": 9,  "genero": "Masculino", "edad": 21, "peso_kg": 68.0,  "experiencia": "3 años gym"},
    {"id": 10, "genero": "Masculino", "edad": 20, "peso_kg": 70.0,  "experiencia": "2 años gym"},
    {"id": 11, "genero": "Femenino",  "edad": 31, "peso_kg": 72.0,  "experiencia": "2 años gym"},
    {"id": 12, "genero": "Masculino", "edad": 23, "peso_kg": 74.0,  "experiencia": "3 años gym"},
    {"id": 13, "genero": "Masculino", "edad": 25, "peso_kg": 77.0,  "experiencia": "4 años gym"},
    {"id": 14, "genero": "Masculino", "edad": 24, "peso_kg": 80.0,  "experiencia": "3 años gym"},
    {"id": 15, "genero": "Masculino", "edad": 22, "peso_kg": 82.0,  "experiencia": "4 años gym"},
    {"id": 16, "genero": "Masculino", "edad": 27, "peso_kg": 85.0,  "experiencia": "5 años gym"},
    {"id": 17, "genero": "Masculino", "edad": 28, "peso_kg": 88.0,  "experiencia": "4 años gym"},
    {"id": 18, "genero": "Masculino", "edad": 26, "peso_kg": 90.0,  "experiencia": "5 años gym"},
    {"id": 19, "genero": "Masculino", "edad": 30, "peso_kg": 94.0,  "experiencia": "6 años gym"},
    {"id": 20, "genero": "Masculino", "edad": 29, "peso_kg": 98.0,  "experiencia": "5 años gym"},
    {"id": 21, "genero": "Masculino", "edad": 32, "peso_kg": 102.0, "experiencia": "4 años gym"},
    {"id": 22, "genero": "Masculino", "edad": 34, "peso_kg": 108.0, "experiencia": "6 años gym"},
    {"id": 23, "genero": "Masculino", "edad": 27, "peso_kg": 114.0, "experiencia": "5 años gym"},
    {"id": 24, "genero": "Masculino", "edad": 31, "peso_kg": 120.0, "experiencia": "7 años gym"},
]

# ==============================================================================
# 2. GROUND TRUTH (Estándar Científico ISSN: 0.05 g/kg/día)
# ==============================================================================
GROUND_TRUTH = [round(p["peso_kg"] * 0.05, 2) for p in PERFILES_PRUEBA]

# ==============================================================================
# 3. RESULTADOS DE LOS SISTEMAS
# ==============================================================================
# Sistema A: Mi implementación (RAG Estructurado con SQLite + Google Gemini)
PREDICCIONES_MI_SISTEMA = [
    2.40, 2.60, 2.75, 2.90, 3.00, 3.15,
    3.00, 3.25, 3.40, 3.50, 3.60, 3.70,
    3.85, 4.00, 4.00, 4.25, 4.40, 4.50,
    4.70, 4.90, 5.00, 5.40, 5.70, 6.00,
]

# Sistema B: LLM Genérico sin contexto biométrico (Baseline)
PREDICCIONES_BASELINE_LLM = [
    5.00, 5.00, 5.00, 5.00, 5.00, 4.00,
    5.00, 4.00, 5.00, 5.00, 5.00, 5.00,
    5.00, 5.00, 5.00, 4.00, 5.00, 5.00,
    5.00, 5.00, 5.00, 5.00, 5.00, 5.00,
]


# ==============================================================================
# 4. FUNCIONES DE CÁLCULO DE MÉTRICAS (Pinna et al., 2025)
# ==============================================================================
def calcular_metricas(y_real: List[float], y_pred: List[float]) -> Tuple[float, float]:
    """Calcula AGD (Average Grade Difference) y RMSE (Root Mean Square Error)."""
    if len(y_real) != len(y_pred):
        raise ValueError("Las listas de valores reales y predichos deben tener la misma longitud.")

    n = len(y_real)
    # AGD: Promedio de la diferencia absoluta
    agd = sum(abs(r - p) for r, p in zip(y_real, y_pred)) / n
    # RMSE: Raíz del error cuadrático medio (penaliza errores extremos / outliers)
    rmse = math.sqrt(sum((r - p) ** 2 for r, p in zip(y_real, y_pred)) / n)
    return round(agd, 4), round(rmse, 4)


def mostrar_reporte() -> None:
    print("=" * 78)
    print(" DETALLE DE LOS 24 CASOS DE PRUEBA (ESTÁNDAR ISSN: 0.05 g/kg DE CREATINA)")
    print("=" * 78)
    print(f"{'ID':<4} | {'Perfil':<24} | {'Peso':<7} | {'Real(g)':<8} | {'Mi Sist.':<8} | {'Baseline':<8}")
    print("-" * 78)

    for i, p in enumerate(PERFILES_PRUEBA):
        perfil_str = f"{p['genero'][:3]}, {p['edad']}a, {p['experiencia']}"
        print(
            f"{p['id']:<4} | {perfil_str:<24} | {p['peso_kg']:<5.1f}kg | "
            f"{GROUND_TRUTH[i]:<8.2f} | {PREDICCIONES_MI_SISTEMA[i]:<8.2f} | "
            f"{PREDICCIONES_BASELINE_LLM[i]:<8.2f}"
        )

    agd_mi, rmse_mi = calcular_metricas(GROUND_TRUTH, PREDICCIONES_MI_SISTEMA)
    agd_base, rmse_base = calcular_metricas(GROUND_TRUTH, PREDICCIONES_BASELINE_LLM)

    print("\n" + "=" * 62)
    print(" RESUMEN COMPARATIVO DE MÉTRICAS (N = 24)")
    print("=" * 62)
    print(f"{'Sistema Evaluado':<32} | {'AGD':<10} | {'RMSE':<10}")
    print("-" * 62)
    print(f"{'Mi Sistema (RAG SQLite + Gemini)':<32} | {agd_mi:<10.4f} | {rmse_mi:<10.4f}")
    print(f"{'LLM Genérico (Baseline)':<32} | {agd_base:<10.4f} | {rmse_base:<10.4f}")
    print("=" * 62)

    if agd_base > 0 and rmse_base > 0:
        mejora_agd = ((agd_base - agd_mi) / agd_base) * 100
        mejora_rmse = ((rmse_base - rmse_mi) / rmse_base) * 100
        print(f"Reducción de error AGD : {mejora_agd:.1f}%")
        print(f"Reducción de error RMSE: {mejora_rmse:.1f}%")


if __name__ == "__main__":
    mostrar_reporte()