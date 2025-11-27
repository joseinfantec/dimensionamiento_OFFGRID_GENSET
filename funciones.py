import pandas as pd
import numpy as np

def print_results(title, results):
    print(f"\n--- {title} ---")
    print(f"NPV: {results['npv']:.2f}")
    pb = results.get('payback_year')
    if pb is not None:
        print(f"Payback (años): {pb:.1f}")
    print(f"Capex: {results['capex']:.2f}")

    years = sorted(results['generacion'].keys())

    data = {
        "Generación total": [results['generacion'][y] for y in years],
        "Consumo desde PV": [results['consumo_desde_pv'][y] for y in years],
        "Consumo desde BESS": [results['consumo_desde_bess'][y] for y in years],
        "Consumo desde Genset": [results['consumo_desde_genset'][y] for y in years],
        "Pérdidas FV": [results['losses_by_year'][y] for y in years],
        #"SOC final": [results['soc_end_by_year'][y] for y in years],
        #"Opex PV+BESS+GEN": [results['assets_opex_by_year'][y] for y in years],
        "Horas generador ON": [results['horas_generador_on'][y] for y in years],
        "Gross Savings": [results['gross_savings'][y] for y in years],
        #"Fuel Savings": [results['fuel_savings_by_year'][y] for y in years],
        "Fuel (lt) híbrido:": [results['fuel_hybrid_by_year'][y] for y in years],
        "Fuel (lt) solo genset:": [results['fuel_genonly_by_year'][y] for y in years],
    }

    df = pd.DataFrame(data, index=years)
    df.index.name = "Año"

    print("\nResultados por año:")
    print(df)

    # ✅ Guardar en Excel opcional
    # df.to_excel("resultados.xlsx")

    return df


def print_results_reducidos(title, best):
    if best is None:
        print(f"\n--- {title} ---")
        print("No se encontró una solución factible.")
        return None

    print(f"\n--- {title} ---")
    print(f"NPV: {best['npv']:.2f}")
    pb = best.get('Payback_yr') or best.get('payback_year')
    if pb is not None:
        print(f"Payback (años): {float(pb):.1f}")
    print(f"Capex: {best['CAPEX']:.2f}")
    print(f"PV: {best['PV_kWp']:.2f} kWp, BESS: {best['E_bess_kWh']:.2f} kWh")

    # Convertir a tabla anual con métricas solicitadas
    years = sorted(best['Fuel_liters_hybrid_by_year'].keys())

    # Claves esperadas que fueron añadidas en optimizer al 'best'
    consumo_pv = best.get('consumo_desde_pv') or {}
    consumo_bess = best.get('consumo_desde_bess') or {}
    generacion = best.get('generacion') or {}
    gen_hours = best.get('horas_generador_on') or {}
    gross_savings = best.get('gross_savings') or {}
    consumo_genset = best.get('consumo_desde_genset') or {}


    data = {
        "Consumo desde Genset": [consumo_genset.get(y, 0.0) for y in years],
        "Consumo desde FV": [consumo_pv.get(y, 0.0) for y in years],
        "Consumo desde BESS": [consumo_bess.get(y, 0.0) for y in years],
        "Pérdidas FV": [best['Losses_by_year'].get(y, 0.0) for y in years],
        "Horas generador ON": [gen_hours.get(y, 0) for y in years],
        "Gross Savings": [gross_savings.get(y, 0.0) for y in years],
        "Generacion FV": [generacion.get(y, 0.0) for y in years],
        # Extras que ya existían
        # "SOC final": [best['SOC_end_by_year'][y] for y in years],
        # "Opex PV+BESS+GEN": [best['Assets_OPEX_by_year'][y] for y in years],
    }

    df = pd.DataFrame(data, index=years)
    df.index.name = "Año"

    print("\nResultados por año:")
    print(df)

    # Guardar a Excel
    # df.to_excel("resultados_optimizador.xlsx")

    return df


def interp_lph_from_curve(percent, curve_dict):
    """
    percent: porcentaje de carga (0..inf), por ejemplo 30 -> 30%
    curve_dict: {25: lph25, 50: lph50, 75: lph75, 100: lph100}
    """
    # Preparar arrays ordenados
    xp = [0, 25, 50, 75, 100]
    fp = [0.0,
          float(curve_dict.get(25, 0.0)),
          float(curve_dict.get(50, 0.0)),
          float(curve_dict.get(75, 0.0)),
          float(curve_dict.get(100, 0.0))]
    percent_clamped = percent
    if percent_clamped < 0:
        percent_clamped = 0.0
    if percent_clamped <= 100:
        lph = float(np.interp(percent_clamped, xp, fp))
        return lph
    else:
        lph_100 = fp[-1]
        if lph_100 == 0:
            return 0.0
        lph = lph_100 * (percent_clamped / 100.0)
        return float(lph)