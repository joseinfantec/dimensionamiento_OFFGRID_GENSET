from multiprocessing import Pool
import numpy as np
from simulator import simulate_operation
import pandas as pd
import time

# Mayor refine_factor, mayor zona explorada escapando de mínimos locales (más tiempo)
# Refine_steps son las veces que refinamos el código.
# nPV y nE puntos iniciales a evaluar, si los aumento, aumento el tiempo (impacto cuadrático)


def evaluate_grid_point(args):
    PV, E, irr, load, cfg = args
    res = simulate_operation(PV, E, irr, load, cfg)
    return (PV, E, res['npv'], res['feasible'], res['capex'],
            res['assets_opex_by_year'],
            res['fuel_hybrid_by_year'],
            res['fuel_genonly_by_year'],
            res['fuel_cost_hybrid'],
            res['fuel_cost_genonly'],
            res['soc_end_by_year'],
            res['losses_by_year'],
            res.get('payback_year'))

def grid_search_optimize(irr_annual, load_annual, cfg, PV_range=(0,500), E_range=(0,500), nPV=21, nE=21, parallel=True, nprocs=4, refine_steps=2, refine_factor=0.25):
    start_time = time.time()
    PV_min, PV_max = PV_range
    E_min, E_max = E_range
    PV_grid = np.linspace(PV_min, PV_max, nPV)
    E_grid = np.linspace(E_min, E_max, nE)
    #E_grid = np.arange(np.ceil(E_min / 12) * 12,E_max + 1, 12)   #PARA EVALUAR BESS SIGENERGY
    tasks = [(pv, eb, irr_annual, load_annual, cfg) for pv in PV_grid for eb in E_grid]
    unique = {}
    for (pv, eb, irr_, load_, cfg_) in tasks:
        key = (round(float(pv), 6), round(float(eb), 6))
        if key not in unique:
            unique[key] = (pv, eb, irr_, load_, cfg_)
    tasks = list(unique.values())

    results = []
    if parallel:
        with Pool(processes=nprocs) as pool:
            for r in pool.imap_unordered(evaluate_grid_point, tasks):
                results.append(r)
    else:
        for t in tasks:
            results.append(evaluate_grid_point(t))


    df = pd.DataFrame(results, columns=['PV_kWp', 'E_bess_kWh', 'npv', 'Feasible',
                                        'CAPEX', 'Assets_OPEX_by_year', 'Fuel_liters_hybrid_by_year', 'Fuel_liters_genonly_by_year',
                                        'Fuel_cost_hybrid', 'Fuel_cost_genonly',
                                        'SOC_end_by_year', 'Losses_by_year', 'Payback_yr', ])
    df_factible = df[df['Feasible'] == True]
    best = None
    if not df_factible.empty:
        idx = df_factible['npv'].idxmax()
        best_row = df_factible.loc[idx]
        best = dict(best_row)

    # Refinamiento
    for step in range(refine_steps):
        if best is None:
            break
        pv0 = best['PV_kWp']
        e0 = best['E_bess_kWh']
        pv_half_span = (PV_max - PV_min) * (refine_factor / (2 ** (step+1)))
        e_half_span = (E_max - E_min) * (refine_factor / (2 ** (step+1)))
        new_PV_min = max(PV_min, pv0 - pv_half_span)
        new_PV_max = min(PV_max, pv0 + pv_half_span)
        new_E_min = max(E_min, e0 - e_half_span)
        new_E_max = min(E_max, e0 + e_half_span)

        PV_grid = np.linspace(new_PV_min, new_PV_max, nPV)
        #E_grid = np.arange(np.ceil(new_E_min / 12) * 12,new_E_max + 1, 12)   #PARA EVALUAR BESS SIGENERGY
        E_grid = np.linspace(new_E_min, new_E_max, nE)
        tasks = [(pv, eb, irr_annual, load_annual, cfg) for pv in PV_grid for eb in E_grid]
        unique = {}
        for (pv, eb, irr_, load_, cfg_) in tasks:
            key = (round(float(pv), 6), round(float(eb), 6))
            if key not in unique:
                unique[key] = (pv, eb, irr_, load_, cfg_)
        tasks = list(unique.values())

        new_results = []
        if parallel:
            with Pool(processes=nprocs) as pool:
                for r in pool.imap_unordered(evaluate_grid_point, tasks):
                    new_results.append(r)
        else:
            for t in tasks:
                new_results.append(evaluate_grid_point(t))

        new_df = pd.DataFrame(new_results, columns=['PV_kWp', 'E_bess_kWh', 'npv', 'Feasible',
                                        'CAPEX', 'Assets_OPEX_by_year', 'Fuel_liters_hybrid_by_year', 'Fuel_liters_genonly_by_year',
                                        'Fuel_cost_hybrid', 'Fuel_cost_genonly',
                                        'SOC_end_by_year', 'Losses_by_year', 'Payback_yr', ])
        df = pd.concat([df, new_df], ignore_index=True)
        df_factible = df[df['Feasible'] == True]
        if df_factible.empty:
            best = None
        else:
            idx = df_factible['npv'].idxmax()
            best_row = df_factible.loc[idx]
            best = dict(best_row)

    # Enriquecer 'best' con métricas detalladas del simulador
    if best is not None:
        detailed = simulate_operation(best['PV_kWp'], best['E_bess_kWh'], irr_annual, load_annual, cfg)
        best['consumo_desde_pv'] = detailed.get('consumo_desde_pv', {})
        best['consumo_desde_bess'] = detailed.get('consumo_desde_bess', {})
        best['generación'] = detailed.get('generación', {})
        best['horas_generador_on'] = detailed.get('horas_generador_on', {})
        best['gross_savings'] = detailed.get('gross_savings', {})
        best['consumo_desde_genset'] = detailed.get('consumo_desde_genset', {})

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱ Tiempo total de optimización: {elapsed:.2f} segundos")
    return best, df
'''''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Supongamos que ya tienes df_grid
# Asegúrate de que las columnas se llamen así:
# 'PV_kWp', 'E_bess_kWh', 'npv'

# Pivot para tener una matriz PV x BESS
pivot_npv = df_grid.pivot_table(values='npv',
                                index='E_bess_kWh',
                                columns='PV_kWp')

plt.figure(figsize=(8,6))
plt.title("NPV en función de PV y BESS", fontsize=14)
plt.xlabel("PV [kWp]")
plt.ylabel("BESS [kWh]")

# cmap puede ser 'viridis', 'plasma', 'coolwarm', etc.
im = plt.imshow(pivot_npv, origin='lower',
                aspect='auto',
                cmap='viridis',
                extent=[df_grid['PV_kWp'].min(), df_grid['PV_kWp'].max(),
                        df_grid['E_bess_kWh'].min(), df_grid['E_bess_kWh'].max()])

plt.colorbar(im, label='NPV [USD]')
plt.tight_layout()
plt.show()
'''''