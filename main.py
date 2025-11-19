from data_loader import read_irradiation_from_excel, expand_monthly_matrix_to_annual_hourly, read_load_hourly_from_excel
from simulator import SimulationConfig, simulate_operation
from graficos import graficar_desde_series
from optimizer import grid_search_optimize
from milp import milp_optimize
from funciones import *

# ===========================
# Carga de datos
# ===========================
path = r"C:\Users\jinfa\OneDrive\Desktop\Versión_Final_Clientes_OFFGRID.xlsm"
sheet_name = "Gen_Cons_Horario"

# Cargar matriz mensual de irradiación (24h x 12 meses)
mat_24x12 = read_irradiation_from_excel(path, sheet_name=sheet_name)

# Expandir a perfil horario anual (8760 horas)
irr_8760 = expand_monthly_matrix_to_annual_hourly(mat_24x12)

# Cargar consumo horario desde columna B (índice 1)
load_8760 = read_load_hourly_from_excel(path, sheet_name=sheet_name)

# ===========================
# Configuración del sistema
# ===========================
cfg = SimulationConfig(
    N_years=15,
    r=0.07,
    ef_charge=0.99,
    ef_discharge=0.99,
    soc_max_frac = 0.95,
    soc_min_frac = 0.05,
    charge_rate=0.9,
    discharge_rate=0.9,
    pv_deg_rate=0.0045,
    C_pv_kWp=813701,
    C_bess_kWh=327250,
    C_diesel_lt=1100,
    C_om_pv_kW_yr=0,
    C_om_bess_kWh_yr=0,
    cpi=0.02,
    diesel_inflation=0.02,
    #bess_capacity_factors=[1,0.9488,0.9168,0.8895,0.8651,0.8426,0.8217,0.8020,0.7834,0.7657,0.7488,0.7326,0.7171,0.7021,0.6875,0.6730,0.6584,0.6437,0.6290,0.6143,0.6000], #Huawei
    bess_capacity_factors=[1,0.9564,0.9257,0.9057,0.8863,0.8675,0.8494,0.8317,0.8149,0.7980,0.7824,0.7672,0.7521,0.7373,0.7228,0.7075,0.6925,0.6777,0.6631,0.6495,0.6377], #Sigenergy   
    DG_performance_factors=[11, 22.1, 31.3, 37.9],
    DG_power=120,            #Potencia Prime
    DG_opex = 1100,         #Opex popr hora de funcionamiento
    )

# ===========================
# Simulación de operación (ejemplo)
# ===========================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   

    PV_test =  0  # kWp 211
    E_test = 0  # kWh 60.3

    # Capturar día 30 de enero durante la simulación principal
    sim_results = simulate_operation(PV_test, E_test, irr_8760, load_8760, cfg, capture_day_of_january=40)
    #print("===== Para PV = ", PV_test, "kWp y BESS =", E_test, "kWh =====")
    #print_results("Resultados de simulación ejemplo", sim_results)


# ===========================
# Graficos
# ===========================
# Pedir captura del día 30 de enero durante la simulación
# Graficar directamente las series capturadas (día 30)
    hourly = sim_results.get('hourly_capture')
    if hourly:
        import matplotlib.pyplot as plt
        graficar_desde_series(hourly)
        plt.show()
'''''
# ===========================
# Optimización Grid Search
# ===========================

    best_grid, df_grid = grid_search_optimize(
        irr_8760, load_8760, cfg,
        PV_range=(100, 260),
        E_range=(40, 100),
        nPV=15,
        nE=15,
        parallel=True,
        nprocs=4
    )

    if best_grid is not None:
        print_results_reducidos("Mejor PV+BESS (Grid Search)", best_grid)
    else:
        print("No se encontró una solución factible en Grid Search.")

# ===========================
# Optimización MILP
# ===========================

PV_options = list(range(50, 201, 10))
E_options = list(range(40, 300, 50))

best_pv, best_e, best_res = milp_optimize(
    irr_annual=irr_8760,
    load_annual=load_8760,
    cfg=cfg,
    PV_options=PV_options,
    E_options=E_options
)

print("\n--- Mejor PV+BESS (MILP) ---")
print(f"PV: {best_pv} kWp, BESS: {best_e} kWh")
print(f"NPV: {best_res['npv']:.2f}")
'''''