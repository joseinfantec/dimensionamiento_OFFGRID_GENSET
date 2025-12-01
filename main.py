from data_loader import read_irradiation_from_excel, expand_monthly_matrix_to_annual_hourly, read_load_hourly_from_excel
from simulator import SimulationConfig, simulate_operation
from graficos import graficar_desde_series, graficar_generador_solo_desde_series
from optimizer import grid_search_optimize
from funciones import *
import os
import tempfile
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import xlwings as xw
import pandas as pd
import datetime

# -----------------------------
#  Funciones de integración Excel
# -----------------------------


# Ruta por defecto (usa el archivo subido si ejecutas desde IDE/terminal)
DEFAULT_EXCEL_PATH = r"C:\Users\jinfa\Downloads\Proyecto_SanFranciscoCHIUCHIU.xlsm"

def get_excel_path():
    """
    Intenta detectar el path del libro desde donde se lanzó (xlwings Book.caller()).
    Si no está disponible (ejecución desde terminal), devuelve DEFAULT_EXCEL_PATH.
    """
    try:
        wb = xw.Book.caller()
        path = wb.fullname
        if path is None or path == "":
            raise Exception("Book.caller() no devolvió fullname")
        return path
    except Exception:
        # fallback: si la variable de entorno XL_PATH está definida, úsala
        env_path = os.environ.get("XLWINGS_EXCEL_PATH")
        if env_path:
            return env_path
        # fallback final: DEFAULT_EXCEL_PATH
        return DEFAULT_EXCEL_PATH


def load_config_from_excel(path: str = None) -> SimulationConfig:
    """
    Lee desde la planilla (SFV+BESS y Financiera_GenSet) todos los parámetros
    necesarios y devuelve un objeto SimulationConfig.
    """
    if path is None:
        path = get_excel_path()

    # Abrir con xlwings (esto funciona si Excel ya está abierto o cuando hay ruta)
    wb = None
    try:
        wb = xw.Book(path)
    except Exception:
        # si Book(path) falla, intentar Book.caller()
        try:
            wb = xw.Book.caller()
        except Exception:
            raise RuntimeError(f"No se pudo abrir el libro: {path}")

    sh_sfv = wb.sheets["SFV+BESS"]
    sh_fin = wb.sheets["Financiera_GenSet"]
    sh_fin_grid = wb.sheets["Financiera_Grid"]
    #sh_cuadro_resumen = wb.sheets["Cuadro Resumen"]

    N_years = int(sh_sfv.range("I3").value)

    ef_charge = float(sh_sfv.range("I8").value)
    ef_discharge = float(sh_sfv.range("I9").value)

    soc_max_frac = float(sh_sfv.range("I12").value)
    soc_min_frac = float(sh_sfv.range("I11").value)

    charge_rate = float(sh_sfv.range("I13").value)
    discharge_rate = float(sh_sfv.range("I13").value) 

    pv_deg_rate = float(sh_sfv.range("D8").value)

    C_diesel_lt = float(sh_fin.range("K3").value)

    C_om_pv_kW_yr = float(sh_fin.range("G8").value)
    C_om_bess_kWh_yr = float(sh_fin.range("G9").value)

    cpi = float(sh_fin.range("K9").value)
    diesel_inflation = float(sh_fin.range("K4").value)

    DG_power = float(sh_sfv.range("Q10").value)
    DG_opex = float(sh_fin.range("K2").value)

    mode_cell = sh_sfv.range("J27").value
    if mode_cell is None:
        supply_mode = "genset"
    supply_mode = str(mode_cell).lower()

    if supply_mode == "grid":
        try:
            C_grid_kWh = float(sh_fin_grid.range("K2").value)
            C_pv_kWp = float(sh_fin_grid.range("G5").value)
            C_bess_kWh = float(sh_fin_grid.range("G3").value)
            r = float(sh_fin_grid.range("C10").value)

        except Exception:
            # fallback: si no existe, dejar valor por defecto en cfg
            pass
    if supply_mode == "genset":
        C_pv_kWp = float(sh_fin.range("G5").value)
        C_bess_kWh = float(sh_fin.range("G3").value)
        r = float(sh_fin.range("C10").value)
        C_grid_kWh = None



    # bess_capacity_factors: M4:M24 (21 valores)
    try:
        df_bess = pd.read_excel(path, sheet_name="SFV+BESS", usecols="M", skiprows=3, nrows=21, header=None, engine="openpyxl")
        bess_capacity_factors = df_bess.iloc[:,0].dropna().astype(float).tolist()
    except Exception:
        bess_capacity_factors = [1.0] * N_years
        print("Advertencia: no se pudieron leer los factores de capacidad BESS, se usarán valores por defecto de 1.0") 
    try:
        df_dg = pd.read_excel(path, sheet_name="SFV+BESS", usecols="Q", skiprows=12, nrows=4, header=None, engine="openpyxl")
        DG_performance_factors = df_dg.iloc[:,0].dropna().astype(float).tolist()
    except Exception:
        DG_performance_factors = [0.0, 0.0, 0.0, 0.0]

    # Construir el objeto SimulationConfig con los valores leídos
    cfg = SimulationConfig(
        N_years = N_years,
        r = r,
        ef_charge = ef_charge,
        ef_discharge = ef_discharge,
        soc_max_frac = soc_max_frac,
        soc_min_frac = soc_min_frac,
        charge_rate = charge_rate,
        discharge_rate = discharge_rate,
        pv_deg_rate = pv_deg_rate,
        C_pv_kWp = C_pv_kWp,
        C_bess_kWh = C_bess_kWh,
        C_diesel_lt = C_diesel_lt,
        C_om_pv_kW_yr = C_om_pv_kW_yr,
        C_om_bess_kWh_yr = C_om_bess_kWh_yr,
        cpi = cpi,
        diesel_inflation = diesel_inflation,
        bess_capacity_factors = bess_capacity_factors,
        DG_performance_factors = DG_performance_factors,
        DG_power = DG_power,
        DG_opex = DG_opex,
        supply_mode = supply_mode,
        C_grid_kWh = C_grid_kWh
    )

    return cfg


def run_optimizer(pv_min: float = None, pv_max: float = None,
                  e_min: float = None, e_max: float = None,
                  nPV: int = 15, nE: int = 15,parallel: bool = True,
                  nprocs: int = 4, refine_steps: int = 2):
    path = get_excel_path()
    wb = xw.Book(path) if path else xw.Book.caller()
    sh_sfv = wb.sheets["SFV+BESS"]
    sh_gen = wb.sheets["Gen_Cons_Horario"]
    sh_cuadro_resumen = wb.sheets["Cuadro Resumen"]

    # Nota: data_loader.read_load_hourly_from_excel y read_irradiation_from_excel esperan path y sheet_name
    mat_24x12 = read_irradiation_from_excel(path, sheet_name="Gen_Cons_Horario")
    irr_8760 = expand_monthly_matrix_to_annual_hourly(mat_24x12)
    load_8760 = read_load_hourly_from_excel(path, sheet_name="Gen_Cons_Horario")

    # cargar configuración
    cfg = load_config_from_excel(path)

    # leer PV/E ranges desde celdas si no fueron entregadas
    if pv_min is None:
        try:
            pv_min = float(sh_sfv.range("J29").value)
        except Exception:
            raise ValueError("No se pudo leer PV mínimo desde la celda J29.")
    if pv_max is None:
        try:
            pv_max = float(sh_sfv.range("J30").value)
        except Exception:
            raise ValueError("No se pudo leer PV máximo desde la celda J30.")
    if e_min is None:
        try:
            e_min = float(sh_sfv.range("J31").value)
        except Exception:
            raise ValueError("No se pudo leer BESS mínimo desde la celda J31.")
    if e_max is None:
        try:
            e_max = float(sh_sfv.range("J32").value)
        except Exception:
            raise ValueError("No se pudo leer BESS máximo desde la celda J32.")
    if 1 == 1:
        if sh_sfv.range("J33").value is None:
            gen_fraction_limit = 1.0
        else:
            gen_fraction_limit = float(sh_sfv.range("J33").value)
    if 1 == 1:
        if sh_sfv.range("J34").value is None:
            asegurar_año = 1.0
        else:
            asegurar_año = float(sh_sfv.range("J34").value)
    
    # Ejecutar optimizador (grid_search por defecto)
    best, df = grid_search_optimize(
        irr_8760, load_8760, cfg,
        PV_range=(pv_min, pv_max),
        E_range=(e_min, e_max),
        nPV=nPV,
        nE=nE,
        parallel=parallel,
        nprocs=nprocs,
        refine_steps=refine_steps,
        gen_fraction_limit=gen_fraction_limit,
        asegurar_año=asegurar_año
    )

    if best is not None:
        # pv y bess óptimos
        pv_opt = best.get('PV_kWp')
        bess_opt = best.get('E_bess_kWh')
        npv_opt = best.get('npv')
        generacion = best.get('generacion')
        consumo_desde_pv = best.get('consumo_desde_pv')
        consumo_desde_bess = best.get('consumo_desde_bess')
        consumo_desde_genset = best.get('consumo_desde_genset')
        fuel_hybrid_by_year = best.get('fuel_hybrid_by_year')
        tiempo_transcurrido_optimizacion = best.get('tiempo_transcurrido_optimizacion')
        fraccion_generador = best.get('gen_fraction_real')
        # Escribir en celdas
        sh_sfv.range("J37").value = pv_opt
        sh_sfv.range("J38").value = bess_opt
        sh_sfv.range("J39").value = npv_opt
        sh_sfv.range("J40").value = fraccion_generador[1]
        sh_cuadro_resumen.range("P4").value = generacion[1]
        sh_cuadro_resumen.range("Q4").value = consumo_desde_pv[1]
        sh_cuadro_resumen.range("R4").value = consumo_desde_bess[1]
        sh_cuadro_resumen.range("S4").value = consumo_desde_genset[1]

    else:
        pv_opt = None
        bess_opt = None
        npv_opt = None
        fraccion_generador = None
        generacion = {}
        consumo_desde_pv = {}
        consumo_desde_bess = {}
        consumo_desde_genset = {}
        fuel_hybrid_by_year = {}
        tiempo_transcurrido_optimizacion = None

        # Escribir en celdas
        sh_sfv.range("J37").value = "error"
        sh_sfv.range("J38").value = "error"
        sh_sfv.range("J39").value = "error"
        sh_sfv.range("J40").value = "error"
        sh_cuadro_resumen.range("P4").value = "error"
        sh_cuadro_resumen.range("Q4").value = "error"
        sh_cuadro_resumen.range("R4").value = "error"
        sh_cuadro_resumen.range("S4").value = "error"
    

    return {"pv_opt": pv_opt, "bess_opt": bess_opt, "npv": npv_opt, "generacion": generacion,
            "consumo_desde_pv": consumo_desde_pv,"consumo_desde_bess": consumo_desde_bess,
            "consumo_desde_genset": consumo_desde_genset,"fuel_hybrid_by_year": fuel_hybrid_by_year,
            "tiempo_transcurrido_optimizacion": tiempo_transcurrido_optimizacion}

# ----------------------------
# Funciones para graficar desde Excel sin correr optimizador
# ----------------------------

def run_plot_simulation(day_of_year: int, pv: float = None, bess: float = None, path: str = None):
    # si path es None, usamos get_excel_path() para detectar el libro actual
    if path is None:
        path = get_excel_path()

    # leer perfiles y configuración
    mat_24x12 = read_irradiation_from_excel(path, sheet_name="Gen_Cons_Horario")
    irr_8760 = expand_monthly_matrix_to_annual_hourly(mat_24x12)
    load_8760 = read_load_hourly_from_excel(path, sheet_name="Gen_Cons_Horario")
    cfg = load_config_from_excel(path)

    # Si pv/bess no entregados, leer desde hoja (J36/J37)
    wb = xw.Book(path)
    sh_sfv = wb.sheets["SFV+BESS"]
    try:
        if pv is None:
            pv_cell = sh_sfv.range("D5").value
            pv = float(pv_cell) if pv_cell is not None else 0.0
    except Exception:
        pv = 0.0

    try:
        if bess is None:
            bess_cell = sh_sfv.range("I7").value
            bess = float(bess_cell) if bess_cell is not None else 0.0
    except Exception:
        bess = 0.0

    # correr simulación con captura del día solicitado
    sim = simulate_operation(pv, bess, irr_8760, load_8760, cfg, capture_day_of_january=int(day_of_year))
    hourly = sim.get("hourly_capture")

    base_date = datetime.date(datetime.date.today().year, 1, 1)
    real_date = base_date + datetime.timedelta(days=day_of_year - 1)

    hourly['day_of_year'] = day_of_year
    hourly['date_str'] = real_date.strftime("%d/%m/%Y")
    hourly['supply_mode'] = cfg.supply_mode.lower()

    load = np.array(hourly["load"], dtype=float)
    from_pv = np.array(hourly["from_pv"], dtype=float)
    from_bess = np.array(hourly["from_bess"], dtype=float)
    from_gen = np.array(hourly["from_gen"], dtype=float)
    pv_gen = np.array(hourly["pv_gen"], dtype=float)

    y_max_data = max(load.max(), from_pv.max(), from_bess.max(), from_gen.max(), pv_gen.max())

    if y_max_data <= 20:
        y_lim = y_max_data + 5
    elif y_max_data <= 100:
        y_lim = y_max_data * 1.10
    else:
        y_lim = y_max_data * 1.15

##### DEBUG: escribir máximos en celdas I60:I64
    #wb = xw.Book.caller()
    #sh = wb.sheets["SFV+BESS"]
    #sh.range("I60").value = load.max()
    #sh.range("I61").value = from_pv.max()
    #sh.range("I62").value = from_bess.max()
    #sh.range("I63").value = from_gen.max()
    #sh.range("I64").value = pv_gen.max()

    return hourly, y_lim

def save_hourly_plot_to_png(hourly_capture: dict, filename: str, y_lim: float = None):
    if hourly_capture is None:
        raise ValueError("hourly_capture es None. Ejecute la simulación con capture_day_of_january.")
    fig, ax = graficar_desde_series(hourly_capture, y_lim=y_lim)
    # asegurar carpeta destino existe
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return filename

def build_generator_only_series(hourly):
    """
    Construye series para el caso sin PV ni BESS: 
    todo el consumo es suplido por el generador.
    """
    hourly_gen = {
        "load": hourly["load"],
        "gen": hourly["load"],     # 100% del consumo
        "date_str": hourly["date_str"],
        "day_of_year": hourly["day_of_year"],
        "supply_mode": hourly["supply_mode"]
    }
    return hourly_gen

def save_generator_only_plot_to_png(hourly_gen, filename, y_lim=None):
    fig, ax = graficar_generador_solo_desde_series(hourly_gen, y_lim=y_lim)
    fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   

    mat_24x12 = read_irradiation_from_excel(DEFAULT_EXCEL_PATH, sheet_name="Gen_Cons_Horario")
    irr_8760 = expand_monthly_matrix_to_annual_hourly(mat_24x12)
    load_8760 = read_load_hourly_from_excel(DEFAULT_EXCEL_PATH, sheet_name="Gen_Cons_Horario")
    cfg = load_config_from_excel(DEFAULT_EXCEL_PATH)

    PV_test =  588  # kWp 211
    E_test = 1821  # kWh 60.3

    # Capturar día 30 de enero durante la simulación principal
    sim_results = simulate_operation(PV_test, E_test, irr_8760, load_8760, cfg, capture_day_of_january=40)
    print("===== Para PV = ", PV_test, "kWp y BESS =", E_test, "kWh =====")
    print_results("Resultados de simulación ejemplo", sim_results)

'''
# ===========================
# Carga de datos
# ===========================
#path = "C:\\Users\\jinfa\\OneDrive\\Desktop\\Version_Final_Clientes_OFFGRID.xlsm"
#sheet_name = "Gen_Cons_Horario"

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
'''