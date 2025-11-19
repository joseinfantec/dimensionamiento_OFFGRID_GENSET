import numpy as np
from funciones import interp_lph_from_curve

class SimulationConfig:
    def __init__(self,
                N_years=15,                    # Número de años de simulación
                r=0.07,                        # Tasa de descuento               
                ef_charge=0.95,                # Eficiencia de carga del BESS  
                ef_discharge=0.95,             # Eficiencia de descarga del BESS
                soc_max_frac = 0.95,
                soc_min_frac = 0.05,
                charge_rate=0.5,               # 0.5C Capacidad máxima de carga       
                discharge_rate=0.5,            # 0.5C Capacidad máxima de descarga 
                bess_capacity_factors=None,    # Lista de factores de capacidad del BESS por año (1.0 = sin degradación)                      
                pv_deg_rate=0.005,             # Degradación 0.5% anual

                C_pv_kWp=817309.0,             # Costo del kW en FV               
                C_bess_kWh=375000.5,           # Costo del kWh en BESS
                C_om_pv_kW_yr= 0,              # Costo operación y mantenimiento FV       
                C_om_bess_kWh_yr= 0,           # Costo operación y mantenimiento BESS          
                
                
                C_diesel_lt= 1100,             # Precio Diesel (CLP/lt)
                DG_performance_factors=None,   # Lista de factores de rendimiento para 25%, 50%, 75% y 100% de carga
                DG_power = 160,                # Potencia PRIME del generador
                DG_opex = 1100,                # Opex por hora de funcionamiento

                cpi = 0.02,                    # Indice CPI de la planilla de excel
                diesel_inflation = 0.02,       # Inflación del costo del diésel
                battery_replacement=None):     # Diccionario {año: costo por kWh} para reemplazo de batería en años específicos

        self.N_years = N_years
        self.r = r
        self.cpi = cpi
        self.diesel_inflation = diesel_inflation
        self.df_year = {y: 1.0 / ((1.0 + r) ** (y)) for y in range(1, N_years + 1)}                 # Factor de descuento por año
        
        self.charge_ef = ef_charge
        self.discharge_ef = ef_discharge
        self.soc_min_frac =soc_min_frac
        self.soc_max_frac = soc_max_frac
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.pv_deg_rate = pv_deg_rate
        self.bess_capacity_factors = bess_capacity_factors if bess_capacity_factors is not None else [1.0]*self.N_years
        self.deg_pv = {y: (1 - self.pv_deg_rate) ** (y) for y in range(1, self.N_years + 1)}

        self.DG_performance_factors = DG_performance_factors
        self.DG_power = DG_power
        

        self.C_diesel_lt = C_diesel_lt
        self.C_pv_kWp = C_pv_kWp
        self.C_bess_kWh = C_bess_kWh
        self.C_om_pv_kW_yr = C_om_pv_kW_yr
        self.C_om_bess_kWh_yr = C_om_bess_kWh_yr
        self.DG_opex = DG_opex

        self.battery_replacement = battery_replacement

def simulate_operation(PV_kWp, E_bess_kWh, irr_annual, load_annual, cfg: SimulationConfig, capture_day_of_january=None):
    hours_per_year = len(irr_annual)
    if len(load_annual) != hours_per_year:
        raise ValueError("irr_annual y load_annual deben tener igual longitud")
    load_total_year = sum(load_annual)


    capex = PV_kWp * cfg.C_pv_kWp + E_bess_kWh * cfg.C_bess_kWh
    feasible = True

    soc = cfg.soc_min_frac * E_bess_kWh * cfg.bess_capacity_factors[1]


    fuel_hybrid_by_year = {}
    fuel_genonly_by_year = {}
    fuel_cost_hybrid = {}
    fuel_cost_genonly = {}
    fuel_savings_cost = {}
    fuel_savings_cost_discounted = {}

    consumo_desde_genset_hybrid = {}
    PV_BESS_GEN_opex_by_year = {}
    soc_end_by_year = {}
    losses_by_year = {}
    consumo_desde_pv = {}
    consumo_desde_bess = {}
    generacion_por_año = {}
    gen_hours = {}
    gross_savings = {}
    net_savings_by_year = {}


    # Preparación de captura horaria (opcional) para un día de enero del año 1
    capture_hours_range = None
    hourly_capture = None
    if isinstance(capture_day_of_january, int) and 1 <= capture_day_of_january <= 100:
        start_h = (capture_day_of_january - 1) * 24
        end_h = start_h + 24
        capture_hours_range = (start_h, end_h)
        hourly_capture = {"load": [], "from_pv": [], "from_bess": [], "from_gen": [], "soc": [], "pv_gen": []}

    curve = {25: cfg.DG_performance_factors[0], 50: cfg.DG_performance_factors[1],
             75: cfg.DG_performance_factors[2], 100: cfg.DG_performance_factors[3]}

    for y in range(1, cfg.N_years + 1):
        degpv = cfg.deg_pv[y]
        bess_factor = cfg.bess_capacity_factors[y]

        pv_served = 0.0
        bess_served = 0.0
        gen_served = 0.0
        fuel_liters_year_hybrid = 0.0
        fuel_liters_year_genonly = 0.0
        losses_year = 0.0
        gen_hours_year = 0
        load_hours_year = 0
        generación_anual = 0.0
        

        hourly_charging_limit = E_bess_kWh * cfg.charge_rate
        hourly_discharging_limit = E_bess_kWh * cfg.discharge_rate

        for h in range(hours_per_year):
            irr = irr_annual[h]
            load = load_annual[h]
            if load > 0:
                load_hours_year += 1

            pv_gen = PV_kWp * irr * degpv  # Se divide en 1000 para pasar de W a kW
            generación_anual += pv_gen
            delivered = 0.0
            gen_kwh = 0.0
            #fuel = 0.0

            pv_to_load = min(pv_gen, load)
            pv_served += pv_to_load
            remaining_load = load - pv_to_load
            pv_excess = pv_gen - pv_to_load

            # Si hay carga remanente, intentar cubrirla con BESS
            if pv_excess > 1e-6:
                space = (E_bess_kWh * bess_factor * cfg.soc_max_frac) - soc
                needed_input_to_fill = space / cfg.charge_ef if cfg.charge_ef > 0 else 0.0
                can_charge = min(pv_excess, hourly_charging_limit/cfg.charge_ef, needed_input_to_fill)
                soc += can_charge * cfg.charge_ef
                losses_year += pv_excess - can_charge


            if remaining_load > 1e-6:
                soc_min = cfg.soc_min_frac * E_bess_kWh * bess_factor
                available_for_discharge = max(0.0, soc - soc_min)
                can_discharge = min(available_for_discharge, hourly_discharging_limit, remaining_load / cfg.discharge_ef)
                delivered = can_discharge * cfg.discharge_ef
                soc -= can_discharge
                remaining_load -= delivered
                bess_served += delivered


            if remaining_load > 1e-6:
                gen_kwh = remaining_load
                gen_served += gen_kwh
                percent_load = (gen_kwh / cfg.DG_power) * 100.0 if cfg.DG_power > 0 else 100.0
                percent_for_fuel = percent_load

                # interpolar litros/hora para el porcentaje dado
                lph = interp_lph_from_curve(percent_for_fuel, curve)
                fuel_liters_year_hybrid += lph  # 1 hora
                gen_hours_year += 1
                remaining_load = 0.0

                #fuel = remaining_load
                #fuel_consumed_year += fuel
                #remaining_load = 0.0
                #gen_hours_year += 1
            
            # Para escenario "solo generador": genset debe servir toda la carga 'load'
            if load > 1e-12:
                percent_only = (load / cfg.DG_power) * 100.0 if cfg.DG_power > 0 else 100.0
                if percent_only > 110.0:
                    raise ValueError("El tamaño del generador no es suficiente para suplir el consumo del caso solo genset.")
                else:
                    fuel_lph_only = interp_lph_from_curve(percent_only, curve)
                fuel_liters_year_genonly += fuel_lph_only


            # Captura horaria del día seleccionado (solo año 1)
            if y == 1 and capture_hours_range is not None and capture_hours_range[0] <= h < capture_hours_range[1]:
                hourly_capture["load"].append(load)
                hourly_capture["from_pv"].append(pv_to_load)
                hourly_capture["from_bess"].append(delivered)
                hourly_capture["from_gen"].append(gen_kwh)
                hourly_capture["soc"].append(soc)
                hourly_capture["pv_gen"].append(pv_gen)
            #    print("Consumo desde BESS ", delivered,". Consumo desde PV ", pv_to_load, "Consumo desde GEN ", fuel, ". SOC ", soc, ". Gen ", pv_gen)

            #if y==1 and h < 24:
            #    print("Consumo desde BESS ", delivered,". Consumo desde PV ", pv_to_load, "Consumo desde GEN ", gen_kwh, ". SOC ", soc, ". Gen ", pv_gen)


        soc_end_by_year[y] = soc
        losses_by_year[y] = losses_year
        fuel_hybrid_by_year[y] = round(float(fuel_liters_year_hybrid), 2)
        fuel_genonly_by_year[y] = round(float(fuel_liters_year_genonly), 2)

        price_year = cfg.C_diesel_lt * ((1 + cfg.diesel_inflation) ** (y))
        fuel_cost_hybrid[y] = round(float(fuel_liters_year_hybrid * price_year), 2)
        fuel_cost_genonly[y] = round(float(fuel_liters_year_genonly * price_year), 2)
        #served_by_clean = (load_total_year - fuel_consumed_year)  # en kWh
        #fuel_savings_year = served_by_clean * cfg.C_gen_kWh * ((1 + cfg.diesel_inflation) ** (y))

        #liters_saved = fuel_liters_year_genonly - fuel_liters_year_hybrid
        cost_saved = fuel_cost_genonly[y] - fuel_cost_hybrid[y]
        #fuel_savings_liters[y] = round(float(liters_saved), 2)
        fuel_savings_cost[y] = round(float(cost_saved), 2)
        fuel_savings_cost_discounted[y] = round(float(cost_saved * cfg.df_year[y]), 2)

        consumo_desde_pv[y] = round(float(pv_served), 2)
        consumo_desde_bess[y] = round(float(bess_served), 2)
        consumo_desde_genset_hybrid[y] = round(float(gen_served), 2)
        generacion_por_año[y] = round(float(generación_anual), 2)
        gen_hours[y] = gen_hours_year

        GEN_opex_year = cfg.DG_opex *(load_hours_year - gen_hours_year) * ((1 + cfg.cpi) ** (y))
        PV_BESS_opex = (cfg.C_om_pv_kW_yr + cfg.C_om_bess_kWh_yr) * ((1 + cfg.cpi) ** (y))
        PV_BESS_GEN_opex_by_year[y] = {"pv_bess": PV_BESS_opex,"gen": GEN_opex_year}

        gross_savings_year = cost_saved - PV_BESS_opex + GEN_opex_year
        gross_savings[y] = gross_savings_year
        net_savings_by_year[y] = (gross_savings_year) * cfg.df_year[y]

        #consumo_desde_genset_hybrid[y] = fuel_consumed_year
        #fuel_savings_by_year[y] = fuel_savings_year

        #if y < cfg.N_years:
        #    soc = min(soc, E_bess_kWh * cfg.bess_capacity_factors[y+1] * cfg.soc_max_frac)
       #if cfg.battery_replacement and (y in cfg.battery_replacement):
       #    repl_cost = cfg.battery_replacement[y] * E_bess_kWh

    # --- Conversiones a float nativo y redondeo a 2 decimales ---
    soc_end_by_year = {y: round(float(v), 2) for y, v in soc_end_by_year.items()}
    losses_by_year = {y: round(float(v), 2) for y, v in losses_by_year.items()}
    PV_BESS_GEN_opex_by_year = {
        y: {"pv_bess": round(float(vals["pv_bess"]), 2), "gen": round(float(vals["gen"]), 2)}
        for y, vals in PV_BESS_GEN_opex_by_year.items()
    }
    gross_savings = {y: round(float(v), 2) for y, v in gross_savings.items()}


    # --- NPV y Payback (descontado) ---
    npv = -capex + sum(net_savings_by_year.values())
    npv = round(float(npv), 2)

    cumulative = 0.0
    payback_year = None
    for y in range(1, cfg.N_years + 1):
        prev_cum = cumulative
        annual = net_savings_by_year[y]
        cumulative += annual
        if cumulative >= capex and annual > 0:
            remaining = capex - prev_cum
            frac = min(max(remaining / annual, 0.0), 1.0)
            payback_year = y - 1 + frac
            break

    results = {
        'capex': round(float(capex), 2),
        'npv': npv,
        'feasible': feasible,
        'fuel_hybrid_by_year': fuel_hybrid_by_year, # litros híbrido por año
        'fuel_genonly_by_year': fuel_genonly_by_year, # litros gen-only por año
        'fuel_cost_hybrid': fuel_cost_hybrid,
        'fuel_cost_genonly': fuel_cost_genonly,
        'fuel_savings_cost': fuel_savings_cost,
        #'fuel_savings_cost_discounted': fuel_savings_cost_discounted,
        'assets_opex_by_year': PV_BESS_GEN_opex_by_year,
        'soc_end_by_year': soc_end_by_year,
        'losses_by_year': losses_by_year,
        'consumo_desde_pv': consumo_desde_pv,
        'consumo_desde_bess': consumo_desde_bess,
        'consumo_desde_genset': consumo_desde_genset_hybrid,
        'generación': generacion_por_año,
        'horas_generador_on': gen_hours,
        'gross_savings': gross_savings,
        'payback_year': payback_year,
        'hourly_capture': hourly_capture
    }


    return results