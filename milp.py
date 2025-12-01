# --- al principio del archivo (mantener imports existentes) ---
import numpy as np
from funciones import interp_lph_from_curve  # si lo usas para fuel

# ... SimulationConfig class queda igual, añade opcionalmente supply defaults:
# dentro de __init__: self.supply_mode = supply_mode(optional) no obligatorio

def simulate_operation(PV_kWp, E_bess_kWh, irr_annual, load_annual, cfg: SimulationConfig,
                       capture_day_of_january=None, supply_mode: str = None):
    """
    simulate_operation unificado para 'genset' y 'grid'.
    supply_mode: 'genset' (default) or 'grid'. Si es None se intenta leer cfg.supply_mode.
    """
    # decidir modo
    mode = supply_mode if supply_mode is not None else getattr(cfg, "supply_mode", "genset")
    mode = mode.lower()
    if mode not in ("genset", "grid"):
        mode = "genset"

    hours_per_year = len(irr_annual)
    if len(load_annual) != hours_per_year:
        raise ValueError("irr_annual y load_annual deben tener igual longitud")
    load_total_year = sum(load_annual)

    capex = PV_kWp * cfg.C_pv_kWp + E_bess_kWh * getattr(cfg, "C_bess_kWh", 0.0)
    feasible = True

    # SOC inicial (como ya tenías)
    soc = cfg.soc_min_frac * E_bess_kWh * cfg.bess_capacity_factors[1]

    # métricas comunes
    supply_kwh_by_year = {}       # kWh provistos por la supply (genset o grid) por año
    fuel_hybrid_by_year = {}      # solo meaningful para genset
    fuel_genonly_by_year = {}
    fuel_cost_hybrid = {}
    fuel_cost_genonly = {}
    consumo_desde_pv = {}
    consumo_desde_bess = {}
    generacion_por_año = {}
    horas_generador_on = {}
    losses_by_year = {}
    PV_BESS_GEN_opex_by_year = {}
    gross_savings = {}
    net_savings_by_year = {}

    # captura horaria
    capture_hours_range = None
    hourly_capture = None
    if isinstance(capture_day_of_january, int) and 1 <= capture_day_of_january <= 365:
        start_h = (capture_day_of_january - 1) * 24
        end_h = start_h + 24
        capture_hours_range = (start_h, end_h)
        hourly_capture = {"load": [], "from_pv": [], "from_bess": [], "from_gen": [], "soc": [], "pv_gen": []}

    # curva generador si aplica
    curve = None
    if mode == "genset":
        # espera que cfg.DG_performance_factors exista
        curve = {25: cfg.DG_performance_factors[0], 50: cfg.DG_performance_factors[1],
                 75: cfg.DG_performance_factors[2], 100: cfg.DG_performance_factors[3]}

    for y in range(1, cfg.N_years + 1):
        degpv = cfg.deg_pv[y]
        bess_factor = cfg.bess_capacity_factors[y]

        pv_served = 0.0
        bess_served = 0.0
        supply_served = 0.0   # gen_kwh or grid kWh
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

            pv_gen = PV_kWp * irr * degpv
            generación_anual += pv_gen
            delivered = 0.0
            supply_kwh = 0.0

            pv_to_load = min(pv_gen, load)
            pv_served += pv_to_load
            remaining_load = load - pv_to_load
            pv_excess = pv_gen - pv_to_load

            # carga BESS con exceso PV
            if pv_excess > 1e-9:
                space = (E_bess_kWh * bess_factor * cfg.soc_max_frac) - soc
                needed_input_to_fill = space / cfg.charge_ef if cfg.charge_ef > 0 else 0.0
                can_charge = min(pv_excess, hourly_charging_limit / cfg.charge_ef, needed_input_to_fill)
                soc += can_charge * cfg.charge_ef
                losses_year += pv_excess - can_charge

            # descarga BESS a carga restante
            if remaining_load > 1e-9:
                soc_min = cfg.soc_min_frac * E_bess_kWh * bess_factor
                available_for_discharge = max(0.0, soc - soc_min)
                can_discharge = min(available_for_discharge, hourly_discharging_limit, remaining_load / cfg.discharge_ef)
                delivered = can_discharge * cfg.discharge_ef
                soc -= can_discharge
                remaining_load -= delivered
                bess_served += delivered

            # si sigue carga, la provee supply (genset o grid)
            if remaining_load > 1e-9:
                supply_kwh = remaining_load
                supply_served += supply_kwh
                if mode == "genset":
                    # fuel/lph con curva: lph para percent carga
                    percent_load = (supply_kwh / cfg.DG_power) * 100.0 if cfg.DG_power > 0 else 100.0
                    lph = interp_lph_from_curve(percent_load, curve)
                    fuel_liters_year_hybrid += lph
                    gen_hours_year += 1
                else:
                    # grid: simplemente contabilizar kWh
                    gen_hours_year += 1
                remaining_load = 0.0

            # captura horaria (solo año 1)
            if y == 1 and capture_hours_range is not None and capture_hours_range[0] <= h < capture_hours_range[1]:
                hourly_capture["load"].append(load)
                hourly_capture["from_pv"].append(pv_to_load)
                hourly_capture["from_bess"].append(delivered)
                # from_gen guarda la energía suministrada por la "supply" (gen o grid)
                hourly_capture["from_gen"].append(supply_kwh)
                hourly_capture["soc"].append(soc)
                hourly_capture["pv_gen"].append(pv_gen)

        # fin bucle horas
        # guardar por año
        supply_kwh_by_year[y] = round(float(supply_served), 2)
        fuel_hybrid_by_year[y] = round(float(fuel_liters_year_hybrid), 2) if mode == "genset" else 0.0
        fuel_genonly_by_year[y] = round(float(fuel_liters_year_genonly), 2) if mode == "genset" else 0.0
        consumo_desde_pv[y] = round(float(pv_served), 2)
        consumo_desde_bess[y] = round(float(bess_served), 2)
        generacion_por_año[y] = round(float(generación_anual), 2)
        horas_generador_on[y] = gen_hours_year
        losses_by_year[y] = round(float(losses_year), 2)

        # opex y savings
        price_year = (cfg.C_diesel_lt if mode == "genset" else cfg.C_grid_kWh) * ((1 + (cfg.diesel_inflation if mode == "genset" else cfg.grid_inflation)) ** (y))
        if mode == "genset":
            fuel_cost_hybrid[y] = round(float(fuel_liters_year_hybrid * price_year), 2)
            fuel_cost_genonly[y] = round(float(fuel_liters_year_genonly * price_year), 2)
            cost_saved = (fuel_cost_genonly[y] - fuel_cost_hybrid[y])
        else:
            # grid: interpretamos grid_savings como lo que se *deja de gastar* por servir con PV/BESS
            served_by_clean = (load_total_year - supply_kwh_by_year[y])
            grid_savings_year = served_by_clean * cfg.C_grid_kWh * ((1 + cfg.grid_inflation) ** (y))
            cost_saved = grid_savings_year

        # PV/BESS OPEX
        PV_BESS_opex = (cfg.C_om_pv_kW_yr + cfg.C_om_bess_kWh_yr) * ((1 + cfg.cpi) ** (y))
        gross_savings[y] = (cost_saved - PV_BESS_opex)
        net_savings_by_year[y] = gross_savings[y] * cfg.df_year[y]

        # ajustar soc para el siguiente año (si corresponde)
        if y < cfg.N_years:
            soc = min(soc, E_bess_kWh * cfg.bess_capacity_factors[y+1] * cfg.soc_max_frac)

    # fin bucle años

    # calcular NPV y payback
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

    # fracción supply por año (supply_kwh_by_year / total consumo anual)
    supply_fraction_real = {y: (supply_kwh_by_year[y] / load_total_year) if load_total_year > 0 else 0.0 for y in supply_kwh_by_year.keys()}

    results = {
        'capex': round(float(capex), 2),
        'npv': npv,
        'feasible': feasible,
        'supply_kwh_by_year': supply_kwh_by_year,   # kWh de genset o grid por año
        'supply_fraction_real': supply_fraction_real,
        'fuel_hybrid_by_year': fuel_hybrid_by_year,
        'fuel_genonly_by_year': fuel_genonly_by_year,
        'fuel_cost_hybrid': fuel_cost_hybrid,
        'fuel_cost_genonly': fuel_cost_genonly,
        'assets_opex_by_year': PV_BESS_GEN_opex_by_year,
        'soc_end_by_year': {y: round(float(v),2) for y, v in enumerate([0]*cfg.N_years, start=1)}, # o usa soc_end_by_year que ya tenías
        'losses_by_year': losses_by_year,
        'consumo_desde_pv': consumo_desde_pv,
        'consumo_desde_bess': consumo_desde_bess,
        'generacion': generacion_por_año,
        'horas_generador_on': horas_generador_on,
        'gross_savings': gross_savings,
        'payback_year': payback_year,
        'hourly_capture': hourly_capture,
        'supply_type': mode
    }

    return results
