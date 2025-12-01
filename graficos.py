import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


# ============================================================
# FORMATEADOR DE HORAS (usado por ambos gráficos)
# ============================================================
def hour_formatter(x, pos):
    """
    Convierte valores decimales en formato hh:mm respetando los límites.
    """
    if x < 0 or x > 23.999:
        return ""
    h = int(np.floor(x + 1e-9))
    m = int(round((x - h) * 60))
    if m == 60:
        h = min(h + 1, 23)
        m = 0
    return f"{h}:{m:02d}"


# ============================================================
# FUNCIÓN INTERNA DE INTERPOLACIÓN + SUAVIZADO
# ============================================================
def _smooth_interp(hours_fine, hours, y):
    y_f = np.interp(hours_fine, hours, y)
    window = 5
    if len(y_f) > window:
        kernel = np.ones(window) / window
        y_pad = np.pad(y_f, (window//2, window-1-window//2), mode='edge')
        y_smooth = np.convolve(y_pad, kernel, mode='valid')
        return y_smooth
    return y_f


# ============================================================
# GRÁFICO GENERAL (FV + BESS + GEN + CONSUMO)
# ============================================================
def graficar_desde_series(
    hourly_capture: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    y_lim: float = None
):
    """Grafica el balance energético diario con y_lim ya calculado en main.py."""
    
    if not hourly_capture:
        raise ValueError("hourly_capture vacío o None")

    load = np.array(hourly_capture["load"], dtype=float)
    from_pv = np.array(hourly_capture["from_pv"], dtype=float)
    from_bess = np.array(hourly_capture["from_bess"], dtype=float)
    from_gen = np.array(hourly_capture["from_gen"], dtype=float)
    pv_gen = np.array(hourly_capture["pv_gen"], dtype=float)
    supply_mode = str(hourly_capture.get("supply_mode", "genset")).lower()

    hours = np.arange(24, dtype=float)
    step = 1.0 / 12.0
    hours_fine = np.arange(0.0, 23.0001, step)

    load_f = _smooth_interp(hours_fine, hours, load)
    pv_f   = _smooth_interp(hours_fine, hours, from_pv)
    bess_f = _smooth_interp(hours_fine, hours, from_bess)
    gen_f  = _smooth_interp(hours_fine, hours, from_gen)
    pv_genf= _smooth_interp(hours_fine, hours, pv_gen)

    date_str = hourly_capture.get("date_str", "XX/XX/XXXX")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(hours_fine, load_f, label="Consumo", color="#726d5f", linewidth=6.0)
    ax.plot(hours_fine, pv_f,   label="Consumo desde FV",    color="#1b801b", linewidth=2.0)
    ax.plot(hours_fine, bess_f, label="Consumo desde BESS",  color="#2046c2", linewidth=2.0)
    if supply_mode == "genset":
        ax.plot(hours_fine, gen_f,  label="Consumo desde Generador", color="#d62728", linewidth=2.0)
    if supply_mode == "grid":
        ax.plot(hours_fine, gen_f,  label="Consumo desde Red", color="#d62728", linewidth=2.0)
    ax.plot(hours_fine, pv_genf,label="Generación FV",       color="#f2a900", linewidth=2.0)

    ax.fill_between(hours_fine, 0, load_f, color="#726d5f",  alpha=0.18)
    ax.fill_between(hours_fine, 0, pv_f,   color="#1b801b",  alpha=0.18)
    ax.fill_between(hours_fine, 0, bess_f, color="#2046c2",  alpha=0.18)
    ax.fill_between(hours_fine, 0, gen_f,  color="#d62728",  alpha=0.18)
    ax.fill_between(hours_fine, 0, pv_genf,color="#f2a900",  alpha=0.18)

    ax.xaxis.set_major_locator(MultipleLocator(4.0))
    ax.xaxis.set_major_formatter(FuncFormatter(hour_formatter))

    ax.set_title(f"Balance Energético Diario {date_str}", fontsize=13)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Energía [kWh]")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, y_lim)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    
    fig.tight_layout()
    return fig, ax


# ============================================================
# GRÁFICO SOLO GENERADOR (MISMO EJE X Y Y_LIM QUE EL OTRO)
# ============================================================
def graficar_generador_solo_desde_series(
    hourly_capture: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    y_lim: float = None
):
    load = np.array(hourly_capture["load"], dtype=float)
    supply_mode = str(hourly_capture.get("supply_mode", "genset")).lower()    
    hours = np.arange(24, dtype=float)
    step = 1.0 / 12.0
    hours_fine = np.arange(0.0, 23.0001, step)

    load_f = _smooth_interp(hours_fine, hours, load)

    date_str = hourly_capture.get("date_str", "XX/XX/XXXX")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(hours_fine, load_f, label="Consumo", color="#726d5f", linewidth=4)
    if supply_mode == "genset":
        ax.plot(hours_fine, load_f, label="Consumo desde Generador", color="#d62728", linewidth=2.5)
    if supply_mode == "grid":
        ax.plot(hours_fine, load_f, label="Consumo desde Red", color="#d62728", linewidth=2.5)

    ax.fill_between(hours_fine, 0, load_f, color="#726d5f", alpha=0.2)
    ax.fill_between(hours_fine, 0, load_f, color="#d62728", alpha=0.15)

    ax.xaxis.set_major_locator(MultipleLocator(4.0))
    ax.xaxis.set_major_formatter(FuncFormatter(hour_formatter))

    if supply_mode == "grid":
        ax.set_title(f"Sólo Red Eléctrica — {date_str}", fontsize=13)
    if supply_mode == "genset":
        ax.set_title(f"Sólo Generador — {date_str}", fontsize=13)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Energía [kWh]")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, y_lim)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.tight_layout()
    return fig, ax
