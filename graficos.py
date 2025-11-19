import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from simulator import SimulationConfig

'''''
def graficar_dia_enero_ano1(
    series: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
):
    """Grafica el consumo horario (24h) para un día dado, usando las series
    ya calculadas externamente por el simulador. Espera claves:
    'load', 'from_pv', 'from_bess', 'from_gen'. Devuelve (fig, ax).
    """
    load = np.array(series["load"], dtype=float)
    from_pv = np.array(series["from_pv"], dtype=float)
    from_bess = np.array(series["from_bess"], dtype=float)
    from_gen = np.array(series["from_gen"], dtype=float)
    pv_gen = np.array(series.get("pv_gen", [0.0]*24), dtype=float)

    hours = np.arange(24, dtype=float)
    step = 1.0 / 12.0
    hours_fine = np.arange(0.0, 23.0 + step, step)

    def interp(y):
        y_f = np.interp(hours_fine, hours, y)
        window = 5
        if len(y_f) > window:
            kernel = np.ones(window) / window
            y_pad = np.pad(y_f, (window//2, window-1-window//2), mode='edge')
            y_smooth = np.convolve(y_pad, kernel, mode='valid')
            return y_smooth
        return y_f

    load_f = interp(load)
    pv_f = interp(from_pv)
    bess_f = interp(from_bess)
    gen_f = interp(from_gen)
    pv_genf = interp(pv_gen)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hours_fine, load_f, label="Consumo", color="#f2a900", linewidth=5.0)
    ax.plot(hours_fine, pv_f, label="Desde FV", color="#1b801b", linewidth=2.0)
    ax.plot(hours_fine, bess_f, label="Desde BESS", color="#2046c2", linewidth=2.0)
    ax.plot(hours_fine, gen_f, label="Desde Generador", color="#d62728", linewidth=2.0)
    ax.plot(hours_fine, pv_genf, label="Generación FV", color="#568c00", linewidth=2.0)

    ax.fill_between(hours_fine, 0, load_f, color="#f2a900", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, pv_f, color="#1b801b", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, bess_f, color="#2046c2", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, gen_f, color="#d62728", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, pv_genf, color="#568c00", alpha=0.18, linewidth=0)

    def hour_formatter(x, pos):
        if x < 0 or x > 23.999:
            return ""
        h = int(np.floor(x + 1e-9))
        m = int(round((x - h) * 60))
        if m == 60:
            h = min(h + 1, 23)
            m = 0
        return f"{h}:{m:02d}"

    ax.xaxis.set_major_locator(MultipleLocator(4.0))
    ax.xaxis.set_major_formatter(FuncFormatter(hour_formatter))

    ax.set_title("Balance Energético Diario 30/01/2026: Consumo y Fuentes de Suministro")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Energía [kWh]")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 120)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.8)
    fig.tight_layout()

    return fig, ax
'''''

def graficar_desde_series(
    hourly_capture: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
):
    """Grafica directamente desde las series capturadas en simulator:
    claves esperadas: 'load', 'from_pv', 'from_bess', 'from_gen'.
    Aplica el mismo estilo y formato (suavizado, colores, rellenos, ejes).
    """
    if not hourly_capture:
        raise ValueError("hourly_capture vacío o None")

    load = np.array(hourly_capture.get("load", []), dtype=float)
    from_pv = np.array(hourly_capture.get("from_pv", []), dtype=float)
    from_bess = np.array(hourly_capture.get("from_bess", []), dtype=float)
    from_gen = np.array(hourly_capture.get("from_gen", []), dtype=float)
    pv_gen = np.array(hourly_capture.get("pv_gen", []), dtype=float)

    if not (len(load) == len(from_pv) == len(from_bess) == len(from_gen) == 24, len(pv_gen) == 24):
        raise ValueError("Se esperan 24 valores por serie en hourly_capture")

    hours = np.arange(24, dtype=float)
    step = 1.0 / 12.0
    hours_fine = np.arange(0.0, 23.0 + step, step)

    def interp(y):
        y_f = np.interp(hours_fine, hours, y)
        window = 5
        if len(y_f) > window:
            kernel = np.ones(window) / window
            y_pad = np.pad(y_f, (window//2, window-1-window//2), mode='edge')
            y_smooth = np.convolve(y_pad, kernel, mode='valid')
            return y_smooth
        return y_f

    load_f = interp(load)
    pv_f = interp(from_pv)
    bess_f = interp(from_bess)
    gen_f = interp(from_gen)
    pv_genf = interp(pv_gen)


    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hours_fine, load_f, label="Consumo", color="#726d5f", linewidth=6.0)
    ax.plot(hours_fine, pv_f, label="Consumo desde FV", color="#1b801b", linewidth=2.0)
    ax.plot(hours_fine, bess_f, label="Consumo desde BESS", color="#2046c2", linewidth=2.0)
    ax.plot(hours_fine, gen_f, label="Consumo desde Generador", color="#d62728", linewidth=2.0)
    ax.plot(hours_fine, pv_genf, label="Generación FV", color="#f2a900", linewidth=2.0)

    ax.fill_between(hours_fine, 0, load_f, color="#726d5f", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, pv_f, color="#1b801b", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, bess_f, color="#2046c2", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, gen_f, color="#d62728", alpha=0.18, linewidth=0)
    ax.fill_between(hours_fine, 0, pv_genf, color="#f2a900", alpha=0.18, linewidth=0)

    def hour_formatter(x, pos):
        if x < 0 or x > 23.999:
            return ""
        h = int(np.floor(x + 1e-9))
        m = int(round((x - h) * 60))
        if m == 60:
            h = min(h + 1, 23)
            m = 0
        return f"{h}:{m:02d}"

    ax.xaxis.set_major_locator(MultipleLocator(4.0))
    ax.xaxis.set_major_formatter(FuncFormatter(hour_formatter))

    ax.set_title("Balance Energético Diario XX/XX/XXXX: Consumo y Fuentes de Suministro")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Energía [kWh]")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 120)
    ax.grid(True, linestyle=":", alpha=0.5)
    #ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5)) # Leyendas a la derecha del gráfico y fuera de este
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))  # Leyendas dentro del gráfico y a la izquierda arriba
    plt.subplots_adjust(right=0.8)
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    print("Este módulo solo contiene utilidades de graficación. Importa y usa graficar_desde_series o graficar_dia_enero_ano1 con series ya calculadas.")


