import xlwings as xw
import main  


def run_optimization():
    wb = xw.Book.caller()

    # === 1) Leer inputs desde Excel ===
    sh_sfv = wb.sheets["SFV+BESS"]

    # Debes reemplazar estas celdas con tus coordenadas reales:
    pv_min = sh_sfv["C5"].value
    pv_max = sh_sfv["D5"].value
    bess_min = sh_sfv["C6"].value
    bess_max = sh_sfv["D6"].value

    # === 2) Llamar a tu optimizador existente ===
    # Tu main.py debe tener una función que reciba los límites
    # Si aún no la tienes, te digo cómo agregarla sin romper nada
    resultado = main.run_optimizer(
        pv_min=pv_min,
        pv_max=pv_max,
        bess_min=bess_min,
        bess_max=bess_max,
    )

    # === 3) Escribir los resultados en Excel ===
    sh_sfv["F5"].value = resultado["pv_optimo"]
    sh_sfv["F6"].value = resultado["bess_optimo"]
    sh_sfv["F7"].value = resultado["vpn"]


def main():
    run_optimization()


if __name__ == "__main__":
    main()
