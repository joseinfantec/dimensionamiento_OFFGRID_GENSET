import xlwings as xw
import traceback

def run_optimization():
    """
    Script invocado desde Excel:
    RunPython("import bridge_xlwings; bridge_xlwings.run_optimization()")
    """
    try:
        import main

        res = main.run_optimizer()

        wb = xw.Book.caller()
        sistema = wb.sheets["SFV+BESS"]

        pv = res.get("pv_opt")
        bess = res.get("bess_opt")

        if pv is None or bess is None:
            wb.app.alert(
                "⚠️ No se encontró una solución óptima.\n\n"
                "Revisa los rangos de PV/BESS o la restricción del generador.",
            )
            sistema.range("I36").value = "Optimización sin solución"
            sistema.range("J37").value = "SIN SOLUCIÓN"
            sistema.range("J38").value = ""
            sistema.range("J39").value = ""
            sistema.range("I40").value = ""
            return

        else:
            sistema.range("I36").value = f"RESULTADO: PV={res['pv_opt']:.2f}, BESS={res['bess_opt']:.2f}"

            # ✅ Nuevo: alerta visual al usuario
            wb.app.alert(
                f"✅ Optimización finalizada con éxito.\n\n"
                f"PV óptimo: {res['pv_opt']:.2f}\n"
                f"BESS óptimo: {res['bess_opt']:.2f}\n"
                f"Tiempo transcurrido: {res['tiempo_transcurrido_optimizacion']:.2f} segundos"
            )
            sistema.range("I43").value = ""
            sistema.range("I44").value = ""

    except Exception as e:
        # Escribe error en la hoja
        try:
            wb = xw.Book.caller()
            sistema = wb.sheets["SFV+BESS"]
            sistema.range("I36").value = "❌ ERROR EN LA OPTIMIZACIÓN"
            sistema.range("I43").value = str(e)
            sistema.range("I44").value = traceback.format_exc()
        except Exception:
            pass

        # Relevanta la excepción para depuración en consola
        raise
