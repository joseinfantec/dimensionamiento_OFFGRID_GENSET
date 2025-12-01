import xlwings as xw
import traceback
import tempfile
import os
import main


def run_optimization():
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
        try:
            wb = xw.Book.caller()
            sistema = wb.sheets["SFV+BESS"]
            sistema.range("I36").value = "❌ ERROR EN LA OPTIMIZACIÓN"
            sistema.range("I43").value = str(e)
            sistema.range("I44").value = traceback.format_exc()
        except Exception:
            pass
        raise

def run_plot():
    try:
        wb = xw.Book.caller()
        sh = wb.sheets["SFV+BESS"]

        dia_a_evaluar = sh.range("M28").value
        try:
            day = int(dia_a_evaluar)
            if day < 1:
                raise ValueError
        except Exception:
            wb.app.alert("Valor de día inválido en M28. Ingrese un entero 1..365.")
            return

        result = main.run_plot_simulation(day)

        if isinstance(result, tuple) and len(result) == 2:
            hourly, y_lim = result
        else:
            hourly = result
            y_lim = None

        if not hourly:
            wb.app.alert("La simulación no devolvió series horarias (hourly_capture vacío).")
            return
        
        if not hourly or not isinstance(hourly, dict):
            wb.app.alert("La simulación no devolvió series horarias válidas.")
            return

        tmpdir = tempfile.gettempdir()
        fname = os.path.join(tmpdir, f"plot_day_{day}.png")
        main.save_hourly_plot_to_png(hourly, fname, y_lim=y_lim)

        try:
            pic = sh.pictures["plot_day"]
            pic.delete()
        except Exception:
            pass

        sh.pictures.add(fname, name="plot_day", update=True, left=sh.range("L30").left, top=sh.range("L30").top)

        # ==== 2) GRÁFICO SOLO GENERADOR =====
        hourly_gen = main.build_generator_only_series(hourly)

        fname_gen = os.path.join(tmpdir, f"plot_generator_{day}.png")
        main.save_generator_only_plot_to_png(hourly_gen, fname_gen, y_lim=y_lim)

        try:
            sh.pictures["plot_gen"].delete()
        except:
            pass
        sh.pictures.add(
            fname_gen,
            name="plot_gen",
            update=True,
            left=sh.range("L55").left,
            top=sh.range("L55").top
        )
        sh.range("L29").value = ""
        sh.range("I43").value = ""
        sh.range("I44").value = ""

        wb.app.alert(f"Gráficos generados para el día {day}.")

    except Exception as e:
        try:
            wb = xw.Book.caller()
            sh = wb.sheets["SFV+BESS"]
            sh.range("L29").value = "❌ ERROR EN LA GRAFICACIÓN"
            sh.range("I43").value = str(e)
            sh.range("I44").value = traceback.format_exc()
        except Exception:
            pass
        raise

