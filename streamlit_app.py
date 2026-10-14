# -*- coding: utf-8 -*-
"""
MEMORIA DESCRIPTIVA Y DE CÁLCULO DE LÍNEA DE BOMBEO
Criterios basados en los manuales MAPAS - CONAGUA (Datos Básicos, Conducción,
Líneas de conducción, Pozos, Cárcamos, Tanques de regulación).

Ejecutar:   pip install streamlit pandas numpy matplotlib openpyxl python-docx
            streamlit run memoria_bombeo.py

IMPORTANTE: todas las tablas precargadas (rugosidades, K, coeficientes de regulación,
tiempos de ciclo, etc.) son VALORES DE REFERENCIA EDITABLES. Verifica cada uno
contra la edición vigente del manual MAPAS antes de emitir una memoria oficial.
"""
import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81          # m/s2
NU = 1.01e-6      # viscosidad cinemática agua 20 °C (m2/s)

# ----------------------------------------------------------------------------
# TABLAS PRECARGADAS (editar/verificar con MAPAS)
# eps: rugosidad absoluta (mm) | C: Hazen-Williams | f_di: DI/DN aprox. | a: celeridad (m/s)
# ----------------------------------------------------------------------------
MATERIALES = {
    "PVC hidráulico":            dict(eps=0.0015, C=150, f_di=0.95, a=400),
    "PEAD (polietileno AD)":     dict(eps=0.0015, C=150, f_di=0.90, a=300),
    "Acero soldado (nuevo)":     dict(eps=0.10,   C=120, f_di=0.98, a=1000),
    "Acero galvanizado":         dict(eps=0.15,   C=120, f_di=1.00, a=1000),
    "Fierro dúctil c/ mortero":  dict(eps=0.12,   C=130, f_di=1.00, a=1100),
    "Fierro fundido (nuevo)":    dict(eps=0.25,   C=130, f_di=1.00, a=1100),
    "Fibrocemento (A-C)":        dict(eps=0.025,  C=135, f_di=1.00, a=900),
    "Concreto presforzado":      dict(eps=0.30,   C=120, f_di=1.00, a=1000),
    "PRFV (fibra de vidrio)":    dict(eps=0.01,   C=140, f_di=0.98, a=450),
}
DN_COMERCIALES = [50, 63, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600,
                  750, 900, 1050, 1200, 1500]

K_TABLE = {
    "Entrada a tubería (borde vivo)": 0.50,
    "Codo 90°": 0.90,
    "Codo 90° radio largo": 0.60,
    "Codo 45°": 0.40,
    "Codo 22.5°": 0.20,
    "Tee paso directo": 0.60,
    "Tee salida lateral": 1.80,
    "Válvula de compuerta (abierta)": 0.20,
    "Válvula de mariposa (abierta)": 0.50,
    "Válvula de retención (columpio)": 2.50,
    "Válvula de retención (doble disco)": 1.50,
    "Válvula de pie con coladera": 1.75,
    "Reducción gradual": 0.15,
    "Ampliación gradual": 0.30,
    "Medidor electromagnético": 0.00,
    "Junta flexible / Dresser": 0.00,
    "Válvula de expulsión de aire": 0.00,
    "Válvula de desagüe (purga)": 0.00,
    "Salida a tanque (descarga)": 1.00,
}
ACC_OPTIONS = list(K_TABLE) + ["Otro (manual)"]
HP_COMERCIALES = [1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100,
                  125, 150, 200, 250, 300, 350, 400, 500]
PN_KGCM2 = [5, 7, 10, 14, 20, 25, 32, 40]

# Coeficiente de regulación (m3 por l/s de Qmd). Valor de referencia 24 h = 14.58
# (MAPAS - Datos Básicos). Para otros horarios usa la curva masa (pestaña Tanques).
CR_24H = 14.58
PATRON_DEMANDA = [0.45, 0.45, 0.45, 0.45, 0.45, 0.60, 0.90, 1.35, 1.50, 1.50, 1.40, 1.40,
                  1.40, 1.40, 1.40, 1.35, 1.25, 1.20, 1.20, 1.10, 0.90, 0.70, 0.55, 0.50]
# ^ PATRÓN GENÉRICO de ejemplo; sustituir por el de la ley de demanda del manual / local.

EQUIPOS = {
    "Pozo profundo": ["Sumergible", "Turbina vertical (flecha larga)"],
    "Cárcamo de bombeo": ["Sumergible (cárcamo húmedo)", "Centrífuga horizontal (cárcamo seco)",
                          "Turbina vertical (flecha larga)"],
}
ETA_B = {"Sumergible": 0.68, "Turbina vertical (flecha larga)": 0.72,
         "Sumergible (cárcamo húmedo)": 0.68, "Centrífuga horizontal (cárcamo seco)": 0.70}


# ----------------------------------------------------------------------------
# FUNCIONES HIDRÁULICAS
# ----------------------------------------------------------------------------
def hf_pipe(Q, D, L, eps_mm, C, method):
    """Pérdida por fricción en un tramo. Q m3/s, D m, L m."""
    if L <= 0 or Q <= 0:
        return 0.0
    if method.startswith("Hazen"):
        return 10.67 * L * Q ** 1.852 / (C ** 1.852 * D ** 4.87)
    v = 4 * Q / (math.pi * D ** 2)
    Re = max(v * D / NU, 1.0)
    e = eps_mm / 1000.0
    f = 0.25 / (math.log10(e / (3.7 * D) + 5.74 / Re ** 0.9)) ** 2   # Swamee-Jain
    if Re < 2300:
        f = 64 / Re
    return f * (L / D) * v ** 2 / (2 * G)


def commercial_up(x_mm, lst=DN_COMERCIALES):
    for d in lst:
        if d >= x_mm - 1e-9:
            return d
    return lst[-1]


def parse_ch(v):
    if isinstance(v, str):
        s = v.strip().replace(" ", "")
        if "+" in s:
            a, b = s.split("+")
            return float(a) * 1000 + float(b.replace(",", "."))
        return float(s.replace(",", "."))
    return float(v)


def read_profile(file):
    name = file.name.lower()
    df = pd.read_csv(file) if name.endswith(".csv") else pd.read_excel(file)
    cols = {c: str(c).lower() for c in df.columns}
    c_ch = next((c for c, l in cols.items() if "cad" in l or "km" in l or "dist" in l), df.columns[0])
    c_z = next((c for c, l in cols.items() if "cota" in l or "elev" in l or l == "z"), df.columns[1])
    out = pd.DataFrame({"ch": df[c_ch].map(parse_ch), "z": pd.to_numeric(df[c_z])})
    return out.dropna().sort_values("ch").reset_index(drop=True)


def demo_profile():
    ch = np.arange(0, 2001, 100.0)
    z = 2240 + 0.022 * ch + 9 * np.sin(ch / 260.0) + 6 * np.sin(ch / 90.0)
    return pd.DataFrame({"ch": ch, "z": np.round(z, 2)})


def hidraulica(prof, cover, D, Q, eps, C, method, acc, z_dest):
    ch = prof["ch"].to_numpy(float)
    zt = prof["z"].to_numpy(float)
    zp = prof["zp"].to_numpy(float) if "zp" in prof else zt - cover
    Lseg = np.hypot(np.diff(ch), np.diff(zp))
    hfs = np.array([hf_pipe(Q, D, l, eps, C, method) for l in Lseg])
    a = acc.copy()
    a = a.dropna(subset=["Cadenamiento (m)", "Elemento"]) if len(a) else a
    rows = []
    for _, r in a.iterrows():
        K = r["K"] if pd.notna(r["K"]) else K_TABLE.get(r["Elemento"], 0.0)
        n = r["Cantidad"] if pd.notna(r["Cantidad"]) else 1
        d = (r["Diámetro (mm)"] / 1000.0) if pd.notna(r["Diámetro (mm)"]) and r["Diámetro (mm)"] > 0 else D
        v = Q / (math.pi * d ** 2 / 4)
        rows.append(dict(c=float(r["Cadenamiento (m)"]), Elemento=r["Elemento"], n=n, K=K,
                         d_mm=d * 1000, v=v, hl=n * K * v ** 2 / (2 * G)))
    accr = pd.DataFrame(rows, columns=["c", "Elemento", "n", "K", "d_mm", "v", "hl"])
    down_hf = np.array([hfs[i:].sum() for i in range(len(ch))])
    down_hl = np.array([accr.loc[accr["c"] >= c - 1e-6, "hl"].sum() for c in ch])
    LE = z_dest + down_hf + down_hl
    return dict(ch=ch, zt=zt, zp=zp, Lseg=Lseg, hfs=hfs, accr=accr, LE=LE,
                P=LE - zp, Pst=z_dest - zp, hf=hfs.sum(), hl=accr["hl"].sum(),
                H0=LE[0], L=Lseg.sum(), v=Q / (math.pi * D ** 2 / 4))


def puntos_alto_bajo(zp):
    hi, lo = [], []
    for i in range(1, len(zp) - 1):
        if zp[i] > zp[i - 1] and zp[i] >= zp[i + 1]:
            hi.append(i)
        if zp[i] < zp[i - 1] and zp[i] <= zp[i + 1]:
            lo.append(i)
    return hi, lo


def potencia(Q, H, eta_b, eta_m, fs=1.15):
    ph = G * Q * H                              # kW (Q m3/s * 9.81 * H) = kW
    p_eje = ph / eta_b
    p_motor = p_eje * fs
    hp_req = p_motor / 0.7457
    hp_com = next((h for h in HP_COMERCIALES if h >= hp_req), HP_COMERCIALES[-1])
    return dict(ph=ph, p_eje=p_eje, p_elec=p_eje / eta_m, hp_req=hp_req, hp_com=hp_com)


def cr_curva_masa(patron, T, h0):
    p = np.array(patron, float)
    p = p / p.sum()
    entra = np.zeros(24)
    for k in range(int(round(T))):
        entra[(h0 + k) % 24] = 1.0 / T
    cum = np.cumsum(entra - p)
    frac = cum.max() - cum.min()
    return frac * 86.4, cum        # m3 por l/s de Qmd


# ----------------------------------------------------------------------------
# GRÁFICAS
# ----------------------------------------------------------------------------
def fig_perfil(prof, res, acc_df, hi, lo, pn_m=None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    ch = res["ch"]
    a1.plot(ch, res["zt"], color="saddlebrown", lw=1.5, label="Terreno")
    a1.plot(ch, res["zp"], color="navy", lw=1.8, label="Tubería")
    a1.plot(ch, res["LE"], color="red", ls="--", lw=1.5, label="Línea de energía (bombeo)")
    a1.axhline(res["LE"][-1] - res["accr"]["hl"].tail(1).sum() if len(res["accr"]) else res["LE"][-1],
               color="c", lw=0.6, ls=":")
    a1.fill_between(ch, res["zt"], res["zt"].min() - 5, color="burlywood", alpha=0.3)
    if len(hi):
        a1.scatter(ch[hi], res["zp"][hi], marker="^", color="orange", s=70, zorder=5, label="Puntos altos")
    if len(lo):
        a1.scatter(ch[lo], res["zp"][lo], marker="v", color="green", s=70, zorder=5, label="Puntos bajos")
    markers = {"Válvula de expulsión de aire": ("*", "darkorange", 160),
               "Válvula de desagüe (purga)": ("s", "green", 60),
               "Válvula de retención (columpio)": ("D", "purple", 50),
               "Válvula de compuerta (abierta)": ("X", "black", 55)}
    seen = set()
    for _, r in res["accr"].iterrows():
        zc = np.interp(r["c"], ch, res["zp"])
        m = markers.get(r["Elemento"], ("o", "gray", 30))
        a1.scatter(r["c"], zc, marker=m[0], color=m[1], s=m[2], zorder=6,
                   label=None if r["Elemento"] in seen else r["Elemento"])
        seen.add(r["Elemento"])
    a1.set_ylabel("Elevación (msnm)")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7, ncol=3, loc="upper left")
    a1.set_title("Perfil de la línea de conducción por bombeo")
    a2.plot(ch, res["P"], color="red", label="Presión de operación (m.c.a.)")
    a2.plot(ch, res["Pst"], color="blue", ls="--", label="Presión estática (bomba parada)")
    a2.axhline(0, color="k", lw=0.8)
    if pn_m:
        a2.axhline(pn_m, color="m", ls=":", label="Clase de presión adoptada")
    a2.set_xlabel("Cadenamiento (m)")
    a2.set_ylabel("Presión (m.c.a.)")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def fig_curva_masa(cum):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(24), cum * 100, color="steelblue")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Volumen acumulado (% del volumen diario)")
    ax.grid(alpha=0.3)
    ax.set_title("Curva masa de regulación")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# ESQUEMAS (no a escala; las cotas rotuladas SÍ son las calculadas)
# ----------------------------------------------------------------------------
from matplotlib.patches import Rectangle

C_AGUA, C_AGUA_U, C_TERR, C_SUELO = "#bcd9ec", "#4a90c4", "#6b4f2a", "#eadfca"


def _etiquetas(ax, items, x_line, x_text, yr, fs=6.5):
    """Rotula niveles evitando que los textos se encimen."""
    items = sorted(items, key=lambda t: -t[0])
    gap = 0.06 * yr
    ypos = []
    for y, _, _ in items:
        ypos.append(y if not ypos else min(y, ypos[-1] - gap))
    for (y, t, c), yp in zip(items, ypos):
        ax.plot([x_line, x_text - 0.2], [y, yp], color=c, lw=0.6)
        ax.text(x_text, yp, t, va="center", ha="left", fontsize=fs, color=c)


def draw_pozo(ax, d):
    top, bot = d["z_t"], d["z_bomba"]
    yr = max(top - bot, 1.0)
    ax.set_xlim(0, 20)
    ax.set_ylim(bot - 0.12 * yr, top + 0.12 * yr)
    ax.axis("off")
    ax.fill_between([0, 6.8], [top, top], [bot - 0.12 * yr] * 2, color=C_SUELO)
    ax.plot([0, 6.8], [top, top], color=C_TERR, lw=2)
    ax.add_patch(Rectangle((4, bot), 2, top - bot + 0.04 * yr, fc="white", ec="k", lw=1.2, zorder=2))
    ax.add_patch(Rectangle((4, bot), 2, d["z_ne"] - bot, fc=C_AGUA, ec="none", zorder=3))
    ax.add_patch(Rectangle((4, bot), 2, d["z_nd"] - bot, fc=C_AGUA_U, ec="none", zorder=3))
    ax.add_patch(Rectangle((4.35, d["z_succ"]), 1.3, 0.10 * yr, fc="gray", ec="k", zorder=4))
    ax.plot([5, 5], [d["z_succ"] + 0.10 * yr, top + 0.04 * yr], color="k", lw=2.5, zorder=4)
    _etiquetas(ax, [(top, f"Terreno {top:.2f}", C_TERR),
                    (d["z_ne"], f"NE {d['z_ne']:.2f}", "#1f77b4"),
                    (d["z_nd"], f"ND {d['z_nd']:.2f}", "#08519c"),
                    (d["z_nda"], f"ND + abatim. {d['z_nda']:.2f}", "#08306b"),
                    (d["z_succ"], f"Succión bomba {d['z_succ']:.2f}", "k"),
                    (bot, f"Colocación (c/holgura) {bot:.2f}", "gray")], 6, 7.2, yr)
    ax.set_title("Pozo profundo (corte)", fontsize=9)


def draw_carcamo(ax, d):
    top, bot = d["z_t"], d["z_fondo"]
    yr = max(top - bot, 1.0)
    ax.set_xlim(0, 20)
    ax.set_ylim(bot - 0.12 * yr, top + 0.15 * yr)
    ax.axis("off")
    ax.fill_between([0, 9.9], [bot - 0.12 * yr] * 2, [bot] * 2, color=C_SUELO)
    ax.fill_between([0, 3], [top] * 2, [bot - 0.12 * yr] * 2, color=C_SUELO)
    ax.fill_between([9, 9.9], [top] * 2, [bot - 0.12 * yr] * 2, color=C_SUELO)
    ax.plot([0, 3], [top, top], color=C_TERR, lw=2)
    ax.plot([9, 9.9], [top, top], color=C_TERR, lw=2)
    ax.add_patch(Rectangle((3, bot), 6, top - bot + 0.05 * yr, fc="white", ec="k", lw=1.5, zorder=2))
    ax.add_patch(Rectangle((3, bot), 6, d["namino"] - bot, fc=C_AGUA, ec="none", zorder=3))
    ax.add_patch(Rectangle((3, d["namino"]), 6, d["namo"] - d["namino"], fc=C_AGUA_U, ec="none", zorder=3))
    ax.add_patch(Rectangle((5.2, bot + 0.01 * yr), 1.2, (d["z_succ"] - bot) + 0.10 * yr, fc="gray", ec="k", zorder=4))
    ax.plot([5.8, 5.8], [d["z_succ"] + 0.10 * yr, top + 0.05 * yr], color="k", lw=2.5, zorder=4)
    ax.text(6.0, (d["namino"] + d["namo"]) / 2, "Vol. útil", fontsize=6, ha="left", va="center",
            color="white", zorder=5)
    _etiquetas(ax, [(top, f"Terreno {top:.2f}", C_TERR),
                    (d["name"], f"NAME {d['name']:.2f}", "#b30000"),
                    (d["namo"], f"NAMO (arranque) {d['namo']:.2f}", "#08519c"),
                    (d["namino"], f"NAMINO (paro) {d['namino']:.2f}", "#08306b"),
                    (d["z_succ"], f"Succión bomba {d['z_succ']:.2f}", "k"),
                    (bot, f"Fondo {bot:.2f}", "gray")], 9, 10.2, yr)
    ax.set_title("Cárcamo de bombeo (corte)", fontsize=9)


def draw_tanque(ax, s):
    z_t, fo, no, co = s["z_t"], s["fondo"], s["namo"], s["corona"]
    lo = min(fo, z_t)
    yr = max(co - lo, 1.0)
    ax.set_xlim(0, 20)
    ax.set_ylim(lo - 0.18 * yr, co + 0.22 * yr)
    ax.axis("off")
    ax.fill_between([0, 7.5], [z_t] * 2, [lo - 0.18 * yr] * 2, color=C_SUELO)
    ax.plot([0, 7.5], [z_t, z_t], color=C_TERR, lw=2)
    elevado = fo > z_t + 0.02 * yr
    if elevado:
        for x in (3.4, 6.6):
            ax.plot([x, x], [z_t, fo], color="k", lw=2.5)
        ax.plot([3.4, 6.6], [z_t, fo], color="k", lw=0.8)
        ax.plot([6.6, 3.4], [z_t, fo], color="k", lw=0.8)
        ax.text(0.2, (z_t + fo) / 2, f"Torre\n{fo - z_t:.1f} m", fontsize=6.5, va="center")
    ax.add_patch(Rectangle((3, fo), 4, co - fo, fc="white", ec="k", lw=1.5, zorder=3))
    ax.add_patch(Rectangle((3, fo), 4, no - fo, fc=C_AGUA_U, ec="none", zorder=4))
    if s.get("recibe"):
        ax.annotate("Llegada\nde la línea", xy=(3.0, no + 0.05 * yr), xytext=(0.1, co + 0.10 * yr),
                    fontsize=6.5, arrowprops=dict(arrowstyle="->", color="navy"), color="navy")
    items = [(co, f"Corona {co:.2f}", "#b30000"), (no, f"NAMO {no:.2f}", "#08519c"),
             (fo, f"Fondo = NAMIN {fo:.2f}", "#08306b")]
    if abs(z_t - fo) > 0.05 * yr:
        items.append((z_t, f"Terreno {z_t:.2f}", C_TERR))
    _etiquetas(ax, items, 7, 8.2, yr)
    ax.set_title(s["nombre"], fontsize=9)


def fig_pozo(d):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    draw_pozo(ax, d)
    fig.tight_layout()
    return fig


def fig_carcamo(d):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    draw_carcamo(ax, d)
    fig.tight_layout()
    return fig


def fig_tanques(estr):
    n = len(estr)
    fig, axs = plt.subplots(1, n, figsize=(4.4 * n, 4.0))
    axs = np.atleast_1d(axs)
    for ax, s_ in zip(axs, estr):
        draw_tanque(ax, s_)
    fig.tight_layout()
    return fig


def fig_descarga(destino, llega_en, h):
    """Croquis pequeño: qué altura capturar en el punto de descarga."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.4, 7.6)
    ax.axis("off")
    ax.fill_between([0, 10], [0, 0], [-2.4, -2.4], color=C_SUELO)
    ax.plot([0, 10], [0, 0], color=C_TERR, lw=2)
    ax.text(6.9, -0.75, "Terreno final", fontsize=6.5, color=C_TERR)
    elevado = destino == "Tanque elevado" or (destino == "Cisterna + tanque elevado" and llega_en == "Tanque elevado")
    cist = destino == "Cisterna" or (destino == "Cisterna + tanque elevado" and llega_en == "Cisterna")
    if elevado:
        for x in (3.8, 6.2):
            ax.plot([x, x], [0, 3.4], color="k", lw=2.5)
        ax.add_patch(Rectangle((3.2, 3.4), 3.6, 2.4, fc="white", ec="k", lw=1.5))
        ax.add_patch(Rectangle((3.2, 3.4), 3.6, 2.0, fc=C_AGUA_U, ec="none"))
        y_n, lab = 5.4, "h = torre + tirante"
        ax.annotate("", xy=(2.6, 3.4), xytext=(2.6, 0), arrowprops=dict(arrowstyle="<->", color="gray"))
        ax.text(0.2, 1.6, "torre", fontsize=6.5, color="gray")
        y_in = 5.9
    elif cist:
        ax.add_patch(Rectangle((3.2, -1.9), 3.6, 3.0, fc="white", ec="k", lw=1.5, zorder=3))
        ax.add_patch(Rectangle((3.2, -1.9), 3.6, 2.5, fc=C_AGUA_U, ec="none", zorder=4))
        y_n, lab = 0.6, "h (≈ 0 si el agua\nllega al nivel del terreno)"
        y_in = 1.3
    else:
        ax.add_patch(Rectangle((3.2, 0), 3.6, 3.4, fc="white", ec="k", lw=1.5, zorder=3))
        ax.add_patch(Rectangle((3.2, 0), 3.6, 2.9, fc=C_AGUA_U, ec="none", zorder=4))
        y_n, lab = 2.9, "h ≈ tirante máx."
        y_in = 3.7
    ax.plot([0.2, 2.2, 2.2, 3.6], [-0.6, -0.6, y_in, y_in], color="navy", lw=2, zorder=5)
    ax.annotate("", xy=(3.6, y_in), xytext=(3.3, y_in), arrowprops=dict(arrowstyle="->", color="navy"), zorder=5)
    ax.text(0.2, -1.3, "Línea de conducción", fontsize=6.5, color="navy")
    ax.plot([6.8, 8.4], [y_n, y_n], color="red", ls="--", lw=1)
    ax.annotate("", xy=(8.2, y_n), xytext=(8.2, 0), arrowprops=dict(arrowstyle="<->", color="red", lw=1.4))
    ax.text(8.35, y_n / 2 if y_n > 1 else y_n + 0.5, lab, fontsize=7, color="red", va="center")
    ax.text(6.9, y_n + 0.15, "NAMO", fontsize=6.5, color="red")
    ax.set_title(f"Altura de descarga h  (valor actual: {h:.1f} m)", fontsize=8)
    fig.tight_layout()
    return fig


def fig_sistema(fd, res, estr):
    n = 2 + len(estr)
    fig, axs = plt.subplots(1, n, figsize=(4.0 * n + 2.0, 4.8),
                            gridspec_kw={"width_ratios": [1.0, 1.9] + [1.0] * len(estr)})
    (draw_pozo if fd["tipo"] == "pozo" else draw_carcamo)(axs[0], fd)
    ax = axs[1]
    ch = res["ch"]
    ax.plot(ch, res["zt"], color=C_TERR, ls="--", lw=1, label="Terreno")
    ax.plot(ch, res["zp"], color="navy", lw=1.8, label="Tubería")
    ax.plot(ch, res["LE"], color="red", ls="--", lw=1.3, label="Línea de energía")
    i_hi = int(np.argmax(res["zp"]))
    ax.scatter([ch[0], ch[i_hi], ch[-1]], [res["zp"][0], res["zp"][i_hi], res["zp"][-1]], color="navy", zorder=5)
    ax.annotate(f"Inicio tubería {res['zp'][0]:.2f}\nLE en bomba {res['LE'][0]:.2f}", (ch[0], res["LE"][0]),
                textcoords="offset points", xytext=(6, -38), fontsize=7, color="red")
    ax.annotate(f"Punto más alto {res['zp'][i_hi]:.2f}\n(cad. {ch[i_hi]:.0f} m)", (ch[i_hi], res["zp"][i_hi]),
                textcoords="offset points", xytext=(-12, 14), fontsize=7, color="navy", ha="right")
    ax.annotate(f"Fin tubería {res['zp'][-1]:.2f}\nLE llegada {res['LE'][-1]:.2f}", (ch[-1], res["zp"][-1]),
                textcoords="offset points", xytext=(-100, -46), fontsize=7, color="navy")
    ax.set_xlabel("Cadenamiento (m)")
    ax.set_ylabel("Cota (msnm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Línea de conducción", fontsize=9)
    for a_, s_ in zip(axs[2:], estr):
        draw_tanque(a_, s_)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# EXPORTACIÓN
# ----------------------------------------------------------------------------
def build_excel(sheets):
    from openpyxl.utils import get_column_letter
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for i, col in enumerate(df.columns, 1):
                w = max([len(str(col))] + [len(str(x)) for x in df[col].head(60)]) + 2
                ws.column_dimensions[get_column_letter(i)].width = min(w, 55)
    return buf.getvalue()


def build_docx(resumen, alertas, figs, titulo):
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    doc.add_heading(titulo, 0)
    doc.add_paragraph(f"Fecha de elaboración: {dt.date.today():%d/%m/%Y}. "
                      "Criterios de referencia: manuales MAPAS - CONAGUA. "
                      "Documento generado automáticamente; debe ser revisado por el proyectista responsable.")
    secciones = list(dict.fromkeys(resumen["Sección"]))
    for s in secciones:
        doc.add_heading(s, 1)
        sub = resumen[resumen["Sección"] == s]
        t = doc.add_table(rows=1, cols=3)
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(["Concepto", "Valor", "Unidad"]):
            t.rows[0].cells[i].text = h
        for _, r in sub.iterrows():
            c = t.add_row().cells
            c[0].text, c[1].text, c[2].text = str(r["Concepto"]), str(r["Valor"]), str(r["Unidad"])
    doc.add_heading("Alertas y observaciones", 1)
    for a in alertas or ["Sin alertas."]:
        doc.add_paragraph(a, style="List Bullet")
    doc.add_heading("Figuras", 1)
    for cap, fig in figs:
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=150)
        b.seek(0)
        doc.add_picture(b, width=Inches(6.3))
        doc.add_paragraph(cap)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


# ============================================================================
# INTERFAZ
# ============================================================================
st.set_page_config(page_title="Memoria de línea de bombeo - MAPAS", layout="wide")
st.title("Memoria técnica descriptiva · Línea de bombeo (criterios MAPAS-CONAGUA)")
st.caption("Valores precargados = referencia editable. Verifica contra el manual vigente.")

RES = []          # (sección, concepto, valor, unidad)
ALERTAS = []
SHEETS = {}
FIGS = []


def add(sec, name, val, unit=""):
    RES.append((sec, name, round(val, 3) if isinstance(val, (float, np.floating)) else val, unit))


tabs = st.tabs(["1 Datos", "2 Perfil", "3 Material y diámetro", "4 Accesorios",
                "5 Hidráulica y bomba", "6 Fuente (pozo/cárcamo)", "7 Tanques", "8 Exportar"])

# ------------------------------------------------------------------ 1 DATOS
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        proyecto = st.text_input("Nombre del proyecto", "Línea de bombeo - Ejemplo")
        fuente = st.selectbox("Tipo de fuente de abastecimiento", list(EQUIPOS))
        equipo = st.selectbox("Tipo de equipo de bombeo", EQUIPOS[fuente])
    with c2:
        T = st.number_input("Horas de bombeo al día", 1.0, 24.0, 20.0, 0.5)
        h0 = st.number_input("Hora de arranque (0-23)", 0, 23, 4)
        eta_b = st.number_input("Eficiencia de la bomba", 0.30, 0.90, ETA_B[equipo], 0.01)
        eta_m = st.number_input("Eficiencia del motor", 0.60, 0.97, 0.90, 0.01)
        fs = st.number_input("Factor de servicio del motor", 1.0, 1.5, 1.15, 0.05)
    with c3:
        destino = st.selectbox("Destino", ["Tanque de regulación superficial (piso)", "Tanque elevado",
                                           "Cisterna", "Cisterna + tanque elevado"])
        llega_en = {"Tanque elevado": "Tanque elevado", "Cisterna": "Cisterna"}.get(destino, "Tanque de piso")
        if destino == "Cisterna + tanque elevado":
            llega_en = st.selectbox("La línea de conducción descarga en:", ["Tanque elevado", "Cisterna"])
        h_llegada = st.number_input(
            "Altura de descarga h: nivel MÁXIMO del agua sobre el terreno final (m) — ver croquis", 0.0, 80.0, 4.0, 0.5)
        cover = st.number_input("Profundidad de colocación de tubería (m)", 0.0, 5.0, 1.0, 0.1)
        v_min = st.number_input("Velocidad mínima recomendada (m/s)", 0.1, 2.0, 0.6, 0.1)
        v_max = st.number_input("Velocidad máxima recomendada (m/s)", 0.5, 5.0, 2.0, 0.1)

    st.markdown("---")
    cA, cB = st.columns([1.7, 1])
    with cA:
        st.markdown("#### Caudales de diseño")
        MODOS = ["Población y dotación", "Conozco Qmed", "Conozco el caudal de bombeo Qb"]
        modo_q = st.radio("¿Cómo definir el caudal?", MODOS, horizontal=True)
        modo = MODOS.index(modo_q)
        k1, k2 = st.columns(2)
        dot = pob = None
        if modo == 0:
            dot = k1.number_input("Dotación (l/hab/día)", 20, 1000, 150, 5)
            pob = k2.number_input("Población de proyecto (hab)", 1, 5000000, 6000, 100)
            Qmed_in = pob * dot / 86400.0
        elif modo == 1:
            Qmed_in = k1.number_input("Qmed (l/s)", 0.1, 5000.0, 10.4, 0.1)
        else:
            Qb_in = k1.number_input("Qb, caudal a bombear (l/s)", 0.1, 5000.0, 20.0, 0.5)
        k3, k4 = st.columns(2)
        CVd = k3.number_input("Coeficiente de variación diaria CVd", 1.0, 2.5, 1.20, 0.01)
        CVh = k4.number_input("Coeficiente de variación horaria CVh", 1.0, 4.0, 1.50, 0.01)
        if modo in (0, 1):
            Qmed = Qmed_in
            Qmd = Qmed * CVd
            Qlps = Qmd * 24.0 / T
        else:
            Qlps = Qb_in
            Qmd = Qlps * T / 24.0
            Qmed = Qmd / CVd
        Qmh = Qmed * CVh
        Qlps = max(Qlps, 0.1)
        tq_q = pd.DataFrame([
            ["Qmed", round(Qmed, 3), "Gasto medio diario (Pob × Dot / 86 400)" if modo == 0 else "Gasto medio diario"],
            ["Qmd = Qmed × CVd", round(Qmd, 3),
             "Obra de captación, línea de conducción (hasta el tanque) y tanque de regulación"],
            ["Qmh = Qmed × CVh", round(Qmh, 3), "Línea de alimentación a la red y red de distribución"],
            ["Qb = Qmd × 24 / T", round(Qlps, 3), "Caudal de bombeo: equipo, columna/succión y línea de conducción"],
        ], columns=["Caudal", "l/s", "Se usa para diseñar"])
        st.dataframe(tq_q, width="stretch", hide_index=True)
        if Qlps > 0 and T < 24:
            st.caption(f"Bombeando {T:g} h/día el equipo debe entregar Qb = Qmd·24/T = {Qlps:.2f} l/s para cubrir Qmd.")
    with cB:
        st.markdown("#### ¿Qué altura capturar en la descarga?")
        figd = fig_descarga(destino, llega_en, h_llegada)
        st.pyplot(figd)
        plt.close(figd)
        st.caption("h = distancia vertical entre el terreno en el último punto del perfil y el nivel máximo del "
                   "agua (NAMO) al que descarga la línea. Piso: ≈ tirante; elevado: torre + tirante; "
                   "cisterna: ≈ 0 si el agua llega al nivel del terreno.")
    if modo == 0:
        add("Datos", "Dotación", dot, "l/hab/día")
        add("Datos", "Población de proyecto", pob, "hab")
    add("Datos", "Qmed", float(Qmed), "l/s")
    add("Datos", "CVd / CVh", f"{CVd} / {CVh}")
    add("Datos", "Qmd (captación, conducción, regulación)", float(Qmd), "l/s")
    add("Datos", "Qmh (alimentación a red y red)", float(Qmh), "l/s")
Q = Qlps / 1000.0

# ------------------------------------------------------------------ 2 PERFIL
with tabs[1]:
    st.write("Sube CSV/Excel con columnas **Cadenamiento** (m, o formato 0+500) y **Cota** (msnm), "
             "desde la fuente hasta el tanque.")
    plantilla = demo_profile().rename(columns={"ch": "Cadenamiento", "z": "Cota"})
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_perfil.csv")
    up = st.file_uploader("Perfil del TERRENO", type=["csv", "xlsx", "xls"], key="up_terreno")
    if up is not None:
        terr = read_profile(up)
    else:
        st.info("Usando perfil de ejemplo hasta que subas tu archivo.")
        terr = demo_profile()
    with st.expander("Opcional: perfil de la TUBERÍA (si ya lo tienes en el mismo formato)"):
        st.caption("Mismo formato que el terreno: Cadenamiento y Cota de tubería (rasante/eje). "
                   "Los cadenamientos pueden ser distintos a los del terreno; se interpolan. "
                   "Si no lo cargas, la tubería = terreno − profundidad de colocación.")
        up_p = st.file_uploader("Perfil de la TUBERÍA", type=["csv", "xlsx", "xls"], key="up_tuberia")
    if up_p is not None:
        pp = read_profile(up_p)
        lo_, hi_ = float(terr["ch"].min()), float(terr["ch"].max())
        ch_all = np.union1d(terr["ch"].to_numpy(float), pp["ch"].to_numpy(float))
        ch_all = ch_all[(ch_all >= lo_) & (ch_all <= hi_)]
        prof = pd.DataFrame({"ch": ch_all,
                             "z": np.round(np.interp(ch_all, terr["ch"], terr["z"]), 3),
                             "zp": np.round(np.interp(ch_all, pp["ch"], pp["z"]), 3)})
        if pp["ch"].min() > lo_ + 0.5 or pp["ch"].max() < hi_ - 0.5:
            st.warning("El perfil de tubería no cubre todo el cadenamiento del terreno; "
                       "los extremos se completan con el valor más cercano.")
        cob = prof["z"] - prof["zp"]
        st.write(f"Tubería cargada por el usuario. Profundidad (terreno − tubería): "
                 f"mín **{cob.min():.2f} m**, máx **{cob.max():.2f} m**.")
        if cob.min() < 0:
            st.error("En algún punto la tubería queda por encima del terreno; revisa el archivo.")
            ALERTAS.append("El perfil de tubería cargado queda por encima del terreno en algún tramo.")
        add("Datos", "Perfil de tubería", "Cargado por el usuario")
    else:
        prof = terr.copy()
        prof["zp"] = prof["z"] - cover
        add("Datos", "Perfil de tubería", f"Terreno − {cover} m")
    st.dataframe(prof.rename(columns={"ch": "Cadenamiento (m)", "z": "Cota terreno (msnm)",
                                      "zp": "Cota tubería (msnm)"}), height=200)
    fp, axp = plt.subplots(figsize=(11, 3))
    axp.plot(prof["ch"], prof["z"], color="saddlebrown", label="Terreno")
    axp.plot(prof["ch"], prof["zp"], color="navy", ls="--", label="Tubería")
    axp.fill_between(prof["ch"], prof["z"], prof["z"].min() - 5, color="burlywood", alpha=0.3)
    axp.set_xlabel("Cadenamiento (m)")
    axp.set_ylabel("Cota (msnm)")
    axp.grid(alpha=0.3)
    axp.legend(fontsize=8)
    st.pyplot(fp)
    plt.close(fp)

z_ground0 = float(prof["z"].iloc[0])
z_end = float(prof["z"].iloc[-1])
z_dest = z_end + h_llegada

# ------------------------------------------------------------------ 3 MATERIAL / DIÁMETRO
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        mat_name = st.selectbox("Material", list(MATERIALES) + ["Otro (manual)"])
        base = MATERIALES.get(mat_name, dict(eps=0.1, C=120, f_di=1.0, a=900))
        if mat_name == "Otro (manual)":
            mat_name = st.text_input("Nombre del material", "Material especial")
            eps = st.number_input("Rugosidad absoluta ε (mm)", 0.0001, 5.0, 0.1, format="%.4f")
            Chw = st.number_input("C de Hazen-Williams", 60, 160, 120)
            f_di = st.number_input("Relación DI/DN", 0.5, 1.2, 1.0, 0.01)
            a_cel = st.number_input("Celeridad onda (m/s)", 100, 1500, 900)
        else:
            eps = st.number_input("Rugosidad absoluta ε (mm) [editable]", 0.0001, 5.0, float(base["eps"]),
                                  format="%.4f")
            Chw = st.number_input("C de Hazen-Williams [editable]", 60, 160, int(base["C"]))
            f_di = st.number_input("Relación DI/DN aprox. [usa DI de catálogo si lo tienes]", 0.5, 1.2,
                                   float(base["f_di"]), 0.01)
            a_cel = base["a"]
        metodo = st.radio("Ecuación de fricción", ["Darcy-Weisbach (Swamee-Jain)", "Hazen-Williams"])
    # tabla comparativa: resultados con accesorios actuales se calculan después;
    # aquí se usa la tabla de accesorios guardada en sesión.
    if "acc_v" not in st.session_state:
        st.session_state.acc_v = 0
    acc_state = st.session_state.get("acc", pd.DataFrame(
        columns=["Cadenamiento (m)", "Elemento", "Cantidad", "K", "Diámetro (mm)"]))
    z_low_prelim = z_ground0 - 10   # solo para comparativo; el real se calcula en pestaña 6
    cand = []
    Dteo = math.sqrt(4 * Q / (math.pi * 1.2)) * 1000
    for dn in DN_COMERCIALES:
        di = dn * f_di / 1000
        if di > 1.0 and dn > 600 and Q < 0.05:
            continue
        r = hidraulica(prof, cover, di, Q, eps, Chw, metodo, acc_state, z_dest)
        ok = v_min <= r["v"] <= v_max
        cand.append(dict(DN_mm=dn, DI_mm=round(di * 1000, 1), V_ms=round(r["v"], 2),
                         hf_m=round(r["hf"], 2), hl_m=round(r["hl"], 2),
                         J_m_km=round(r["hf"] / max(r["L"], 1) * 1000, 2),
                         Cumple_v="SÍ" if ok else "NO"))
    cand = pd.DataFrame(cand)
    with c2:
        st.write(f"Diámetro teórico (v≈1.2 m/s): **{Dteo:.0f} mm**")
        st.dataframe(cand, height=300)
        ok_list = cand[cand["Cumple_v"] == "SÍ"]["DN_mm"].tolist()
        dn_default = ok_list[len(ok_list) // 2] if ok_list else 200
        DN = st.selectbox("Diámetro nominal propuesto (mm)", DN_COMERCIALES,
                          index=DN_COMERCIALES.index(dn_default))
        di_ov = st.number_input("Diámetro interior real (mm) - 0 = DN × relación", 0.0, 2000.0, 0.0)
D = (di_ov if di_ov > 0 else DN * f_di) / 1000.0
v_line = Q / (math.pi * D ** 2 / 4)
with tabs[2]:
    if v_line < v_min:
        st.error(f"Velocidad {v_line:.2f} m/s < mínima {v_min}. Reduce el diámetro.")
        ALERTAS.append(f"Velocidad en línea {v_line:.2f} m/s menor a la mínima ({v_min} m/s).")
    elif v_line > v_max:
        st.error(f"Velocidad {v_line:.2f} m/s > máxima {v_max}. Aumenta el diámetro.")
        ALERTAS.append(f"Velocidad en línea {v_line:.2f} m/s mayor a la máxima ({v_max} m/s).")
    else:
        st.success(f"Velocidad {v_line:.2f} m/s: cumple ({v_min}-{v_max} m/s).")
    SHEETS["Diametros_candidatos"] = cand

# ------------------------------------------------------------------ 6 FUENTE (se calcula antes de 5 por dependencias)
# Diámetros de succión / descarga de bomba
D_succ = commercial_up(math.sqrt(4 * Q / (math.pi * 1.5)) * 1000)      # v ≤ 1.5 m/s
D_desc = commercial_up(math.sqrt(4 * Q / (math.pi * 3.0)) * 1000)      # v ≤ 3.0 m/s

with tabs[5]:
    st.subheader(fuente)
    if fuente == "Pozo profundo":
        c1, c2 = st.columns(2)
        with c1:
            ne = st.number_input("Profundidad del nivel estático NE (m bajo terreno)", 0.0, 800.0, 40.0, 1.0)
            nd = st.number_input("Profundidad del nivel dinámico ND a Qb (m bajo terreno)", 0.0, 800.0, 55.0, 1.0)
            abat = st.number_input("Abatimiento adicional esperado (m)", 0.0, 100.0, 5.0, 1.0)
        with c2:
            s_min = st.number_input("Sumergencia mínima sobre succión (m)", 0.5, 20.0, 3.0, 0.5)
            holg = st.number_input("Holgura sobre fondo de pozo / tazones (m)", 0.0, 20.0, 3.0, 0.5)
            d_col = st.number_input("Diámetro interior columna (mm)", 50.0, 600.0,
                                    float(commercial_up(math.sqrt(4 * Q / (math.pi * 2.0)) * 1000)))
        prof_bomba = nd + abat + s_min + holg
        h_col = hf_pipe(Q, d_col / 1000, prof_bomba, 0.15, 120, metodo) + 0.5   # +0.5 m cabezal/codo
        z_low = z_ground0 - nd
        z_high = z_ground0 - ne
        v_col = Q / (math.pi * (d_col / 1000) ** 2 / 4)
        st.write(f"Profundidad de colocación de bomba: **{prof_bomba:.1f} m**; pérdida en columna: "
                 f"**{h_col:.2f} m**; v columna = {v_col:.2f} m/s")
        if v_col > 2.5:
            ALERTAS.append(f"Velocidad en columna {v_col:.2f} m/s alta; aumentar diámetro de columna.")
        add("Pozo profundo", "Nivel estático (profundidad)", ne, "m")
        add("Pozo profundo", "Nivel dinámico (profundidad)", nd, "m")
        add("Pozo profundo", "Profundidad de colocación de bomba", prof_bomba, "m")
        add("Pozo profundo", "Pérdida en columna", h_col, "m")
        add("Pozo profundo", "Diámetro de columna", d_col, "mm")
        z_t = z_ground0
        FUENTE_D = dict(tipo="pozo", z_t=z_t, z_ne=z_t - ne, z_nd=z_t - nd, z_nda=z_t - nd - abat,
                        z_succ=z_t - (nd + abat + s_min), z_bomba=z_t - prof_bomba)
        NIV_FUENTE = [
            ("Terreno en el pozo", z_t, "Cota de terreno (inicio del perfil)"),
            ("Nivel estático (NE)", FUENTE_D["z_ne"], f"{ne:.1f} m bajo terreno"),
            ("Nivel dinámico (ND) a Qb", FUENTE_D["z_nd"], f"{nd:.1f} m bajo terreno"),
            ("ND + abatimiento adicional", FUENTE_D["z_nda"], f"{nd + abat:.1f} m bajo terreno"),
            ("Succión de la bomba", FUENTE_D["z_succ"], f"ND + abatim. + sumergencia = {nd + abat + s_min:.1f} m"),
            ("Profundidad de colocación (incl. holgura)", FUENTE_D["z_bomba"], f"{prof_bomba:.1f} m bajo terreno"),
        ]
        cf1, cf2 = st.columns(2)
        figf = fig_pozo(FUENTE_D)
        cf1.pyplot(figf)
        plt.close(figf)
        cf2.dataframe(pd.DataFrame(NIV_FUENTE, columns=["Elemento", "Cota (msnm)", "Referencia"]).round(2),
                      hide_index=True, width="stretch")
    else:
        c1, c2 = st.columns(2)
        with c1:
            prof_carc = st.number_input("Profundidad del fondo del cárcamo (m bajo terreno)", 1.0, 30.0, 4.0, 0.5)
            ancho = st.number_input("Ancho interior propuesto (m)", 1.0, 30.0, 3.0, 0.5)
            tciclo_def = 5 if Qlps < 30 else (8 if Qlps < 150 else 12)
            tc = st.number_input("Tiempo mínimo de ciclo de arranques (min) [verificar MAPAS]", 2.0, 60.0,
                                 float(tciclo_def), 0.5)
        with c2:
            h_succ = st.number_input("Pérdidas en succión y piezas especiales de la estación (m)", 0.0, 10.0, 0.8, 0.1)
            margen = st.number_input("Margen NAME sobre NAMO (m)", 0.0, 2.0, 0.3, 0.1)
        Ds = D_succ / 1000
        Fr = (Q / (math.pi * Ds ** 2 / 4)) / math.sqrt(G * Ds)
        S = Ds * (1 + 2.3 * Fr)                      # sumergencia mínima (Hydraulic Institute)
        C_fondo = max(0.15, 0.3 * Ds)
        V_util = Q * tc * 60 / 4                     # m3  V = Q·t/4
        z_fondo = z_ground0 - prof_carc
        namino = z_fondo + C_fondo + S
        # área: largo
        # NAMO - NAMINO = V/A
        largo = max(1.5, 1.0)
        A_req_h = 1.0
        # se itera el largo para que la altura útil sea razonable (≈0.8-1.5 m)
        h_util = st.number_input("Altura útil entre NAMINO y NAMO (m)", 0.3, 4.0, 1.0, 0.1)
        A = V_util / h_util
        largo = A / ancho
        namo = namino + h_util
        name_ = namo + margen
        z_low = namino
        z_high = namo
        h_col = h_succ
        st.write(f"Volumen útil V = Q·tc/4 = **{V_util:.1f} m³**; planta **{ancho:.2f} × {largo:.2f} m**")
        st.write(f"Cotas: fondo {z_fondo:.2f} | NAMINO {namino:.2f} | NAMO {namo:.2f} | NAME {name_:.2f}  "
                 f"(sumergencia mínima S = {S:.2f} m)")
        if name_ > z_ground0 - 0.3:
            st.error("El NAME queda muy cerca/sobre el terreno: profundiza el cárcamo o ajusta dimensiones.")
            ALERTAS.append("Cárcamo: NAME demasiado cercano al nivel de terreno.")
        for k, v, u in [("Volumen útil", V_util, "m³"), ("Largo", largo, "m"), ("Ancho", ancho, "m"),
                        ("Sumergencia mínima", S, "m"), ("Cota fondo", z_fondo, "msnm"),
                        ("Cota NAMINO (paro)", namino, "msnm"), ("Cota NAMO (arranque)", namo, "msnm"),
                        ("Cota NAME", name_, "msnm")]:
            add("Cárcamo", k, v, u)
        FUENTE_D = dict(tipo="carcamo", z_t=z_ground0, z_fondo=z_fondo, z_succ=z_fondo + C_fondo,
                        namino=namino, namo=namo, name=name_)
        NIV_FUENTE = [
            ("Terreno en el cárcamo", z_ground0, "Cota de terreno (inicio del perfil)"),
            ("NAME (nivel de aguas máximas extraordinarias)", name_, f"NAMO + {margen:.2f} m"),
            ("NAMO (arranque de bomba)", namo, f"NAMINO + {h_util:.2f} m (volumen útil)"),
            ("NAMINO (paro de bomba)", namino, f"Fondo + holgura + sumergencia = {C_fondo + S:.2f} m"),
            ("Succión de la bomba", z_fondo + C_fondo, f"Fondo + {C_fondo:.2f} m"),
            ("Fondo del cárcamo", z_fondo, f"{prof_carc:.1f} m bajo terreno"),
        ]
        cf1, cf2 = st.columns(2)
        figf = fig_carcamo(FUENTE_D)
        cf1.pyplot(figf)
        plt.close(figf)
        cf2.dataframe(pd.DataFrame(NIV_FUENTE, columns=["Elemento", "Cota (msnm)", "Referencia"]).round(2),
                      hide_index=True, width="stretch")
    add("Bombeo", "Diámetro succión (v≤1.5 m/s)", D_succ, "mm")
    add("Bombeo", "Diámetro descarga bomba (v≤3 m/s)", D_desc, "mm")

# ------------------------------------------------------------------ 4 ACCESORIOS
# puntos altos/bajos
zp_tmp = prof["zp"].to_numpy(float)
hi, lo = puntos_alto_bajo(zp_tmp)
ch_arr = prof["ch"].to_numpy(float)


def sugerir():
    rows = []
    c0, cf = ch_arr[0], ch_arr[-1]
    for el, n in [("Ampliación gradual", 1), ("Válvula de retención (columpio)", 1),
                  ("Válvula de compuerta (abierta)", 1), ("Medidor electromagnético", 1),
                  ("Codo 90°", 1)]:
        rows.append([c0, el, n, K_TABLE[el], np.nan])
    for i in hi:
        rows.append([ch_arr[i], "Válvula de expulsión de aire", 1, 0.0, np.nan])
    for i in lo:
        rows.append([ch_arr[i], "Válvula de desagüe (purga)", 1, 0.0, np.nan])
    rows.append([cf, "Válvula de compuerta (abierta)", 1, K_TABLE["Válvula de compuerta (abierta)"], np.nan])
    rows.append([cf, "Salida a tanque (descarga)", 1, 1.0, np.nan])
    return pd.DataFrame(rows, columns=["Cadenamiento (m)", "Elemento", "Cantidad", "K", "Diámetro (mm)"])


if "acc" not in st.session_state or st.session_state.acc.empty:
    st.session_state.acc = sugerir()

with tabs[3]:
    st.write("Puntos altos detectados (posibles bolsas de aire): " +
             (", ".join(f"{ch_arr[i]:.0f} m" for i in hi) or "ninguno") +
             " | Puntos bajos (desagüe): " + (", ".join(f"{ch_arr[i]:.0f} m" for i in lo) or "ninguno"))
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Regenerar sugerencia automática"):
            st.session_state.acc = sugerir()
            st.session_state.acc_v += 1
            st.rerun()
        long_max = st.number_input("Si hay tramos largos sin punto alto, valvula cada (m)", 100, 3000, 1000, 100)
        st.caption("K en blanco = se toma de la tabla precargada. Para 'Otro (manual)' escribe K.\n"
                   "Diámetro en blanco = diámetro de la línea.")
    with c2:
        acc = st.data_editor(
            st.session_state.acc, num_rows="dynamic", key=f"acc_ed_{st.session_state.acc_v}",
            column_config={
                "Elemento": st.column_config.SelectboxColumn(options=ACC_OPTIONS, required=True),
                "Cadenamiento (m)": st.column_config.NumberColumn(format="%.1f"),
                "Cantidad": st.column_config.NumberColumn(min_value=1, step=1, default=1),
                "K": st.column_config.NumberColumn(format="%.2f"),
                "Diámetro (mm)": st.column_config.NumberColumn(format="%.0f")}, width="stretch")
    st.session_state.acc = acc
    # sin punto alto en tramos largos
    air = sorted(acc.loc[acc["Elemento"] == "Válvula de expulsión de aire", "Cadenamiento (m)"].dropna())
    marks = [ch_arr[0]] + air + [ch_arr[-1]]
    for a_, b_ in zip(marks[:-1], marks[1:]):
        if b_ - a_ > long_max:
            st.warning(f"Tramo {a_:.0f}–{b_:.0f} m sin válvula de aire ({b_ - a_:.0f} m): considera una intermedia.")
    missing = [ch_arr[i] for i in hi if not any(abs(ch_arr[i] - x) < 1 for x in air)]
    for m_ in missing:
        st.warning(f"Punto alto en {m_:.0f} m sin válvula de expulsión de aire.")
        ALERTAS.append(f"Punto alto en cad. {m_:.0f} m sin válvula de expulsión de aire.")
    # diámetros de accesorios
    d_air = commercial_up(max(25, DN / 12), [25, 38, 50, 75, 100, 150, 200])
    d_drain = commercial_up(max(50, DN / 4), [50, 75, 100, 150, 200, 250])
    dacc = pd.DataFrame([
        ["Válvula de compuerta / mariposa / retención en línea", DN, "Igual al diámetro de la línea"],
        ["Válvula de expulsión de aire", d_air, "≈ D/12 (orificio/conexión), mín. 25 mm [verificar]"],
        ["Válvula de desagüe (purga)", d_drain, "≈ D/4, mín. 50 mm [verificar]"],
        ["Succión de bomba", D_succ, "v ≤ 1.5 m/s"],
        ["Descarga de bomba", D_desc, "v ≤ 3.0 m/s (ampliación gradual a DN de línea)"],
    ], columns=["Accesorio", "DN recomendado (mm)", "Criterio"])
    st.write("**Diámetros de accesorios**")
    st.dataframe(dacc, width="stretch")
    SHEETS["Diametros_accesorios"] = dacc
    add("Accesorios", "DN válvula de aire", d_air, "mm")
    add("Accesorios", "DN válvula de desagüe", d_drain, "mm")

# ------------------------------------------------------------------ 5 HIDRÁULICA
res = hidraulica(prof, cover, D, Q, eps, Chw, metodo, acc, z_dest)
TDH_max = res["H0"] - z_low + h_col
TDH_min = res["H0"] - z_high + h_col
pot = potencia(Q, TDH_max, eta_b, eta_m, fs)
dH_surge = a_cel * res["v"] / G
P_op = res["P"].max()
P_st = res["Pst"].max()
P_surge = max(P_op, P_st) + dH_surge
pn_sel = next((p for p in PN_KGCM2 if p * 10 >= P_surge), PN_KGCM2[-1])
hi_r, lo_r = puntos_alto_bajo(res["zp"])

with tabs[4]:
    c = st.columns(5)
    c[0].metric("Longitud (m)", f"{res['L']:.1f}")
    c[1].metric("Σ hf fricción (m)", f"{res['hf']:.2f}")
    c[2].metric("Σ hl locales (m)", f"{res['hl']:.2f}")
    c[3].metric("CDT máx (ND/NAMINO) m", f"{TDH_max:.2f}")
    c[4].metric("CDT mín (NE/NAMO) m", f"{TDH_min:.2f}")
    c = st.columns(5)
    c[0].metric("Potencia hidráulica kW", f"{pot['ph']:.2f}")
    c[1].metric("Potencia al eje kW", f"{pot['p_eje']:.2f}")
    c[2].metric("Motor requerido HP", f"{pot['hp_req']:.1f}")
    c[3].metric("Motor comercial HP", f"{pot['hp_com']}")
    c[4].metric("Potencia eléctrica kW", f"{pot['p_elec']:.2f}")
    # presiones
    st.write(f"Presión máx. operación: **{P_op:.1f} m**, estática: **{P_st:.1f} m**, "
             f"sobrepresión estimada (Joukowsky, cierre instantáneo, a={a_cel} m/s): **{dH_surge:.1f} m** → "
             f"Presión máx. de diseño ≈ **{P_surge:.1f} m ≈ {P_surge / 10:.1f} kg/cm²** → clase mínima **{pn_sel} kg/cm²**.")
    st.caption("Estimación conservadora; MAPAS exige análisis de transitorios (golpe de ariete) con tu software.")
    neg = [(res["ch"][i], res["P"][i]) for i in hi_r if res["P"][i] < 2]
    for c_, p_ in neg:
        msg = f"Punto alto en {c_:.0f} m con presión de operación {p_:.1f} m.c.a. (riesgo de presión negativa/bolsa de aire)."
        st.warning(msg)
        ALERTAS.append(msg)
    if res["P"].min() < 0:
        ALERTAS.append("Existen presiones negativas en operación: la línea piezométrica queda bajo la tubería.")
        st.error("Hay presiones negativas en operación: revisa diámetro, posición de bomba o trazo.")
    fg = fig_perfil(prof, res, acc, hi_r, lo_r, pn_sel * 10)
    st.pyplot(fg)
    FIGS.append(("Figura 1. Perfil, línea de energía y presiones.", fg))
    tbl = pd.DataFrame({"Cadenamiento": res["ch"], "Terreno": res["zt"], "Tubería": res["zp"],
                        "Línea de energía": res["LE"].round(2), "Presión operación": res["P"].round(2),
                        "Presión estática": res["Pst"].round(2)})
    st.dataframe(tbl, height=250)
    st.write("Pérdidas locales por accesorio")
    st.dataframe(res["accr"].rename(columns={"c": "Cad. (m)", "n": "Cant.", "d_mm": "Ø (mm)",
                                             "v": "V (m/s)", "hl": "hl (m)"}).round(3))
    SHEETS["Perfil_hidraulico"] = tbl
    SHEETS["Perdidas_locales"] = res["accr"].round(4)
    # costo energético
    tarifa = st.number_input("Tarifa eléctrica ($/kWh) para costo anual estimado", 0.0, 20.0, 2.5, 0.1)
    kwh_anio = pot["p_elec"] * T * 365
    st.write(f"Energía anual ≈ **{kwh_anio:,.0f} kWh**, costo ≈ **${kwh_anio * tarifa:,.0f}**")

    sec = "Hidráulica"
    add("Datos", "Proyecto", proyecto)
    add("Datos", "Fuente", fuente)
    add("Datos", "Equipo", equipo)
    add("Datos", "Caudal de bombeo", Qlps, "l/s")
    add("Datos", "Horas de bombeo", T, "h/día")
    add("Datos", "Destino", destino)
    add(sec, "Material", mat_name)
    add(sec, "Ecuación de fricción", metodo)
    add(sec, "Rugosidad absoluta", eps, "mm")
    add(sec, "Diámetro nominal / interior", f"{DN} / {D * 1000:.1f}", "mm")
    add(sec, "Velocidad", res["v"], "m/s")
    add(sec, "Longitud", res["L"], "m")
    add(sec, "Pérdidas por fricción", res["hf"], "m")
    add(sec, "Pérdidas locales", res["hl"], "m")
    add(sec, "Cota nivel de descarga", z_dest, "msnm")
    add(sec, "Cota nivel mínimo fuente (ND/NAMINO)", z_low, "msnm")
    add(sec, "Carga dinámica total máxima", TDH_max, "m")
    add(sec, "Carga dinámica total mínima", TDH_min, "m")
    add(sec, "Presión máxima de operación", P_op, "m.c.a.")
    add(sec, "Sobrepresión estimada", dH_surge, "m")
    add(sec, "Clase mínima de tubería", pn_sel, "kg/cm²")
    add("Bomba", "Potencia hidráulica", pot["ph"], "kW")
    add("Bomba", "Eficiencia bomba / motor", f"{eta_b} / {eta_m}")
    add("Bomba", "Potencia al eje", pot["p_eje"], "kW")
    add("Bomba", "Motor requerido (FS)", pot["hp_req"], "HP")
    add("Bomba", "Motor comercial", pot["hp_com"], "HP")
    add("Bomba", "Potencia eléctrica", pot["p_elec"], "kW")

# ------------------------------------------------------------------ 7 TANQUES
with tabs[6]:
    st.write(f"Qmd (definido en la pestaña Datos; con él se diseña el tanque de regulación) = **{Qmd:.2f} l/s**")
    met = st.radio("Método de regulación", ["Coeficiente de regulación (MAPAS, 24 h)", "Curva masa (patrón editable)"])
    if T < 24 and met.startswith("Coef"):
        st.warning("El coeficiente 14.58 corresponde a bombeo continuo 24 h; para menos horas usa curva masa.")
    cum = None
    if met.startswith("Coef"):
        Cr = st.number_input("Coeficiente de regulación Cr (m³ por l/s)", 1.0, 40.0, CR_24H, 0.01)
    else:
        patron = st.data_editor(pd.DataFrame({"Hora": range(24), "Coef. demanda horaria": PATRON_DEMANDA}),
                                key="patron", width="stretch")
        Cr, cum = cr_curva_masa(patron["Coef. demanda horaria"].tolist(), T, int(h0))
        st.write(f"Cr calculado por curva masa = **{Cr:.2f} m³/(l/s)** (referencia 24 h MAPAS: {CR_24H})")
        figm = fig_curva_masa(cum)
        st.pyplot(figm)
        FIGS.append(("Figura 2. Curva masa de regulación.", figm))
    Vreg = Cr * Qmd
    c1, c2 = st.columns(2)
    with c1:
        t_res = st.number_input("Tiempo de almacenamiento de reserva/emergencia (h de Qmd)", 0.0, 72.0, 0.0, 1.0)
        Vres = Qmd * 3.6 * t_res
    with c2:
        h_tq = st.number_input("Tirante útil propuesto (m)", 1.0, 15.0, 3.5, 0.5)
        forma = st.radio("Forma", ["Circular", "Rectangular (largo = 1.5 × ancho)"], horizontal=True)
    pct_el = 100.0
    if destino == "Cisterna + tanque elevado":
        pct_el = st.slider("% del volumen de regulación alojado en el tanque elevado", 0, 100, 40)
    add("Tanques", "Qmd", Qmd, "l/s")
    add("Tanques", "Coeficiente de regulación", Cr, "m³/(l/s)")
    add("Tanques", "Volumen de regulación", Vreg, "m³")
    add("Tanques", "Volumen de reserva", Vres, "m³")

    def dims(V, h):
        A = V / h
        if forma == "Circular":
            return f"Ø {math.sqrt(4 * A / math.pi):.2f} m × {h} m"
        b = math.sqrt(A / 1.5)
        return f"{1.5 * b:.2f} × {b:.2f} m × {h} m"

    tq = []
    if destino == "Cisterna + tanque elevado":
        V_el = Vreg * pct_el / 100
        V_ci = Vreg - V_el + Vres
        tq = [("Tanque elevado", V_el), ("Cisterna", V_ci)]
    elif destino == "Tanque elevado":
        tq = [("Tanque elevado", Vreg + Vres)]
    elif destino == "Cisterna":
        tq = [("Cisterna", Vreg + Vres)]
    else:
        tq = [("Tanque de regulación", Vreg + Vres)]
    rows_t = []
    for n_, v_ in tq:
        rows_t.append([n_, round(v_, 1), dims(v_, h_tq), "Reg. + reserva" if n_ != "Tanque elevado" else "Regulación"])
        add("Tanques", f"{n_} - volumen", v_, "m³")
        add("Tanques", f"{n_} - dimensiones", dims(v_, h_tq))
    tdf = pd.DataFrame(rows_t, columns=["Estructura", "Volumen (m³)", "Dimensiones", "Contenido"])
    st.dataframe(tdf, width="stretch")
    st.info("Ajusta volúmenes comerciales/constructivos. Reserva contra incendio y emergencia: "
            "definir según lineamientos del organismo operador.")
    SHEETS["Tanques"] = tdf

    # ---------------- niveles de las estructuras
    st.markdown("#### Niveles de las estructuras de almacenamiento")
    bordo = st.number_input("Bordo libre (m)", 0.2, 2.0, 0.5, 0.1)

    def estr_(nombre, namo, z_t, recibe=False):
        return dict(nombre=nombre, z_t=z_t, fondo=namo - h_tq, namin=namo - h_tq, namo=namo,
                    corona=namo + bordo, recibe=recibe)

    ESTR = []
    if destino != "Cisterna + tanque elevado":
        nm = {"Tanque de regulación superficial (piso)": "Tanque de regulación (piso)",
              "Tanque elevado": "Tanque elevado", "Cisterna": "Cisterna"}[destino]
        ESTR.append(estr_(nm, z_dest, z_end, True))
    else:
        ESTR.append(estr_(llega_en, z_dest, z_end, True))
        otro = "Cisterna" if llega_en == "Tanque elevado" else "Tanque elevado"
        if otro == "Cisterna":
            prof_ci = st.number_input("Profundidad del fondo de la cisterna bajo el terreno (m)", 0.0, 15.0,
                                      float(h_tq), 0.5)
            ESTR.append(estr_("Cisterna", z_end - prof_ci + h_tq, z_end))
        else:
            torre = st.number_input("Altura del fondo del tanque elevado sobre el terreno (m)", 0.0, 60.0, 12.0, 1.0)
            ESTR.append(estr_("Tanque elevado", z_end + torre + h_tq, z_end))
        st.caption("La línea descarga en la primera estructura; la segunda se ubica en el mismo sitio. "
                   "El rebombeo entre ambas no se calcula en esta app.")
    cn1, cn2 = st.columns([1.2, 1])
    fig_t = fig_tanques(ESTR)
    cn1.pyplot(fig_t)
    FIGS.append(("Figura. Niveles de las estructuras de almacenamiento.", fig_t))
    ndf = pd.DataFrame([[e["nombre"], e["fondo"], e["namo"], e["corona"], e["fondo"] - e["z_t"]] for e in ESTR],
                       columns=["Estructura", "Fondo = NAMIN (msnm)", "NAMO (msnm)", "Corona (msnm)",
                                "Fondo sobre(+)/bajo(−) terreno (m)"]).round(2)
    cn2.dataframe(ndf, hide_index=True, width="stretch")
    for e in ESTR:
        rel = e["fondo"] - e["z_t"]
        if rel > 0.05:
            cn2.caption(f"{e['nombre']}: fondo a {rel:.1f} m sobre el terreno (torre o base).")
        elif rel < -0.05:
            cn2.caption(f"{e['nombre']}: semienterrado, fondo {abs(rel):.1f} m bajo el terreno.")
    SHEETS["Niveles_tanques"] = ndf

    # ---------------- alimentación a la red
    st.markdown("#### Alimentación a la red (se diseña con Qmh)")
    st.write(f"Qmh = **{Qmh:.2f} l/s**. La línea de alimentación y la red quedan fuera del alcance de esta "
             "memoria; se da un diámetro orientativo.")
    dq = pd.DataFrame([[v_, round(math.sqrt(4 * Qmh / 1000 / (math.pi * v_)) * 1000),
                        commercial_up(math.sqrt(4 * Qmh / 1000 / (math.pi * v_)) * 1000)] for v_ in (1.0, 1.5, 2.0)],
                      columns=["Velocidad (m/s)", "Ø teórico (mm)", "DN comercial ≥ (mm)"])
    st.dataframe(dq, hide_index=True)
    add("Tanques", "Qmh (alimentación a red)", float(Qmh), "l/s")

# ------------------------------------------------------------------ 5b ESQUEMA GENERAL Y NIVELES
with tabs[4]:
    st.markdown("---")
    st.subheader("Esquema general de niveles (no a escala; las cotas rotuladas son las calculadas)")
    fsys = fig_sistema(FUENTE_D, res, ESTR)
    st.pyplot(fsys)
    FIGS.append(("Figura 3. Esquema general de niveles: fuente, línea de conducción y destino.", fsys))
    i_hi = int(np.argmax(res["zp"]))
    i_lo = int(np.argmin(res["zp"]))
    NIV = [("Fuente", e_, c_, r_) for e_, c_, r_ in NIV_FUENTE]
    NIV += [
        ("Bombeo", "Cabezal de descarga (inicio de la tubería)", res["zp"][0], "Cota de tubería al inicio del perfil"),
        ("Bombeo", "Energía entregada por la bomba (nivel mín. fuente + CDT máx.)", z_low + TDH_max,
         f"{z_low:.2f} + {TDH_max:.2f} m"),
        ("Bombeo", "Línea de energía a la descarga (tras columna/succión)", res["H0"], "Inicio de la línea de conducción"),
        ("Línea", "Punto más alto de la tubería", res["zp"][i_hi], f"Cad. {res['ch'][i_hi]:.0f} m"),
        ("Línea", "Línea de energía en el punto más alto", res["LE"][i_hi],
         f"Presión de operación {res['P'][i_hi]:.1f} m.c.a."),
        ("Línea", "Punto más bajo de la tubería", res["zp"][i_lo], f"Cad. {res['ch'][i_lo]:.0f} m"),
        ("Línea", "Fin de la tubería (llegada)", res["zp"][-1], f"Cad. {res['ch'][-1]:.0f} m"),
        ("Destino", "Terreno en el punto final", z_end, "Último punto del perfil"),
        ("Destino", "Nivel máximo de descarga (NAMO de la estructura receptora)", z_dest, f"Terreno + h = {h_llegada:.1f} m"),
    ]
    for e in ESTR:
        NIV += [("Destino", f"{e['nombre']}: fondo = NAMIN", e["fondo"], f"{e['fondo'] - e['z_t']:+.2f} m respecto al terreno"),
                ("Destino", f"{e['nombre']}: NAMO", e["namo"], f"Tirante útil {h_tq:.2f} m"),
                ("Destino", f"{e['nombre']}: corona", e["corona"], f"NAMO + bordo libre {bordo:.2f} m")]
    nivdf = pd.DataFrame(NIV, columns=["Grupo", "Elemento", "Cota (msnm)", "Referencia"])
    nivdf["Cota (msnm)"] = nivdf["Cota (msnm)"].astype(float).round(2)
    st.subheader("Tabla de niveles de la instalación")
    st.dataframe(nivdf, hide_index=True, width="stretch")
    SHEETS["Niveles"] = nivdf
    for _, r_ in nivdf.iterrows():
        add("Niveles: " + r_["Grupo"], r_["Elemento"], float(r_["Cota (msnm)"]), "msnm")

# ------------------------------------------------------------------ 8 EXPORTAR
with tabs[7]:
    resumen = pd.DataFrame(RES, columns=["Sección", "Concepto", "Valor", "Unidad"])
    resumen["Valor"] = resumen["Valor"].astype(str)
    st.dataframe(resumen, height=400, width="stretch")
    if ALERTAS:
        st.warning("**Alertas:**\n\n" + "\n".join(f"- {a}" for a in dict.fromkeys(ALERTAS)))
    SHEETS = {"Resumen": resumen, **SHEETS}
    SHEETS["Perfil_entrada"] = prof.rename(columns={"ch": "Cadenamiento (m)", "z": "Cota terreno (msnm)",
                                                 "zp": "Cota tubería (msnm)"})
    SHEETS["Accesorios_tabla"] = acc
    c1, c2, c3 = st.columns(3)
    c1.download_button("Descargar Excel", build_excel(SHEETS), "memoria_bombeo.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("Descargar CSV resumen", resumen.to_csv(index=False).encode("utf-8-sig"),
                       "resumen_bombeo.csv")
    try:
        docx_bytes = build_docx(resumen, list(dict.fromkeys(ALERTAS)), FIGS,
                                f"Memoria técnica descriptiva - {proyecto}")
        c3.download_button("Descargar memoria Word", docx_bytes, "memoria_bombeo.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except ImportError:
        c3.info("Instala python-docx para exportar Word.")
