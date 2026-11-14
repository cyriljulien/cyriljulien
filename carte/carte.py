"""Relief de l'année : chaque jour de travail soulève le terrain, l'équation de la chaleur
lisse le relief, les isolignes en tirent une carte. Sortie : deux SVG (clair, sombre).

Usage : python carte.py <contributions.json> <dossier_fonts> <dossier_sortie>
contributions.json : {"AAAA-MM-JJ": nombre, ...}
"""

import base64
import html
import datetime as dt
import io
import json
import math
import sys
from pathlib import Path

import contourpy
import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont

L, H = 1280, 400  # viewBox
CELLULE = 4  # px par maille du champ
NX, NY = L // CELLULE, H // CELLULE
MOIS = ["JANV.", "FÉVR.", "MARS", "AVR.", "MAI", "JUIN", "JUIL.", "AOÛT", "SEPT.", "OCT.", "NOV.", "DÉC."]
MOIS_LONG = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

THEMES = {
    "claire": dict(fond="#f0ebe3", encre="45,37,32", accent="#d87d5d", doux="#6f645c"),
    "sombre": dict(fond="#1c1a19", encre="237,235,232", accent="#e8926e", doux="#a59d96"),
}


def champ(contribs: dict[str, int], fin: dt.date) -> tuple[np.ndarray, dt.date]:
    """Sources gaussiennes posées sur la grille semaines x jours, puis diffusion."""
    debut = fin - dt.timedelta(days=364)
    debut -= dt.timedelta(days=debut.weekday())  # lundi
    nb_sem = (fin - debut).days // 7 + 1
    z = np.zeros((NY, NX))
    marge_x, marge_y = 0.04 * NX, 0.22 * NY
    for i in range((fin - debut).days + 1):
        jour = debut + dt.timedelta(days=i)
        n = contribs.get(jour.isoformat(), 0)
        if n <= 0:
            continue
        sem, wd = i // 7, jour.weekday()
        x = marge_x + sem * (NX - 2 * marge_x) / max(nb_sem - 1, 1)
        y = marge_y + wd * (NY - 2 * marge_y) / 6
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < NX and 0 <= iy < NY:
            z[iy, ix] += math.log1p(n)
    # Relief de fond déterministe, pour que les semaines calmes restent un paysage.
    rng = np.random.default_rng(fin.year)
    yy, xx = np.mgrid[0:NY, 0:NX] / NX
    for _ in range(6):
        kx, ky, ph = rng.uniform(2, 7), rng.uniform(2, 7), rng.uniform(0, 2 * np.pi)
        z += 0.012 * np.sin(2 * np.pi * (kx * xx + ky * yy) + ph)
    # Équation de la chaleur, schéma explicite, bords de Neumann.
    alpha = 0.24
    for _ in range(520):
        p = np.pad(z, 1, mode="edge")
        z = z + alpha * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * z)
    z -= z.min()
    if z.max() > 0:
        z /= z.max()
    return z ** 0.8, debut


def chemins(z: np.ndarray, niveau: float) -> str:
    gen = contourpy.contour_generator(z=z, line_type=contourpy.LineType.Separate)
    d = []
    for ligne in gen.lines(niveau):
        if len(ligne) < 6:
            continue
        pts = ligne * CELLULE + CELLULE / 2
        d.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts))
    return " ".join(d)


def police(chemin: Path, texte: str, wght: int | None = None) -> str:
    f = TTFont(chemin)
    if wght is not None and "fvar" in f:
        from fontTools.varLib import instancer

        f = instancer.instantiateVariableFont(f, {"wght": wght})
    opts = subset.Options()
    opts.layout_features = ["kern", "liga"]
    s = subset.Subsetter(opts)
    s.populate(text=texte)
    s.subset(f)
    buf = io.BytesIO()
    f.flavor = "woff2"
    f.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


def svg(z, debut, fin, theme, polices, textes) -> str:
    t = THEMES[theme]
    niveaux = np.linspace(0.04, 0.98, 26)
    lignes = []
    for k, niv in enumerate(niveaux):
        d = chemins(z, niv)
        if not d:
            continue
        haut = niv > 0.8
        maitresse = k % 5 == 0
        couleur = t["accent"] if haut else f"rgba({t['encre']},{0.5 if maitresse else 0.24})"
        ep = 1.3 if maitresse else 0.8
        lignes.append(f'<path d="{d}" stroke="{couleur}" stroke-width="{ep}" fill="none" stroke-linejoin="round"/>')
    # Sommet
    iy, ix = np.unravel_index(np.argmax(z), z.shape)
    sx, sy = ix * CELLULE + CELLULE / 2, iy * CELLULE + CELLULE / 2
    nb_sem = (fin - debut).days // 7 + 1
    sem = round((ix - 0.04 * NX) / (NX - 0.08 * NX) * (nb_sem - 1))
    date_sommet = debut + dt.timedelta(weeks=min(max(sem, 0), nb_sem - 1))
    etiquette = f"SOMMET · {MOIS[date_sommet.month - 1]} {date_sommet.year}"
    # Repères de mois en pied de carte
    reperes = []
    j = dt.date(debut.year, debut.month, 1)
    while j <= fin:
        x = (0.04 * NX + ((j - debut).days / 7) * (NX - 0.08 * NX) / (nb_sem - 1)) * CELLULE
        if j >= debut and x < L - 80:
            reperes.append(
                f'<line x1="{x:.1f}" y1="{H - 30}" x2="{x:.1f}" y2="{H - 22}" stroke="rgba({t["encre"]},0.45)" stroke-width="1"/>'
                f'<text x="{x + 5:.1f}" y="{H - 21}" class="mono">{MOIS[j.month - 1]}</text>'
            )
        j = dt.date(j.year + (j.month == 12), j.month % 12 + 1, 1)
    gar, ita, mono = polices
    textes = {k: html.escape(v) for k, v in textes.items()}
    # Sommet : seulement hors du voile du titre ; étiquette sous le triangle s'il touche la légende.
    if sx < 0.45 * L:
        sommet = ""
    else:
        droite = sx < L - 320
        ey = sy - 8 if sy > 110 else sy + 24
        sommet = (f'<g transform="translate({sx:.1f},{sy:.1f})"><path d="M0,-7 L6,4 L-6,4 Z" fill="{t["accent"]}"/></g>'
                  f'<text x="{sx + (12 if droite else -12):.1f}" y="{ey:.1f}" class="mono" text-anchor="{"start" if droite else "end"}" style="fill:{t["accent"]}">{etiquette}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {L} {H}" width="{L}" height="{H}" role="img" aria-label="{textes['alt']}">
<style>
@font-face{{font-family:G;src:url(data:font/woff2;base64,{gar}) format('woff2')}}
@font-face{{font-family:GI;src:url(data:font/woff2;base64,{ita}) format('woff2')}}
@font-face{{font-family:M;src:url(data:font/woff2;base64,{mono}) format('woff2')}}
.mono{{font-family:M,ui-monospace,monospace;font-size:15px;letter-spacing:.12em;fill:{t['doux']}}}
text{{paint-order:stroke;stroke:{t['fond']};stroke-width:9px;stroke-linejoin:round}}
</style>
<defs>
<clipPath id="cadre"><rect x="0" y="0" width="{L}" height="{H}" rx="20"/></clipPath>
<linearGradient id="voile" x1="0" x2="1" y1="0" y2="0">
<stop offset="0" stop-color="{t['fond']}" stop-opacity="1"/><stop offset=".30" stop-color="{t['fond']}" stop-opacity=".92"/><stop offset=".46" stop-color="{t['fond']}" stop-opacity="0"/>
</linearGradient>
</defs>
<g clip-path="url(#cadre)">
<rect width="{L}" height="{H}" fill="{t['fond']}"/>
{''.join(lignes)}
<rect width="{L}" height="{H}" fill="url(#voile)"/>
{sommet}
<text x="56" y="66" class="mono">{textes['surtitre']}</text>
<text x="52" y="168" style="font-family:G,Georgia,serif;font-size:96px;letter-spacing:-1.5px;fill:rgb({t['encre']})">Cyril Julien</text>
<text x="56" y="214" style="font-family:GI,Georgia,serif;font-size:26px;fill:rgb({t['encre']})">{textes['phrase']}</text>
{''.join(reperes)}
<text x="{L - 56}" y="66" class="mono" text-anchor="end">{textes['legende']}</text>
</g>
</svg>"""


def main():
    contribs = json.loads(Path(sys.argv[1]).read_text()) or {}
    fonts, sortie = Path(sys.argv[2]), Path(sys.argv[3])
    sortie.mkdir(parents=True, exist_ok=True)
    fin = dt.date.fromisoformat(max(contribs)) if contribs else dt.date.today()
    fin = max(fin, dt.date.today())
    z, debut = champ(contribs, fin)
    textes = json.loads((Path(__file__).parent / "textes.json").read_text())
    tout = "".join(textes.values()) + "".join(MOIS) + "SOMMET · 0123456789"
    polices = (
        police(fonts / "EBGaramond[wght].ttf", tout, 400),
        police(fonts / "EBGaramond-Italic[wght].ttf", tout, 400),
        police(fonts / "GeistMono[wght].ttf", tout, 400),
    )
    for theme in THEMES:
        (sortie / f"carte-{theme}.svg").write_text(svg(z, debut, fin, theme, polices, textes))
    print("ok", debut, fin, f"{sum(contribs.values())} contributions")


if __name__ == "__main__":
    main()
