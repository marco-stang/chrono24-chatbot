"""Konfidenzintervalle für die Eval-Zahlen.

Warum das hier steht: Die Trefferquoten dieses Projekts stammen aus 14 bis 33
Fragen. Eine Punktzahl wie "91 %" suggeriert dort eine Genauigkeit, die die
Stichprobe nicht hergibt -- 30/33 heißt in Wahrheit "irgendwo zwischen 76 % und
97 %". Ohne diese Spanne daneben wurde über mehrere Sitzungen diskutiert, ob
88 % schlechter ist als 94 %; beide liegen im selben Intervall.

Wilson statt des naiven Normal-Intervalls, weil die Ränder real vorkommen:
15/15 und 0/14 sind gemessene Werte, und das Normal-Intervall liefert dort
Grenzen außerhalb von [0, 1] bzw. eine Breite von null.
"""
from __future__ import annotations

from math import sqrt

# 1.96 = 95 % zweiseitig. Bewusst fest verdrahtet: ein konfigurierbares
# Konfidenzniveau lädt dazu ein, es zu senken, bis das Intervall gefällt.
Z_95 = 1.96

# Ab dieser Intervallbreite ist die Zahl keine Messung mehr, sondern eine
# Richtungsangabe. 0.20 entspricht bei 33 Fragen rund sieben Fragen Spielraum.
TOO_WIDE = 0.20


def wilson_interval(hits: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson-Score-Intervall für einen Anteil hits/total.

    Ein leeres Set gibt (0.0, 1.0) zurück -- keine Aussage, nicht "0 %".
    """
    if total < 0 or hits < 0:
        raise ValueError(f"hits und total müssen >= 0 sein, sind {hits} und {total}")
    if hits > total:
        raise ValueError(f"mehr Treffer als Fragen: {hits} von {total}")
    if total == 0:
        return (0.0, 1.0)

    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def format_rate(hits: int, total: int) -> str:
    """Trefferquote mit Intervall, für die Konsolen- und CI-Ausgabe.

    Hängt bei zu breitem Intervall einen Warnhinweis an -- er soll im CI-Log
    auffallen, statt im README-Kleingedruckten zu stehen.
    """
    rate = hits / total if total else 0.0
    low, high = wilson_interval(hits, total)
    text = f"{rate:.0%} ({hits}/{total}, 95%-KI [{low:.0%}, {high:.0%}])"
    if high - low > TOO_WIDE:
        text += f"  <- Stichprobe zu klein: {high - low:.0%} Punkte breit"
    return text
