#!/usr/bin/env python3
"""Exact one-sided McNemar power by enumeration (spec section 8). No API calls."""
import json
from math import comb
def crit(n, a):
    for b in range(n + 1):
        if sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n <= a: return b
    return n + 1
def power(N, pd, th, a):
    tot = 0.0
    for d in range(N + 1):
        pD = comb(N, d) * pd ** d * (1 - pd) ** (N - d)
        if pD < 1e-14: continue
        c = crit(d, a); tot += pD * sum(comb(d, b) * th ** b * (1 - th) ** (d - b) for b in range(c, d + 1))
    return tot
out = {}
for label, pd, ths in [("glm_M8", 22 / 69, (.773, .70, .65)), ("deepseek_M1", 5 / 69, (1.0, .85, .75))]:
    for N in (69, 120, 291):
        for th in ths:
            out[f"{label}|N={N}|theta={th}"] = {f"alpha={.05/k:.4f}": round(power(N, pd, th, .05 / k), 3) for k in (1, 2, 5)}
print(json.dumps(out, indent=1))
