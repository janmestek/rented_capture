"""Kontrola referenčních vektorů po nafocení.

  gcloud storage cp -r "gs://projectcirculate-8b8ce.appspot.com/refs/*" /tmp/refs/
  python3 tool/ref_check.py /tmp/refs

Pro každý snímek pohledu: nejlepší shoda s ostatními snímky téhož pohledu
(leave-one-out) a nejvyšší shoda s jinými pohledy téhož SKU.
Appka zamyká při podobnosti >= 0.65 a odstupu od druhého pohledu > 0.03.
"""
import itertools, json, os, sys

TH = 0.65

def cos(a, b):
    return sum(x * y for x, y in zip(a, b))

root = sys.argv[1]
for sku in sorted(os.listdir(root)):
    views = {}
    for f in sorted(os.listdir(os.path.join(root, sku))):
        d = json.load(open(os.path.join(root, sku, f)))
        views[f[:-5]] = d["golden"] + d.get("learned", [])
    for v, vs in views.items():
        nn = [max((cos(a, b) for j, b in enumerate(vs) if j != i), default=0)
              for i, a in enumerate(vs)]
        ok = sum(x >= TH for x in nn)
        print(f"{sku}/{v}: n={len(vs)} shoda>={TH}: {ok}/{len(vs)}"
              f"{'' if ok * 2 >= len(vs) else '  <- nafotit znovu'}")
    for (v1, a), (v2, b) in itertools.combinations(views.items(), 2):
        m = max(cos(x, y) for x in a for y in b)
        if m >= TH:
            print(f"  ! {sku}: {v1} a {v2} si jsou podobné ({m:.2f})")
