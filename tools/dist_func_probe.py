# Functional probe — run with cwd set to a package root (so `import vidlore` resolves
# to THAT tree). Proves the code imports and the registry resolves to 71 primitives.
import importlib

results = {}
import vidlore.motion_graphics.registry as R
reg = R.REGISTRY
results["registry_file"] = R.__file__
results["registry_count"] = len(reg)
results["registry_callable"] = sum(1 for v in reg.values() if callable(v))
results["required_inputs_keys"] = len(getattr(R, "REQUIRED_INPUTS", {}) or {})

# Each named engine tool must import and expose its key symbol.
checks = [
    ("factual_guard", "guard"),
    ("asset_qa", None),
    ("editorial_qa", None),
    ("qa_autofix", None),
    ("render_quarantine", None),
    ("card_style_guard", None),
    ("period_guard", None),
    ("visual_relevance", None),
    ("music", None),
    ("sfx", None),
]
guard_ok = 0
for mod, sym in checks:
    m = importlib.import_module("vidlore." + mod)
    if sym is None or hasattr(m, sym):
        guard_ok += 1
results["named_tools_import_ok"] = "%d/%d" % (guard_ok, len(checks))

# Visual-variation selector + anti-repetition: class + select() method must exist.
import vidlore.motion_graphics.variants as V
results["variant_selector"] = hasattr(V, "VariantSelector") and hasattr(V.VariantSelector, "select")

# Black-frame repair lives in assemble.py — confirm the symbols are present.
import vidlore.assemble as A
src = ""
try:
    import inspect
    src = inspect.getsource(A)
except Exception:
    pass
results["assemble_has_blackframe"] = ("blackdetect" in src) or ("black_frame" in src) or ("black-frame" in src)

ok = (results["registry_count"] == 71
      and results["registry_callable"] == 71
      and results["named_tools_import_ok"] == "%d/%d" % (len(checks), len(checks))
      and results["variant_selector"])
print("FUNC_PROBE %s" % ("PASS" if ok else "CHECK"))
for k in ("registry_file", "registry_count", "registry_callable", "required_inputs_keys",
          "named_tools_import_ok", "variant_selector", "assemble_has_blackframe"):
    print("   %-22s = %s" % (k, results[k]))
