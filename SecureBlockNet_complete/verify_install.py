import importlib, sys
mods=['tensorflow','keras','sklearn','numpy','pandas','scipy','matplotlib','cryptography','yaml','joblib']
failed=[]
for m in mods:
    try:
        mod=importlib.import_module(m); print(f'[OK] {m}: {getattr(mod,"__version__","installed")}')
    except Exception as e:
        failed.append((m,str(e))); print(f'[MISSING/ERROR] {m}: {e}')
if failed:
    print('\nInstall dependencies with: pip install -r requirements.txt'); sys.exit(1)
print('\nEnvironment ready.')
