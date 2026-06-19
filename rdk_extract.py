"""
rdk_extract.py - Pull embedded Python macros out of a RoboDK station into .py files.

Read-only on the .rdk: it loads the station and SAVES each Python program item
to ./macros/<name>.py. It does NOT overwrite the original .rdk.

Usage:
    python rdk_extract.py "241113_AutoScan.rdk"
"""
import os
import sys
from robodk.robolink import ITEM_TYPE_PROGRAM_PYTHON, ITEM_TYPE_PROGRAM
from rdk_session import connect

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "macros")


def main():
    station = sys.argv[1] if len(sys.argv) > 1 else "241113_AutoScan.rdk"
    station_path = station if os.path.isabs(station) else os.path.join(HERE, station)

    os.makedirs(OUT, exist_ok=True)
    RDK = connect()  # private headless instance - never touches your open window
    print("Loading station (this can take a while for a large .rdk)...")
    st = RDK.AddFile(station_path)
    if not st.Valid():
        print("ERROR: could not load", station_path)
        sys.exit(1)
    print("Loaded:", st.Name())

    py_items = RDK.ItemList(ITEM_TYPE_PROGRAM_PYTHON)
    gui_items = RDK.ItemList(ITEM_TYPE_PROGRAM)
    print("Python macros found:", len(py_items))
    print("GUI programs found :", len(gui_items), "(instruction lists, not editable as text)")

    manifest = []
    for it in py_items:
        name = it.Name()
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        dest = os.path.join(OUT, safe + ".py")
        try:
            it.Save(dest)
            ok = os.path.isfile(dest) and os.path.getsize(dest) > 0
            print(("  OK  " if ok else "  ?? "), name, "->", dest)
            manifest.append((name, dest, ok))
        except Exception as e:
            print("  FAIL", name, "->", e)
            manifest.append((name, dest, False))

    print("\nExtracted", sum(1 for _, _, ok in manifest if ok), "of", len(py_items), "macros into", OUT)


if __name__ == "__main__":
    main()
