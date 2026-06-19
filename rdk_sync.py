"""
rdk_sync.py - Push edited .py macros from ./macros back INTO a RoboDK station.

Strategy (robust): for each macros/<name>.py, delete the existing Python program
item of that name and re-add the edited .py via AddFile (RoboDK imports a .py as a
Python program item). Then save to a NEW .rdk so the source is never clobbered
unless you pass --inplace.

Usage:
    python rdk_sync.py "241113_AutoScan.rdk"            # -> 241113_AutoScan.synced.rdk
    python rdk_sync.py "241113_AutoScan.rdk" --inplace  # overwrite the source .rdk
"""
import os
import sys
from robodk.robolink import ITEM_TYPE_PROGRAM_PYTHON
from rdk_session import connect

HERE = os.path.dirname(os.path.abspath(__file__))
MAC = os.path.join(HERE, "macros")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    inplace = "--inplace" in sys.argv
    station = args[0] if args else "241113_AutoScan.rdk"
    station_path = station if os.path.isabs(station) else os.path.join(HERE, station)

    RDK = connect()  # private headless instance - never touches your open window
    print("Loading station...")
    st = RDK.AddFile(station_path)
    if not st.Valid():
        print("ERROR: could not load", station_path)
        sys.exit(1)

    # existing python items by name
    existing = {it.Name(): it for it in RDK.ItemList(ITEM_TYPE_PROGRAM_PYTHON)}

    updated = 0
    for fname in sorted(os.listdir(MAC)):
        if not fname.endswith(".py"):
            continue
        name = fname[:-3]
        path = os.path.join(MAC, fname)
        old = existing.get(name)
        parent = old.Parent() if old is not None else 0
        if old is not None:
            old.Delete()
        new = RDK.AddFile(path, parent)
        if new.Valid():
            print("  updated:", name)
            updated += 1
        else:
            print("  FAIL   :", name)

    out = station_path if inplace else station_path[:-4] + ".synced.rdk"
    st.Save(out)
    print(f"\nWrote {updated} macros. Saved station -> {out}")


if __name__ == "__main__":
    main()
