"""Dump the D435i's advanced-mode configuration AS FOUND, over SSH, into the repo.

Read-only on the camera. Run once BEFORE any depth_units/preset change so the
configuration the 2026-08-13 characterisation was measured under is on record
(audit R4.1). The camera server holds the device exclusively, so the service is
stopped for the ~2 s read and started again whatever happens.

    py -3.10 tools/jetson_dump_asfound.py        # -> server/presets/custom-as-found-<date>.json

Connection and sudo come from ``jetson_deploy`` (key from ~/.ssh/jetson_robodk,
password from secrets/jetson.env) rather than a bare ``ssh`` call: sudo on this
device is NOT passwordless, and the link drops often enough that the retry logic
in that helper is the difference between a dump and a half-stopped service.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from jetson_deploy import Jetson, SECRETS, UNIT_NAME, VENV_PY, load_env, shq  # noqa: E402

# Runs inside the Jetson's server venv. Prints ONE json line on stdout.
REMOTE = r'''
import json, sys
import pyrealsense2 as rs
devs = rs.context().query_devices()
if len(devs) == 0:
    sys.exit("no RealSense device")
dev = devs[0]
adv = rs.rs400_advanced_mode(dev)
if not adv.is_enabled():
    sys.exit("advanced mode is not enabled on this device")
print(json.dumps({"serial": dev.get_info(rs.camera_info.serial_number),
                  "firmware": dev.get_info(rs.camera_info.firmware_version),
                  "librealsense": rs.__version__,
                  "advanced_mode": json.loads(adv.serialize_json())}))
'''


def main() -> int:
    j = Jetson(load_env(SECRETS))
    print(f"stopping {UNIT_NAME} for the read (it holds the device exclusively)")
    j.sudo(f"systemctl stop {UNIT_NAME}", check=True)
    try:
        rc, out, err = j.run(f"{VENV_PY} -c {shq(REMOTE)}", quiet=True)
    finally:
        print(f"starting {UNIT_NAME}")
        j.sudo(f"systemctl start {UNIT_NAME}", check=True)
    if rc != 0:
        print(err.strip() or out.strip(), file=sys.stderr)
        return 1
    payload = json.loads(out.strip().splitlines()[-1])
    payload["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = os.path.join(REPO, "server", "presets")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"custom-as-found-{time.strftime('%Y-%m-%d')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    # serialize_json() returns parameters as a FLAT map of "param-<name>" keys
    # (schema version 1) -- there is no nested depth-table block.
    params = payload["advanced_mode"].get("parameters") or {}
    print(f"wrote {path}")
    print(f"serial {payload['serial']} fw {payload['firmware']} "
          f"librealsense {payload['librealsense']} "
          f"param-depthunits {params.get('param-depthunits')} "
          f"param-zunits {params.get('param-zunits')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
