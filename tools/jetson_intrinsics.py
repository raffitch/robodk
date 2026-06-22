"""Read the RealSense COLOR-stream intrinsics (K + distortion) straight off the
camera on the Jetson, so the configured intrinsics can be verified/updated.

The camera is held by the ``realsense-camera`` systemd service, so this briefly
**stops** it, opens the color stream at the requested resolution, reads the
factory intrinsics, then **restarts** the service. Read-only on the device.

The D4xx color stream reports distortion model ``inverse_brown_conrady``; its
``project_point_to_pixel`` uses the same forward formula as OpenCV's
``projectPoints``, so the 5 coeffs ``[k1,k2,p1,p2,k3]`` are directly usable as
OpenCV ``dist_coeffs`` (and the K is fx/fy/ppx/ppy). A model of ``none``/all-zero
coeffs means the factory considers color already rectified.

Run: python tools/jetson_intrinsics.py [WIDTHxHEIGHT]   # default 1280x720
"""
import base64
import json
import sys

from jetson_deploy import Jetson, load_env, SECRETS, VENV_PY, UNIT_NAME

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROBE = '''
import pyrealsense2 as rs, json, time
W, H = {w}, {h}
res, err = None, ""
for _ in range(8):
    try:
        pipe = rs.pipeline(); cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
        prof = pipe.start(cfg)
        vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vsp.get_intrinsics()
        dev = prof.get_device()
        res = {{"width": intr.width, "height": intr.height,
                "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                "model": str(intr.model), "coeffs": [float(c) for c in intr.coeffs],
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "fw": dev.get_info(rs.camera_info.firmware_version)}}
        pipe.stop()
        break
    except Exception as e:
        err = str(e); time.sleep(1.5)
print("INTRINSICS_JSON " + json.dumps(res) if res else "PROBE_FAILED " + err)
'''


def probe(j: "Jetson", w: int, h: int) -> dict:
    print(f"stopping {UNIT_NAME} to free the camera...")
    j.sudo(f"systemctl stop {UNIT_NAME}", check=True, quiet=True)
    try:
        b64 = base64.b64encode(PROBE.format(w=w, h=h).encode()).decode()
        _, out, _ = j.run(f"echo {b64} | base64 -d | {VENV_PY} -", quiet=True)
    finally:
        print(f"restarting {UNIT_NAME}...")
        j.sudo(f"systemctl start {UNIT_NAME}", quiet=True)
    for line in out.splitlines():
        if line.startswith("INTRINSICS_JSON "):
            return json.loads(line[len("INTRINSICS_JSON "):])
        if line.startswith("PROBE_FAILED"):
            raise SystemExit(f"camera probe failed on the Jetson: {line}")
    raise SystemExit(f"no intrinsics in remote output:\n{out}")


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "1280x720"
    w, h = (int(x) for x in res.lower().split("x"))
    j = Jetson(load_env(SECRETS))
    try:
        info = probe(j, w, h)
    finally:
        j.close()
    k1, k2, p1, p2, k3 = (info["coeffs"] + [0, 0, 0, 0, 0])[:5]
    print("\n=== RealSense color intrinsics ===")
    print(f"device : {info['name']}  serial {info['serial']}  fw {info['fw']}")
    print(f"stream : {info['width']}x{info['height']}  model {info['model']}")
    print(f"K      : fx={info['fx']:.4f} fy={info['fy']:.4f} "
          f"cx(ppx)={info['ppx']:.4f} cy(ppy)={info['ppy']:.4f}")
    print(f"dist   : [{k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f}]")
    print("\nJSON (for config / copy-paste):")
    print(json.dumps({
        "intrinsics_row": [[info["fx"], 0, info["ppx"]],
                           [0, info["fy"], info["ppy"]],
                           [0, 0, 1]],
        "dist_coeffs": [k1, k2, p1, p2, k3],
    }, indent=2))


if __name__ == "__main__":
    main()
