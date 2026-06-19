"""
jetson_probe.py - SSH into the 3D-scanner Jetson, install our SSH key for
passwordless access, and gather software/camera-firmware info.

Reads credentials from ../secrets/jetson.env (git-ignored).
Run: python tools/jetson_probe.py
"""
import os
import sys
import paramiko

# Force UTF-8 console so non-ASCII output from the Jetson doesn't crash on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SECRETS = os.path.join(HERE, "..", "secrets", "jetson.env")
PUBKEY = os.path.expanduser("~/.ssh/jetson_robodk.pub")


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


# Discovery commands: (label, command). Non-sudo where possible.
CMDS = [
    ("hostname",            "hostname"),
    ("uname",               "uname -a"),
    ("os-release",          "cat /etc/os-release 2>/dev/null | head -6"),
    ("L4T / JetPack",       "cat /etc/nv_tegra_release 2>/dev/null"),
    ("nvidia-l4t pkg",      "dpkg -l 2>/dev/null | grep -i 'nvidia-l4t-core' | head -3"),
    ("CUDA",                "cat /usr/local/cuda/version.txt 2>/dev/null; nvcc --version 2>/dev/null | tail -2"),
    ("python3",             "python3 --version; which python3"),
    ("pyrealsense2",        "python3 -c 'import pyrealsense2 as rs; print(rs.__version__)' 2>&1"),
    ("librealsense apt",    "dpkg -l 2>/dev/null | grep -i realsense"),
    ("realsense devices",   "rs-enumerate-devices -s 2>/dev/null || rs-enumerate-devices 2>/dev/null | grep -iE 'Name|Serial|Firmware|Product Line|USB' | head -20"),
    ("lsusb (intel cam)",   "lsusb 2>/dev/null | grep -iE 'intel|realsense|8086'"),
    ("listening :1024",     "ss -tlnp 2>/dev/null | grep 1024 || netstat -tlnp 2>/dev/null | grep 1024"),
    ("home contents",       "ls -la ~ 2>/dev/null"),
    ("python scripts in ~", "find ~ -maxdepth 3 -name '*.py' 2>/dev/null | head -30"),
    ("autostart services",  "systemctl list-units --type=service --state=running 2>/dev/null | grep -ivE 'systemd|dbus|networkd|resolved|cron|ssh|udev|getty|accounts|polkit|rsyslog|snapd|wpa| modem|avahi|bluetooth|colord|gdm|thermald|upower|ufw' | head -25"),
    ("user systemd units",  "ls -1 /etc/systemd/system/ 2>/dev/null | grep -vE '\\.wants|\\.target' | head -30"),
    ("uptime / load",       "uptime"),
    ("disk",                "df -h / 2>/dev/null | tail -1"),
]


def main():
    env = load_env(SECRETS)
    host = env["JETSON_HOST"]
    user = env["JETSON_USER"]
    pw = env["JETSON_PASSWORD"]

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {user}@{host} ...")
    cli.connect(host, username=user, password=pw, timeout=15)
    print("Connected.\n")

    # 1) Install our public key for passwordless future access
    install_key = "--no-key" not in sys.argv
    if install_key and os.path.isfile(PUBKEY):
        with open(PUBKEY) as f:
            pub = f.read().strip()
        cmd = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f"grep -qF '{pub}' ~/.ssh/authorized_keys 2>/dev/null || "
            f"echo '{pub}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
            "echo KEY_INSTALLED"
        )
        _, out, err = cli.exec_command(cmd)
        print("[ssh key]", (out.read().decode() + err.read().decode()).strip(), "\n")

    # 2) Run discovery
    for label, cmd in CMDS:
        _, out, err = cli.exec_command(cmd, timeout=30)
        o = out.read().decode(errors="replace").rstrip()
        e = err.read().decode(errors="replace").rstrip()
        print("=" * 70)
        print(f"## {label}")
        print("-" * 70)
        if o:
            print(o)
        if e and not o:
            print("(stderr)", e)
        if not o and not e:
            print("(no output)")
    cli.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
