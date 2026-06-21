"""
jetson_deploy.py - Manage the RealSense camera server on the Jetson as a systemd service,
and deploy updates by pulling THIS repo on the Jetson.

Reads creds from ../secrets/jetson.env (git-ignored). Uses the installed SSH key, falling
back to password. sudo uses JETSON_SUDO_PASSWORD.

Commands:
    python tools/jetson_deploy.py bootstrap     # one-time: clone repo, install+enable service, auto-pull, retire dead cron
    python tools/jetson_deploy.py deploy        # git pull on Jetson + restart service
    python tools/jetson_deploy.py setup-autopull # install/refresh the auto-pull timer (Jetson follows origin/main)
    python tools/jetson_deploy.py status        # service + auto-pull timer + port + recent logs
    python tools/jetson_deploy.py logs          # tail journal
    python tools/jetson_deploy.py start|stop|restart
"""
import os
import sys
import time
import paramiko

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SECRETS = os.path.join(HERE, "..", "secrets", "jetson.env")
KEY = os.path.expanduser("~/.ssh/jetson_robodk")
UNIT_LOCAL = os.path.join(HERE, "..", "server", "realsense-camera.service")
AUTOPULL_SH_LOCAL = os.path.join(HERE, "..", "server", "jetson-autopull.sh")
AUTOPULL_SVC_LOCAL = os.path.join(HERE, "..", "server", "jetson-autopull.service")
AUTOPULL_TIMER_LOCAL = os.path.join(HERE, "..", "server", "jetson-autopull.timer")

REPO_URL = "https://github.com/raffitch/robodk.git"
REPO_DIR = "/home/jetson/robodk"
VENV_PY = "/home/jetson/EtherSenseServer/ethenv/bin/python"
SERVER = "/home/jetson/robodk/server/server_unicast_syncronous.py"
UNIT_NAME = "realsense-camera"
AUTOPULL_NAME = "jetson-autopull"
AUTOPULL_BIN = "/usr/local/bin/jetson-autopull.sh"


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


CONN_ERRORS = (paramiko.SSHException, EOFError, ConnectionResetError, OSError)


class Jetson:
    def __init__(self, env):
        self.env = env
        self.sudo_pw = env.get("JETSON_SUDO_PASSWORD", "")
        self.cli = None
        self._connect()

    def _connect(self):
        last = None
        for attempt in range(1, 7):
            try:
                cli = paramiko.SSHClient()
                cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                kw = dict(hostname=self.env["JETSON_HOST"], username=self.env["JETSON_USER"],
                          timeout=15, banner_timeout=15, auth_timeout=15)
                if os.path.isfile(KEY):
                    kw["key_filename"] = KEY
                else:
                    kw["password"] = self.env["JETSON_PASSWORD"]
                cli.connect(**kw)
                self.cli = cli
                return
            except Exception as e:
                last = e
                print(f"  connect attempt {attempt} failed: {e}; retrying...")
                time.sleep(3)
        raise SystemExit(f"Could not connect to Jetson after retries: {last}")

    def run(self, cmd, check=False, quiet=False):
        # Retry across connection drops (flaky link): reconnect and re-run.
        last = None
        for attempt in range(1, 5):
            try:
                _, out, err = self.cli.exec_command(cmd, timeout=180)
                o = out.read().decode(errors="replace")
                e = err.read().decode(errors="replace")
                rc = out.channel.recv_exit_status()
                if not quiet:
                    if o.strip():
                        print(o.rstrip())
                    if e.strip():
                        print(e.rstrip())
                if check and rc != 0:
                    raise SystemExit(f"FAILED (rc={rc}): {cmd}")
                return rc, o, e
            except SystemExit:
                raise
            except CONN_ERRORS as e:
                last = e
                print(f"  [link dropped on attempt {attempt}: {e}] reconnecting...")
                time.sleep(3)
                self._connect()
        raise SystemExit(f"Command kept failing across reconnects: {cmd}\nlast: {last}")

    def sudo(self, cmd, check=False, quiet=False):
        # -S reads password from stdin, -p '' suppresses prompt text
        full = f"echo '{self.sudo_pw}' | sudo -S -p '' bash -c {shq(cmd)}"
        return self.run(full, check=check, quiet=quiet)

    def put(self, local_path, remote_path):
        # Upload a file via SFTP, with reconnect on drop.
        for attempt in range(1, 5):
            try:
                sftp = self.cli.open_sftp()
                sftp.put(local_path, remote_path)
                sftp.close()
                return
            except CONN_ERRORS as e:
                print(f"  [link dropped on upload attempt {attempt}: {e}] reconnecting...")
                time.sleep(3)
                self._connect()
        raise SystemExit(f"Upload kept failing: {local_path} -> {remote_path}")

    def close(self):
        try:
            self.cli.close()
        except Exception:
            pass


def shq(s):
    return "'" + s.replace("'", "'\\''") + "'"


def step(msg):
    print(f"\n=== {msg} ===")


def bootstrap(j):
    step("1/6 Clone or update the monorepo on the Jetson")
    rc, o, _ = j.run(f"test -d {REPO_DIR}/.git && echo EXISTS || echo MISSING", quiet=True)
    if "EXISTS" in o:
        print("repo already present; fetching latest")
        j.run(f"cd {REPO_DIR} && git fetch --quiet origin && git checkout main && git pull --quiet origin main")
    else:
        print("cloning...")
        rc, o, e = j.run(f"git clone {REPO_URL} {REPO_DIR}")
        if rc != 0:
            raise SystemExit("Clone failed - the Jetson likely can't auth to the private repo. "
                             "Fix: add a deploy key or store a GitHub token in ~/.git-credentials on the Jetson.")

    step("2/6 Verify the venv has the camera deps")
    rc, o, _ = j.run(f"{VENV_PY} -c \"import pyrealsense2, turbojpeg, lz4, numpy; print('deps ok')\"")
    if "deps ok" not in o:
        raise SystemExit("venv is missing required deps; aborting before install.")

    step("3/6 Test-run the server briefly (confirm camera + port 1024) BEFORE enabling at boot")
    rc, act, _ = j.run(f"systemctl is-active {UNIT_NAME} 2>/dev/null", quiet=True)
    if act.strip() == "active":
        print("service already active and holding the camera — skipping standalone test-run "
              "(a live service that's LISTENING already proves the camera works).")
    else:
        _test_run_server(j)

    step("4/6 Install + enable the systemd service")
    # Upload unit to /tmp via SFTP (robust), then move into place as root.
    j.put(os.path.abspath(UNIT_LOCAL), f"/tmp/{UNIT_NAME}.service")
    j.sudo(f"mv /tmp/{UNIT_NAME}.service /etc/systemd/system/{UNIT_NAME}.service "
           f"&& chown root:root /etc/systemd/system/{UNIT_NAME}.service "
           f"&& chmod 644 /etc/systemd/system/{UNIT_NAME}.service", check=True, quiet=True)
    j.sudo("systemctl daemon-reload", check=True)
    j.sudo(f"systemctl enable {UNIT_NAME}", check=True)
    j.sudo(f"systemctl restart {UNIT_NAME}", check=True)
    time.sleep(6)

    step("5/6 Verify the service is up and listening")
    j.run(f"systemctl is-active {UNIT_NAME}; systemctl is-enabled {UNIT_NAME}")
    j.run("ss -tln | grep ':1024' && echo LISTENING || echo NOPORT")

    step("6/6 Retire the dead EtherSense cron entries")
    # The legacy autostart lived in /etc/crontab (system-wide), not a user crontab.
    rc, before, _ = j.sudo("grep -c EtherSense /etc/crontab 2>/dev/null || echo 0", quiet=True)
    n = before.strip().splitlines()[-1] if before.strip() else "0"
    if n != "0":
        print(f"removing {n} dead EtherSense line(s) from /etc/crontab (backup: /etc/crontab.pre-cleanup.bak)")
        j.sudo("cp -n /etc/crontab /etc/crontab.pre-cleanup.bak; sed -i '/EtherSense/d' /etc/crontab",
               check=True, quiet=True)
        print("cron cleaned.")
    else:
        print("no dead EtherSense entries in /etc/crontab (already clean).")
    # also check user crontab for completeness
    j.sudo("crontab -l 2>/dev/null | sed -i '/EtherSense/d' - 2>/dev/null; true", quiet=True)

    setup_autopull(j)

    print("\nBOOTSTRAP COMPLETE. The camera server is a systemd service (auto-start on "
          "boot) and the Jetson now auto-pulls origin/main every ~2 min.")


def _test_run_server(j):
    j.run("pkill -f server_unicast_syncronous.py 2>/dev/null; sleep 1", quiet=True)
    j.run(f"nohup {VENV_PY} {SERVER} >/tmp/cam_test.log 2>&1 & sleep 7; true", quiet=True)
    rc, o, _ = j.run("ss -tln | grep ':1024' && echo LISTENING || echo NOPORT", quiet=True)
    _, log, _ = j.run("cat /tmp/cam_test.log", quiet=True)
    j.run("pkill -f server_unicast_syncronous.py 2>/dev/null; true", quiet=True)
    print("server test log:\n" + log.rstrip())
    if "LISTENING" not in o:
        raise SystemExit("Server did not bind port 1024 in test-run; aborting before install. "
                         "Check the log above (camera plugged in / not busy?).")
    print(">> test-run OK: bound port 1024")


def setup_autopull(j):
    """Install/refresh the auto-pull timer so the Jetson follows origin/main on its
    own: every couple of minutes it fetches, and if main moved it hard-resets to it
    (restarting the camera only when server/ changed and no client is mid-capture).

    The puller script is installed to /usr/local/bin (not run from the repo) so that
    even a broken pull can't disable the mechanism that would fix it. Idempotent —
    safe to re-run to push an updated script/units."""
    step("Install the auto-pull timer (Jetson follows origin/main automatically)")
    # CRLF guard: this repo lives on Windows; a \r in the shebang line breaks exec.
    j.put(os.path.abspath(AUTOPULL_SH_LOCAL), "/tmp/jetson-autopull.sh")
    j.put(os.path.abspath(AUTOPULL_SVC_LOCAL), f"/tmp/{AUTOPULL_NAME}.service")
    j.put(os.path.abspath(AUTOPULL_TIMER_LOCAL), f"/tmp/{AUTOPULL_NAME}.timer")
    j.sudo(
        "sed -i 's/\\r$//' /tmp/jetson-autopull.sh "
        f"&& install -m 755 -o root -g root /tmp/jetson-autopull.sh {AUTOPULL_BIN} "
        f"&& install -m 644 -o root -g root /tmp/{AUTOPULL_NAME}.service "
        f"/etc/systemd/system/{AUTOPULL_NAME}.service "
        f"&& install -m 644 -o root -g root /tmp/{AUTOPULL_NAME}.timer "
        f"/etc/systemd/system/{AUTOPULL_NAME}.timer",
        check=True, quiet=True)
    j.sudo("systemctl daemon-reload", check=True)
    j.sudo(f"systemctl enable --now {AUTOPULL_NAME}.timer", check=True)
    print("auto-pull timer installed and enabled.")
    j.run(f"systemctl is-active {AUTOPULL_NAME}.timer; "
          f"systemctl list-timers {AUTOPULL_NAME}.timer --no-pager 2>/dev/null | head -3")


def deploy(j):
    step("Pulling latest on the Jetson and restarting the service")
    j.run(f"cd {REPO_DIR} && git fetch --quiet origin && git checkout main && git pull origin main", check=True)
    j.sudo(f"systemctl restart {UNIT_NAME}", check=True)
    time.sleep(5)
    j.run(f"systemctl is-active {UNIT_NAME}")
    j.run("ss -tln | grep ':1024' && echo LISTENING || echo NOPORT")


def status(j):
    j.run(f"systemctl is-active {UNIT_NAME} 2>/dev/null; systemctl is-enabled {UNIT_NAME} 2>/dev/null")
    j.run("ss -tln | grep ':1024' && echo LISTENING || echo 'NOT listening on 1024'")
    print("--- auto-pull timer ---")
    j.run(f"systemctl is-active {AUTOPULL_NAME}.timer 2>/dev/null; "
          f"systemctl list-timers {AUTOPULL_NAME}.timer --no-pager 2>/dev/null | head -3")
    print("--- recent logs ---")
    j.sudo(f"journalctl -u {UNIT_NAME} -n 15 --no-pager")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    env = load_env(SECRETS)
    j = Jetson(env)
    try:
        if cmd == "bootstrap":
            bootstrap(j)
        elif cmd == "deploy":
            deploy(j)
        elif cmd in ("setup-autopull", "autopull"):
            setup_autopull(j)
        elif cmd == "status":
            status(j)
        elif cmd == "logs":
            j.sudo(f"journalctl -u {UNIT_NAME} -n 40 --no-pager")
        elif cmd in ("start", "stop", "restart"):
            j.sudo(f"systemctl {cmd} {UNIT_NAME}", check=True)
            time.sleep(3)
            status(j)
        else:
            print(f"unknown command: {cmd}")
    finally:
        j.close()


if __name__ == "__main__":
    main()
