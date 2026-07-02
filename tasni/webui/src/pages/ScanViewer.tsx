// In-browser 3D review of a scan: the fused point cloud + the proposed work
// frame (axes) and rectangle, in Three.js — so the operator decides in the browser
// before anything touches RoboDK (no Open3D/OpenCV popups). The cloud is fetched as
// a compact binary blob from the scan module's /preview.bin endpoint:
//   <uint32 N><float32 N*3 xyz mm><float32 N*3 rgb 0..1>  (little-endian).
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface Props {
  nonce: number;                 // bump to (re)load the latest result
  src: string;                   // preview.bin URL
  frameT?: number[][] | null;    // 4x4 base->work-frame (mm), row-major
  corners?: number[][] | null;   // (4,3) rectangle corners (mm)
}

export default function ScanViewer({ nonce, src, frameT, corners }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const msgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let raf = 0;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f0d);
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1e6);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const resize = () => {
      const w = mount.clientWidth || 600;
      const h = mount.clientHeight || 380;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    mount.appendChild(renderer.domElement);
    resize();
    window.addEventListener("resize", resize);

    const setMsg = (t: string) => { if (msgRef.current) msgRef.current.textContent = t; };
    setMsg("loading 3D preview…");

    (async () => {
      let n = 0, pts: Float32Array, cols: Float32Array;
      try {
        const url = new URL(src, window.location.origin);
        url.searchParams.set("_", String(nonce));
        const buf = await fetch(url, { cache: "no-store" }).then((r) => {
          if (!r.ok) throw new Error(String(r.status));
          return r.arrayBuffer();
        });
        n = new DataView(buf).getUint32(0, true);
        pts = new Float32Array(buf, 4, n * 3);
        cols = new Float32Array(buf, 4 + n * 3 * 4, n * 3);
      } catch {
        setMsg("no 3D preview available");
        return;
      }
      if (disposed) return;
      if (n === 0) { setMsg("scan produced no points"); return; }
      setMsg("");

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
      geo.computeBoundingSphere();
      const bs = geo.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 1);
      const R = bs.radius || 1;
      scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
        size: Math.max(1.5, R / 150), vertexColors: true, sizeAttenuation: true,
      })));

      // Proposed work rectangle (in base/world coords).
      if (corners && corners.length === 4) {
        const cg = new THREE.BufferGeometry().setFromPoints(
          corners.map((c) => new THREE.Vector3(c[0], c[1], c[2])));
        scene.add(new THREE.LineLoop(cg, new THREE.LineBasicMaterial({ color: 0x35c2ff })));
      }
      // Proposed work frame: an axes triad at frame_T (X red, Y green, Z blue).
      if (frameT && frameT.length === 4) {
        const f = frameT, m = new THREE.Matrix4();
        m.set(f[0][0], f[0][1], f[0][2], f[0][3],
              f[1][0], f[1][1], f[1][2], f[1][3],
              f[2][0], f[2][1], f[2][2], f[2][3],
              f[3][0], f[3][1], f[3][2], f[3][3]);
        const axes = new THREE.AxesHelper(R * 0.6);
        axes.matrixAutoUpdate = false;
        axes.matrix.copy(m);
        scene.add(axes);
      }

      // Frame the camera on the cloud.
      const c = bs.center;
      camera.position.set(c.x + R * 1.7, c.y - R * 1.7, c.z + R * 1.4);
      camera.near = Math.max(R / 500, 0.01);
      camera.far = R * 500;
      camera.updateProjectionMatrix();
      controls.target.copy(c);
      controls.update();
    })();

    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
  }, [nonce, src, frameT, corners]);

  return (
    <div style={{ position: "relative", width: "100%", height: 380 }}>
      <div ref={mountRef} style={{ width: "100%", height: "100%", borderRadius: 8, overflow: "hidden" }} />
      <div ref={msgRef} style={{
        position: "absolute", inset: 0, display: "grid", placeItems: "center",
        color: "#9fb8a6", pointerEvents: "none", font: "14px ui-monospace, monospace",
      }} />
    </div>
  );
}
