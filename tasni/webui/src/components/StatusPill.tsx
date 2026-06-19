export default function StatusPill({
  label, ok, detail,
}: { label: string; ok: boolean | null | undefined; detail?: string }) {
  const cls = ok === true ? "ok" : ok === false ? "bad" : "unknown";
  const text = ok === true ? "ok" : ok === false ? "down" : "—";
  return (
    <span className="pill" title={detail}>
      <span className={`dot ${cls}`} />
      {label}: {text}
    </span>
  );
}
