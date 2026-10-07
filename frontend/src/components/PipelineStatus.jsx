export default function PipelineStatus({ stages = [], timing }) {
  return (
    <div>
      <ul className="space-y-1.5">
        {stages.map((s) => {
          const ok = s.status === "ok";
          return (
            <li key={s.stage} className="flex items-center gap-3 text-sm" title={s.note || ""}>
              <span aria-hidden="true" className={`w-4 text-center font-semibold ${ok ? "text-ok" : "text-muted"}`}>{ok ? "✓" : "–"}</span>
              <span className={ok ? "text-ink" : "text-muted line-through decoration-line"}>{s.label}</span>
              <span className={`ml-auto rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                s.domain === "quantum" ? "bg-[#9085e9]/15 text-[#9085e9]" : "bg-accent/15 text-accent"}`}>{s.domain}</span>
              <span className="w-16 text-right font-mono text-xs text-muted">{ok && s.ms != null ? `${s.ms.toFixed(1)} ms` : s.status === "not_applicable" ? "n/a" : ""}</span>
            </li>
          );
        })}
      </ul>
      {timing?.total_ms != null && <p className="mt-3 border-t border-line pt-2 text-xs text-muted">Total {timing.total_ms.toFixed(0)} ms. Quantum stages run as one circuit on a classical simulator.</p>}
    </div>
  );
}
