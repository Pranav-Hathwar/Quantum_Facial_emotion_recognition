import { SEVERITY_STYLE } from "../utils/emotions";

export function Card({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-line bg-surface ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-3.5">
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-ink">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, hint, tone }) {
  const toneCls = tone === "alert" ? "text-crit" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-surface px-5 py-4">
      <div className="text-xs font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className={`mt-1.5 text-3xl font-semibold tabular-nums ${toneCls}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold text-ink">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-ink2">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Button({ children, variant = "primary", className = "", ...props }) {
  const styles = {
    primary: "bg-accent text-white hover:brightness-110 disabled:opacity-50",
    ghost: "border border-line text-ink2 hover:bg-surface2 disabled:opacity-50",
    danger: "bg-crit/90 text-white hover:bg-crit disabled:opacity-50",
  };
  return (
    <button {...props} className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed ${styles[variant]} ${className}`}>
      {children}
    </button>
  );
}

export function Spinner({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-3 py-8 text-sm text-muted" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label}
    </div>
  );
}

export function ErrorBox({ error, className = "" }) {
  if (!error) return null;
  return (
    <div role="alert" className={`rounded-lg border border-crit/40 bg-crit/10 px-4 py-3 text-sm text-crit ${className}`}>
      {error.message || String(error)}
    </div>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-6 py-10 text-center">
      <div className="text-sm font-medium text-ink2">{title}</div>
      {children && <div className="mx-auto mt-2 max-w-xl text-sm text-muted">{children}</div>}
    </div>
  );
}

export function SeverityBadge({ severity }) {
  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold tracking-wide ${SEVERITY_STYLE[severity] || "border-line text-muted"}`}>
      {severity}
    </span>
  );
}

export function StatusPill({ status }) {
  const map = { ONLINE: "text-ok", OFFLINE: "text-muted", NEW: "text-crit", ACKNOWLEDGED: "text-warn", RESOLVED: "text-ok" };
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${map[status] || "text-muted"}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {status}
    </span>
  );
}

export function Disclaimer({ children, compact }) {
  return (
    <p className={`rounded-lg border border-line bg-surface2/60 px-4 text-xs leading-relaxed text-muted ${compact ? "py-2" : "py-3"}`}>
      <span className="font-semibold text-ink2">Limitations. </span>
      {children ||
        "Facial emotion predictions are probabilistic estimates of facial expression. They do not establish a person's actual mental state, intent or dangerousness, and are affected by lighting, occlusion, camera angle, image quality, pose, dataset bias and individual differences."}
    </p>
  );
}

export function Field({ label, children }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium uppercase tracking-wider text-muted">{label}</span>
      {children}
    </label>
  );
}

export const inputCls = "w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink placeholder:text-muted";
