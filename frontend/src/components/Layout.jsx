import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";

const I = {
  dashboard: "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z",
  live: "M15 10l4.5-3v10L15 14M4 6h9a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V8a2 2 0 012-2z",
  image: "M4 5h16a1 1 0 011 1v12a1 1 0 01-1 1H4a1 1 0 01-1-1V6a1 1 0 011-1zm2 11l4-5 3 4 2-2 3 3",
  chart: "M4 20V10m6 10V4m6 16v-7m4 7H2",
  alert: "M12 3l10 18H2L12 3zm0 7v5m0 3v.01",
  atom: "M12 12m-1.5 0a1.5 1.5 0 103 0 1.5 1.5 0 10-3 0M12 12m-10 0c0-2.5 4.5-4.5 10-4.5s10 2 10 4.5-4.5 4.5-10 4.5S2 14.5 2 12zM7 4.5c1.5-2 5.5 1.5 8.5 6.5s4 9.5 2.5 10.5-5.5-1.5-8.5-6.5S5.5 6.5 7 4.5z",
  perf: "M3 3v18h18M7 14l4-4 3 3 5-6",
  camera: "M3 8h4l2-3h6l2 3h4v11H3V8zm9 8a3.5 3.5 0 100-7 3.5 3.5 0 000 7z",
  settings: "M12 15a3 3 0 100-6 3 3 0 000 6zm7.4-3a7.4 7.4 0 00-.1-1.2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 00-2-1.2L14.5 3h-4l-.4 2.6a7.6 7.6 0 00-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 000 2.4l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 002 1.2l.4 2.6h4l.4-2.6a7.6 7.6 0 002-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2z",
};
const Icon = ({ name }) => (
  <svg viewBox="0 0 24 24" className="h-[18px] w-[18px] shrink-0" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d={I[name]} /></svg>
);

const NAV = [
  ["/", "Dashboard", "dashboard"], ["/live", "Live Surveillance", "live"], ["/image", "Image Analysis", "image"],
  ["/analytics", "Emotion Analytics", "chart"], ["/alerts", "Alerts", "alert"], ["/quantum", "Quantum Model", "atom"],
  ["/performance", "Model Performance", "perf"], ["/cameras", "Cameras", "camera"], ["/settings", "Settings", "settings"],
];

export default function Layout() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const { data: health } = useApi(() => api.health(), [], 30000);
  const { data: alerts } = useApi(() => api.alerts({ status: "NEW", limit: 100 }), [], 10000);
  const newAlerts = alerts?.length || 0;
  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-line bg-surface md:flex">
        <div className="border-b border-line px-5 py-4">
          <div className="text-lg font-semibold tracking-tight">QuantumVision</div>
          <div className="text-xs text-muted">Emotion-Aware Surveillance System</div>
        </div>
        <nav className="flex-1 space-y-0.5 overflow-y-auto p-3">
          {NAV.map(([to, text, icon]) => (
            <NavLink key={to} to={to} end={to === "/"}
              className={({ isActive }) => `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${isActive ? "bg-accent/15 text-ink" : "text-ink2 hover:bg-surface2"}`}>
              <Icon name={icon} />
              <span>{text}</span>
              {to === "/alerts" && newAlerts > 0 && <span className="ml-auto rounded-full bg-crit px-2 py-0.5 text-[11px] font-semibold text-white">{newAlerts}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-line p-4 text-xs text-muted">
          <div className="truncate text-ink2">{user?.name}</div>
          <div className="mb-2">{user?.role}</div>
          <button className="text-accent hover:underline" onClick={() => { logout(); nav("/login"); }}>Sign out</button>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-4 border-b border-line bg-surface px-5 py-3 md:px-8">
          <div className="md:hidden font-semibold">QuantumVision</div>
          <p className="hidden text-xs text-muted lg:block">
            Research prototype · Predictions are probabilistic and are not evidence of anyone's mental state, intent or dangerousness · Alerts are statistical anomalies for human review.
          </p>
          <div className="ml-auto flex items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5"><span className={`h-2 w-2 rounded-full ${health?.status === "ok" ? "bg-ok" : "bg-warn"}`} />API {health?.status || "…"}</span>
            <span>{health?.models_available?.length ? `Models: ${health.models_available.join(", ")}` : "No trained model"}</span>
          </div>
        </header>
        <nav className="flex gap-1 overflow-x-auto border-b border-line bg-surface px-3 py-2 md:hidden">
          {NAV.map(([to, text]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `whitespace-nowrap rounded-md px-3 py-1.5 text-xs ${isActive ? "bg-accent/20 text-ink" : "text-ink2"}`}>{text}</NavLink>
          ))}
        </nav>
        <main className="flex-1 overflow-y-auto px-5 py-6 md:px-8">
          <Outlet />
          <footer className="mt-10 border-t border-line pt-4 text-xs text-muted">
            QuantumVision – academic implementation of Chapter 12 (Hybrid Quantum Deep Transfer Learning for Emotion-Aware Surveillance). The quantum circuit runs on a classical simulator; no quantum advantage is claimed.
            {health && !health.store_images && " Raw images are not stored."}
          </footer>
        </main>
      </div>
    </div>
  );
}
