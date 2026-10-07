import { useState } from "react";
import { BaselineCompare } from "../components/Charts";
import { Button, Card, Disclaimer, ErrorBox, PageHeader, SeverityBadge, Spinner, StatusPill, inputCls } from "../components/ui";
import { useAuth } from "../hooks/useAuth";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";
import { label } from "../utils/emotions";

const LEVELS = [
  ["LOW", "Small deviation from this camera's baseline."],
  ["MEDIUM", "Sustained emotional change across the group."],
  ["HIGH", "Large crowd-level anomaly (5+ faces on average)."],
  ["CRITICAL", "Very large and sustained anomaly (8+ faces) — operator review needed."],
];

export default function Alerts() {
  const { can } = useAuth();
  const [status, setStatus] = useState("");
  const [open, setOpen] = useState(null);
  const [note, setNote] = useState("");
  const [err, setErr] = useState(null);
  const list = useApi(() => api.alerts({ status, limit: 200 }), [status], 5000);
  const sel = list.data?.find((a) => a.id === open);

  const act = async (fn) => {
    setErr(null);
    try { await fn(sel.id, note || undefined); setNote(""); list.reload(); } catch (e) { setErr(e); }
  };
  const topEmotions = (dist) => Object.entries(dist || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([e, v]) => `${label(e)} ${(v * 100).toFixed(0)}%`).join(" · ");

  return (
    <>
      <PageHeader title="Alerts" subtitle="Emotional-anomaly alerts: a sustained, group-level shift away from a camera's normal emotion mix. They prompt a human to look — they are not detections of danger or intent." actions={
        <select className={`${inputCls} !w-44`} value={status} onChange={(e) => { setStatus(e.target.value); setOpen(null); }} aria-label="Status filter">
          <option value="">All statuses</option><option>NEW</option><option>ACKNOWLEDGED</option><option>RESOLVED</option></select>} />
      <ErrorBox error={list.error} className="mb-4" />
      <div className="grid gap-6 xl:grid-cols-5">
        <Card title="Alert log" className="xl:col-span-3">
          {list.loading && !list.data ? <Spinner /> : !list.data?.length ? <p className="text-sm text-muted">No alerts{status ? ` with status ${status}` : ""}. The system raises one only after a camera has a baseline and a sustained group-level deviation occurs.</p> : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-muted"><tr><th className="py-2 pr-3">ID</th><th className="pr-3">Severity</th><th className="pr-3">Camera</th><th className="pr-3">Time</th><th className="pr-3">Score</th><th>Status</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {list.data.map((a) => (
                    <tr key={a.id} onClick={() => setOpen(a.id)} className={`cursor-pointer hover:bg-surface2 ${open === a.id ? "bg-accent/10" : ""}`}>
                      <td className="py-2 pr-3 font-mono text-xs text-muted">#{a.id}</td><td className="pr-3"><SeverityBadge severity={a.severity} /></td>
                      <td className="pr-3 text-ink">{a.camera_id}</td><td className="pr-3 text-xs text-ink2">{new Date(a.timestamp).toLocaleString()}</td>
                      <td className="pr-3 tabular-nums text-ink2">{a.anomaly_score.toFixed(2)}</td><td><StatusPill status={a.status} /></td></tr>))}
                </tbody>
              </table>
            </div>)}
        </Card>
        <div className="space-y-6 xl:col-span-2">
          {sel ? (
            <Card title={`Alert #${sel.id} — Emotional Anomaly`} subtitle={`${sel.camera_id} · ${new Date(sel.timestamp).toLocaleString()}`} actions={<SeverityBadge severity={sel.severity} />}>
              <dl className="mb-3 grid grid-cols-3 gap-3 text-sm">
                <div><dt className="text-xs text-muted">Anomaly score</dt><dd className="text-lg font-semibold tabular-nums">{sel.anomaly_score.toFixed(2)}</dd></div>
                <div><dt className="text-xs text-muted">Avg faces / frame</dt><dd className="text-lg font-semibold tabular-nums">{sel.face_count.toFixed(1)}</dd></div>
                <div><dt className="text-xs text-muted">Sustained for</dt><dd className="text-lg font-semibold tabular-nums">{sel.duration_s.toFixed(0)} s</dd></div>
              </dl>
              <p className="mb-1 text-xs text-muted">Detected: <span className="text-ink2">{topEmotions(sel.emotion_distribution)}</span></p>
              <p className="mb-3 text-xs text-muted">Normal baseline: <span className="text-ink2">{topEmotions(sel.baseline_distribution)}</span></p>
              <BaselineCompare current={sel.emotion_distribution} baseline={sel.baseline_distribution} />
              {sel.operator_action && <pre className="mt-3 whitespace-pre-wrap rounded-lg bg-bg p-3 text-xs text-ink2">{sel.operator_action}{sel.handled_by ? `\n— ${sel.handled_by}` : ""}</pre>}
              <ErrorBox error={err} className="mt-3" />
              {can("OPERATOR") ? (sel.status !== "RESOLVED" && (
                <div className="mt-4 space-y-2">
                  <input className={inputCls} placeholder="Operator note (what you checked / did)" value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} />
                  <div className="flex gap-2">
                    {sel.status === "NEW" && <Button variant="ghost" onClick={() => act(api.acknowledge)}>Acknowledge</Button>}
                    <Button onClick={() => act(api.resolve)}>Resolve</Button></div>
                </div>)) : <p className="mt-4 text-xs text-muted">Viewers cannot change alert status.</p>}
              <p className="mt-4 text-xs text-muted">{sel.disclaimer}</p>
            </Card>) : <Card title="Alert details"><p className="text-sm text-muted">Select an alert to review the emotion mix against the baseline.</p></Card>}
          <Card title="Severity levels">
            <ul className="space-y-2">{LEVELS.map(([l, t]) => <li key={l} className="flex items-start gap-3 text-sm"><SeverityBadge severity={l} /><span className="text-ink2">{t}</span></li>)}</ul>
            <p className="mt-3 text-xs text-muted">A single angry, fearful or surprised face never raises an alert. The engine needs ≥3 faces on average, ≥8 s of sustained change, and a trusted per-camera baseline.</p>
          </Card>
        </div>
      </div>
      <div className="mt-6"><Disclaimer /></div>
    </>
  );
}
