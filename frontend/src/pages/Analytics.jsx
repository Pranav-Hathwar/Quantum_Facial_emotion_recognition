import { useState } from "react";
import { DistributionBars, DistributionDonut, EmotionTimeline } from "../components/Charts";
import { Card, Disclaimer, ErrorBox, PageHeader, Spinner, inputCls } from "../components/ui";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";
import { EMOTION_COLORS, label, pct } from "../utils/emotions";

const WINDOWS = [[15, "Last 15 min"], [60, "Last hour"], [360, "Last 6 hours"], [1440, "Last 24 hours"]];

export default function Analytics() {
  const [camera, setCamera] = useState("");
  const [minutes, setMinutes] = useState(60);
  const cams = useApi(() => api.cameras(), []);
  const bucket = minutes <= 15 ? 15 : minutes <= 60 ? 60 : minutes <= 360 ? 300 : 900;
  const a = useApi(() => api.analytics({ camera_id: camera, minutes, bucket_seconds: bucket }), [camera, minutes], 8000);
  const d = a.data;
  return (
    <>
      <PageHeader title="Emotion Analytics" subtitle="History of predicted facial expressions. Counts are of predictions, not of people." actions={
        <div className="flex gap-2">
          <select className={`${inputCls} !w-52`} value={camera} onChange={(e) => setCamera(e.target.value)} aria-label="Camera">
            <option value="">All cameras and uploads</option>{(cams.data || []).map((c) => <option key={c.id} value={c.id}>{c.id} · {c.name}</option>)}</select>
          <select className={`${inputCls} !w-40`} value={minutes} onChange={(e) => setMinutes(+e.target.value)} aria-label="Time window">
            {WINDOWS.map(([m, t]) => <option key={m} value={m}>{t}</option>)}</select>
        </div>} />
      <ErrorBox error={a.error} className="mb-4" />
      {!d ? <Spinner /> : (
        <div className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-3">
            <Card title="Distribution"><DistributionDonut distribution={d.distribution} /></Card>
            <Card title="Share by emotion" className="lg:col-span-2"><DistributionBars distribution={d.distribution} /></Card>
          </div>
          <Card title="Timeline" subtitle={`${d.total_faces_detected.toLocaleString()} predictions in ${d.total_analyses.toLocaleString()} analyses`}><EmotionTimeline timeline={d.timeline} /></Card>
          <Card title="Recent predictions" subtitle="Most recent 50">
            {!d.recent.length ? <p className="text-sm text-muted">Nothing recorded in this window.</p> : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase tracking-wider text-muted"><tr><th className="py-2 pr-4">Time</th><th className="pr-4">Source</th><th className="pr-4">Face</th><th className="pr-4">Emotion</th><th>Confidence</th></tr></thead>
                  <tbody className="divide-y divide-line">
                    {d.recent.map((r, i) => (
                      <tr key={i}><td className="py-1.5 pr-4 font-mono text-xs text-ink2">{new Date(r.timestamp).toLocaleTimeString()}</td>
                        <td className="pr-4 text-ink2">{r.camera_id || r.source}</td><td className="pr-4 text-ink2">#{r.face_id}</td>
                        <td className="pr-4 font-medium" style={{ color: EMOTION_COLORS[r.emotion] }}>{label(r.emotion)}</td><td className="tabular-nums text-ink2">{pct(r.confidence)}</td></tr>))}
                  </tbody>
                </table>
              </div>)}
          </Card>
          <Disclaimer />
        </div>)}
    </>
  );
}
