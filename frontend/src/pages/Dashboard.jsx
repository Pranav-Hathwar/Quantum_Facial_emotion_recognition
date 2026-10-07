import { Link } from "react-router-dom";
import { DistributionBars, DistributionDonut, EmotionTimeline } from "../components/Charts";
import { Card, Disclaimer, ErrorBox, PageHeader, SeverityBadge, Spinner, StatTile, StatusPill } from "../components/ui";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";
import { EMOTION_COLORS, label, pct } from "../utils/emotions";

export default function Dashboard() {
  const a = useApi(() => api.analytics({ minutes: 60, bucket_seconds: 60 }), [], 5000);
  const alerts = useApi(() => api.alerts({ status: "NEW", limit: 5 }), [], 5000);
  const cams = useApi(() => api.cameras(), [], 5000);
  const d = a.data;
  return (
    <>
      <PageHeader title="Dashboard" subtitle="Overview of the last 60 minutes across all cameras and uploads." />
      <ErrorBox error={a.error} className="mb-4" />
      {!d && a.loading ? <Spinner /> : d && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <StatTile label="Faces detected" value={d.total_faces_detected.toLocaleString()} />
            <StatTile label="Analyses" value={d.total_analyses.toLocaleString()} hint="images + frames" />
            <StatTile label="Dominant emotion" value={d.dominant_emotion ? <span style={{ color: EMOTION_COLORS[d.dominant_emotion] }}>{label(d.dominant_emotion)}</span> : "—"} />
            <StatTile label="Active alerts" value={d.active_alerts} tone={d.active_alerts ? "alert" : undefined} />
            <StatTile label="Cameras online" value={`${d.connected_cameras}/${d.total_cameras}`} />
          </div>
          <div className="grid gap-6 lg:grid-cols-3">
            <Card title="Emotion distribution" className="lg:col-span-1"><DistributionDonut distribution={d.distribution} /></Card>
            <Card title="Share by emotion" className="lg:col-span-2"><DistributionBars distribution={d.distribution} /></Card>
          </div>
          <Card title="Emotion timeline" subtitle="Faces per emotion per minute"><EmotionTimeline timeline={d.timeline} /></Card>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Latest alerts" actions={<Link to="/alerts" className="text-xs text-accent hover:underline">View all</Link>}>
              {!alerts.data?.length ? <p className="text-sm text-muted">No new emotional-anomaly alerts.</p> : (
                <ul className="divide-y divide-line">
                  {alerts.data.map((al) => (
                    <li key={al.id} className="flex items-center gap-3 py-2.5 text-sm">
                      <SeverityBadge severity={al.severity} />
                      <span className="text-ink">{al.camera_id}</span>
                      <span className="text-muted">anomaly score {al.anomaly_score.toFixed(2)}</span>
                      <span className="ml-auto text-xs text-muted">{new Date(al.timestamp).toLocaleTimeString()}</span>
                    </li>))}
                </ul>)}
            </Card>
            <Card title="Cameras" actions={<Link to="/cameras" className="text-xs text-accent hover:underline">Manage</Link>}>
              <ul className="divide-y divide-line">
                {(cams.data || []).map((c) => (
                  <li key={c.id} className="flex items-center gap-3 py-2.5 text-sm">
                    <span className="font-mono text-xs text-muted">{c.id}</span><span className="text-ink">{c.name}</span>
                    <span className="ml-auto flex items-center gap-4"><StatusPill status={c.status} /><span className="w-24 text-right text-xs text-ink2">{c.current_emotion ? label(c.current_emotion) : "—"}</span></span>
                  </li>))}
              </ul>
            </Card>
          </div>
          <Disclaimer />
        </div>
      )}
    </>
  );
}
