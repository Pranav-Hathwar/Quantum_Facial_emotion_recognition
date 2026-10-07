import { useState } from "react";
import { Button, Card, ErrorBox, Field, PageHeader, Spinner, StatusPill, inputCls } from "../components/ui";
import { useAuth } from "../hooks/useAuth";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";
import { label } from "../utils/emotions";

export default function Cameras() {
  const { can } = useAuth();
  const cams = useApi(() => api.cameras(), [], 5000);
  const [form, setForm] = useState({ name: "", location: "", source: "webcam" });
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e) => {
    e.preventDefault(); setBusy(true); setErr(null);
    try { await api.createCamera(form); setForm({ name: "", location: "", source: "webcam" }); cams.reload?.(); }
    catch (x) { setErr(x); } finally { setBusy(false); }
  };
  return (
    <>
      <PageHeader title="Cameras" subtitle="Registered camera sources. Select a camera on the Live page to start analysis." />
      <ErrorBox error={cams.error} className="mb-4" />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Cameras" className="lg:col-span-2">
          {!cams.data && cams.loading ? <Spinner /> : (
            <div className="overflow-x-auto"><table className="w-full text-sm">
              <thead><tr className="text-left text-xs uppercase tracking-wider text-muted"><th className="py-2">ID</th><th>Name</th><th>Location</th><th>Source</th><th>Status</th><th>Emotion</th><th>Faces</th><th>Alert</th></tr></thead>
              <tbody className="divide-y divide-line text-ink2">
                {(cams.data || []).map((c) => (
                  <tr key={c.id}><td className="py-2 font-mono text-xs">{c.id}</td><td className="text-ink">{c.name}</td><td>{c.location || "—"}</td><td>{c.source}</td>
                    <td><StatusPill status={c.status} /></td><td>{c.current_emotion ? label(c.current_emotion) : "—"}</td><td className="tabular-nums">{c.face_count}</td><td>{c.alert_state}</td></tr>))}
              </tbody></table></div>)}
        </Card>
        <Card title="Add camera" subtitle={can("ADMIN") ? undefined : "Admin role required"}>
          <form onSubmit={submit} className="space-y-3">
            <Field label="Name"><input className={inputCls} required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} disabled={!can("ADMIN")} /></Field>
            <Field label="Location"><input className={inputCls} value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} disabled={!can("ADMIN")} /></Field>
            <Field label="Source"><input className={inputCls} value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })} disabled={!can("ADMIN")} placeholder="webcam or video file" /></Field>
            <ErrorBox error={err} />
            <Button type="submit" disabled={!can("ADMIN") || busy}>{busy ? "Adding…" : "Add camera"}</Button>
          </form>
        </Card>
      </div>
    </>
  );
}
