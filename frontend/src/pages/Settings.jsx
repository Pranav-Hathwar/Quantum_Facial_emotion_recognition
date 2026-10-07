import { useState } from "react";
import { Button, Card, Disclaimer, ErrorBox, Field, PageHeader, Spinner, inputCls } from "../components/ui";
import { useAuth } from "../hooks/useAuth";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";

export default function Settings() {
  const { user, can } = useAuth();
  const health = useApi(() => api.health(), []);
  const info = useApi(() => api.modelInfo(), []);
  const users = useApi(() => (can("ADMIN") ? api.users() : Promise.resolve([])), []);
  const [form, setForm] = useState({ email: "", name: "", password: "", role: "VIEWER" });
  const [err, setErr] = useState(null);
  const add = async (e) => {
    e.preventDefault(); setErr(null);
    try { await api.createUser(form); setForm({ email: "", name: "", password: "", role: "VIEWER" }); users.reload?.(); } catch (x) { setErr(x); }
  };
  return (
    <>
      <PageHeader title="Settings" subtitle="System status, loaded models, and user management." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Account"><dl className="space-y-1 text-sm text-ink2"><div>Signed in as <span className="text-ink">{user?.email}</span></div><div>Role <span className="text-ink">{user?.role}</span></div></dl></Card>
        <Card title="System health">
          {!health.data ? <Spinner /> : <pre className="overflow-auto text-xs text-ink2">{JSON.stringify(health.data, null, 2)}</pre>}
        </Card>
        <Card title="Loaded models" className="lg:col-span-2">
          {!info.data ? <Spinner /> : (info.data.loaded.length ? (
            <ul className="space-y-2 text-sm text-ink2">{info.data.loaded.map((m) => (
              <li key={m.kind}><span className="font-medium text-ink">{m.kind}</span> — {m.error ? <span className="text-crit">{m.error}</span> : <>run {m.run}, mode {m.mode}, {m.head_parameters?.toLocaleString?.() ?? "?"} head params{m.smoke_test_model && <span className="ml-2 text-warn">smoke-test weights (not trained)</span>}</>}</li>))}</ul>
          ) : <p className="text-sm text-muted">No trained checkpoints found in the models directory.</p>)}
        </Card>
        {can("ADMIN") && (
          <>
            <Card title="Users"><ul className="divide-y divide-line text-sm">{(users.data || []).map((u) => <li key={u.id} className="flex justify-between py-2 text-ink2"><span>{u.email}</span><span className="text-xs text-muted">{u.role}</span></li>)}</ul></Card>
            <Card title="Create user">
              <form onSubmit={add} className="space-y-3">
                <Field label="Email"><input type="email" required className={inputCls} value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></Field>
                <Field label="Name"><input required className={inputCls} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
                <Field label="Password"><input type="password" required minLength={8} className={inputCls} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
                <Field label="Role"><select className={inputCls} value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}><option>VIEWER</option><option>OPERATOR</option><option>ADMIN</option></select></Field>
                <ErrorBox error={err} /><Button type="submit">Create user</Button>
              </form>
            </Card>
          </>
        )}
      </div>
      <div className="mt-6"><Disclaimer /></div>
    </>
  );
}
