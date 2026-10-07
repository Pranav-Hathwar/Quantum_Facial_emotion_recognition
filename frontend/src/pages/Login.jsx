import { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { Button, Disclaimer, ErrorBox, Field, inputCls } from "../components/ui";

export default function Login() {
  const { user, login } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("admin@quantumvision.local");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/" replace />;

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try { await login(email, password); nav("/"); } catch (err) { setError(err); } finally { setBusy(false); }
  };
  return (
    <div className="flex min-h-full items-center justify-center bg-bg px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <h1 className="text-3xl font-semibold tracking-tight">QuantumVision</h1>
          <p className="mt-1 text-sm text-ink2">Hybrid quantum-classical facial emotion recognition for smart-city surveillance</p>
        </div>
        <form onSubmit={submit} className="space-y-4 rounded-xl border border-line bg-surface p-6">
          <Field label="Email"><input className={inputCls} type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="username" /></Field>
          <Field label="Password"><input className={inputCls} type="password" value={password} onChange={(e) => setPassword(e.target.value)} required autoComplete="current-password" /></Field>
          <ErrorBox error={error} />
          <Button type="submit" disabled={busy} className="w-full">{busy ? "Signing in…" : "Sign in"}</Button>
        </form>
        <div className="mt-4"><Disclaimer compact /></div>
        <p className="mt-3 text-center text-xs text-muted">Academic prototype. Roles: Admin, Operator, Viewer.</p>
      </div>
    </div>
  );
}
