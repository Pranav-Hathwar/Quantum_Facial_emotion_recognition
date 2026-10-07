import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import { useAuth } from "./hooks/useAuth";
import Alerts from "./pages/Alerts";
import Analytics from "./pages/Analytics";
import Cameras from "./pages/Cameras";
import Dashboard from "./pages/Dashboard";
import ImageAnalysis from "./pages/ImageAnalysis";
import Live from "./pages/Live";
import Login from "./pages/Login";
import ModelPerformance from "./pages/ModelPerformance";
import QuantumModel from "./pages/QuantumModel";
import Settings from "./pages/Settings";

function Protected({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="p-10"><Spinner /></div>;
  return user ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<Protected><Layout /></Protected>}>
        <Route index element={<Dashboard />} />
        <Route path="live" element={<Live />} />
        <Route path="image" element={<ImageAnalysis />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="quantum" element={<QuantumModel />} />
        <Route path="performance" element={<ModelPerformance />} />
        <Route path="cameras" element={<Cameras />} />
        <Route path="settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
