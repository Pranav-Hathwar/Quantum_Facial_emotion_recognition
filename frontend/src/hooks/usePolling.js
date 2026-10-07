import { useCallback, useEffect, useRef, useState } from "react";

// Load data once and optionally refresh it on an interval. Returns {data, error, loading, reload}.
export function useApi(fn, deps = [], intervalMs = 0) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const reload = useCallback(async () => {
    try {
      const data = await fnRef.current();
      setState({ data, error: null, loading: false });
    } catch (e) {
      setState((s) => ({ data: s.data, error: e, loading: false }));
    }
  }, []);
  useEffect(() => {
    setState((s) => ({ ...s, loading: true }));
    reload();
    if (!intervalMs) return undefined;
    const id = setInterval(reload, intervalMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, intervalMs, reload]);
  return { ...state, reload };
}
