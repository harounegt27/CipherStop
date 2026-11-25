import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL;

function AlertDetail({ events }) {
  const { id } = useParams();

  // 1) look for the event in the live list
  const liveEvent = events.find((e) => e.id === id);

  // 2) fallback (page refreshed): load it from GET /history
  const [fetchedEvent, setFetchedEvent] = useState(null);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (liveEvent) return;
    fetch(`${API_URL}/history`)
      .then((res) => res.json())
      .then((list) => {
        const found = list.find((e) => e.id === id);
        if (found) setFetchedEvent(found);
        else setNotFound(true);
      })
      .catch(() => setError("Could not load history from the backend."));
  }, [id, liveEvent]);

  // the live version is always the most recent one
  const event = liveEvent || fetchedEvent;

  async function sendAction(action) {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/alerts/${id}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      if (!res.ok) throw new Error("Request failed");
      const updated = await res.json();
      // if the page was opened after a refresh, update the local copy
      setFetchedEvent(updated);
    } catch (err) {
      setError("Action failed. Is the backend running?");
    } finally {
      setBusy(false);
    }
  }

  if (error && !event) return <p style={{ color: "red" }}>{error}</p>;
  if (notFound) return <p>Alert not found. <Link to="/">Back to Live Monitor</Link></p>;
  if (!event) return <p>Loading...</p>;

  const chartData = Object.entries(event.scores).map(([fraction, score]) => ({
    fraction,
    score,
  }));

  const isPending = event.status === "pending";

  return (
    <div>
      <p><Link to="/">&larr; Back to Live Monitor</Link></p>
      <h2>Alert Detail</h2>

      <ul>
        <li>Execution: {event.execution || "unnamed execution"}</li>
        <li>Time: {new Date(event.timestamp).toLocaleString()}</li>
        <li>Type: {event.type}</li>
        <li>First detection at: {event.detection_point || "none"}</li>
        <li>Max score: {event.max_score.toFixed(4)}</li>
        <li>Status: <strong>{event.status}</strong></li>
      </ul>

      <div style={{ width: "100%", height: 250 }}>
        <ResponsiveContainer>
          <BarChart data={chartData}>
            <XAxis dataKey="fraction" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <ReferenceLine y={event.threshold} stroke="red" strokeDasharray="4 4" />
            <Bar dataKey="score" fill="#1a56db" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: 16 }}>
        <button
          onClick={() => sendAction("isolate")}
          disabled={!isPending || busy}
          style={{ marginRight: 12 }}
        >
          Isolate
        </button>
        <button onClick={() => sendAction("dismiss")} disabled={!isPending || busy}>
          Dismiss
        </button>
        {!isPending && <p>This alert was already handled ({event.status}).</p>}
        {error && <p style={{ color: "red" }}>{error}</p>}
      </div>
    </div>
  );
}

export default AlertDetail;