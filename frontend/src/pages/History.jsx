import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const API_URL = import.meta.env.VITE_API_URL;

function History({ events }) {
  const [past, setPast] = useState([]);
  const [error, setError] = useState("");
  const [onlyAlerts, setOnlyAlerts] = useState(false);

  // Load the stored history once when the page opens
  useEffect(() => {
    fetch(`${API_URL}/history`)
      .then((res) => res.json())
      .then((list) => setPast(list))
      .catch(() => setError("Could not load history from the backend."));
  }, []);

  // Merge: live events win over stored ones (same id = most recent version)
  const byId = new Map();
  past.forEach((e) => byId.set(e.id, e));
  events.forEach((e) => byId.set(e.id, e));

  const rows = [...byId.values()]
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .filter((e) => (onlyAlerts ? e.type === "alert" : true));

  return (
    <div>
      <h2>History</h2>

      <label>
        <input
          type="checkbox"
          checked={onlyAlerts}
          onChange={(e) => setOnlyAlerts(e.target.checked)}
        />{" "}
        Alerts only
      </label>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {rows.length === 0 ? (
        <p>No events yet.</p>
      ) : (
        <div style={{ overflowX: "auto", marginTop: 12 }}>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "2px solid #ccc" }}>
                <th>Time</th>
                <th>Execution</th>
                <th>Type</th>
                <th>Max score</th>
                <th>Detected at</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((e) => (
                <tr key={e.id} style={{ borderBottom: "1px solid #eee" }}>
                  <td>{new Date(e.timestamp).toLocaleString()}</td>
                  <td>{e.execution || "unnamed"}</td>
                  <td>{e.type}</td>
                  <td>{e.max_score.toFixed(4)}</td>
                  <td>{e.detection_point || "-"}</td>
                  <td>{e.status}</td>
                  <td>
                    {e.type === "alert" && <Link to={`/alert/${e.id}`}>Details</Link>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default History;