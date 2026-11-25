import { Link } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const THRESHOLD = 0.85;

function LiveMonitor({ events }) {
  // events are newest first -> reverse for the chart (oldest on the left)
  const chartData = [...events].reverse().map((e, i) => ({
    n: i + 1,
    time: new Date(e.timestamp).toLocaleTimeString(),
    score: e.max_score,
  }));

  const latest = events[0];
  const showBanner = latest && latest.type === "alert" && latest.status === "pending";

  return (
    <div>
      <h2>Live Monitor</h2>

      {showBanner && (
        <div
          style={{
            background: "#fee2e2",
            border: "1px solid #dc2626",
            color: "#991b1b",
            padding: 12,
            borderRadius: 6,
            marginBottom: 16,
          }}
        >
          <strong>Ransomware detected</strong> (first detection at{" "}
          {latest.detection_point}, score {latest.max_score.toFixed(4)}){" "}
          <Link to={`/alert/${latest.id}`}>View details</Link>
        </div>
      )}

      {events.length === 0 ? (
        <p>Waiting for traces... (run the replayer or POST /predict)</p>
      ) : (
        <>
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={chartData}>
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <ReferenceLine
                  y={THRESHOLD}
                  stroke="red"
                  strokeDasharray="4 4"
                  label="threshold 0.85"
                />
                <Line type="monotone" dataKey="score" stroke="#1a56db" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h3>Latest execution</h3>
          <p>
            {latest.execution || "unnamed execution"} -{" "}
            <strong>{latest.type === "alert" ? "ALERT" : "normal"}</strong>
          </p>
          <ul>
            {Object.entries(latest.scores).map(([fraction, score]) => (
              <li key={fraction}>
                {fraction}: {score.toFixed(4)}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default LiveMonitor;