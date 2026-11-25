import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import useAlertSocket from "./hooks/useAlertSocket";
import LiveMonitor from "./pages/LiveMonitor";
import AlertDetail from "./pages/AlertDetail";
import History from "./pages/History";

const linkStyle = ({ isActive }) => ({
  marginRight: 16,
  textDecoration: "none",
  fontWeight: isActive ? "bold" : "normal",
  color: isActive ? "#1a56db" : "#333",
});

function App() {
  // One single WebSocket for the whole app, shared with the pages as props
  const { events, connected } = useAlertSocket();

  return (
    <BrowserRouter>
      <nav
        style={{
          display: "flex",
          alignItems: "center",
          padding: "12px 20px",
          borderBottom: "1px solid #ddd",
        }}
      >
        <strong style={{ marginRight: 24 }}>CipherStop</strong>
        <NavLink to="/" style={linkStyle} end>
          Live Monitor
        </NavLink>
        <NavLink to="/history" style={linkStyle}>
          History
        </NavLink>

        <span style={{ marginLeft: "auto", color: connected ? "green" : "red" }}>
          ● {connected ? "connected" : "disconnected"}
        </span>
      </nav>

      <main style={{ padding: 20, textAlign: "left" }}>
        <Routes>
          <Route path="/" element={<LiveMonitor events={events} />} />
          <Route path="/alert/:id" element={<AlertDetail events={events} />} />
          <Route path="/history" element={<History events={events} />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;