import { useEffect, useRef, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL;

export default function useAlertSocket() {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const retryRef = useRef(null);

  useEffect(() => {
    let closedByUs = false;

    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (msg) => {
        const event = JSON.parse(msg.data);

        setEvents((prev) => {
          if (event.type === "alert_update") {
            return prev.map((e) => (e.id === event.id ? event : e));
          }
          return [event, ...prev];
        });
      };

      ws.onclose = () => {
        setConnected(false);
        if (!closedByUs) {
          retryRef.current = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      closedByUs = true;
      clearTimeout(retryRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { events, connected };
}