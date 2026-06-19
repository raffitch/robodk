// One WebSocket for the whole app. Components subscribe to job events
// (progress / log / frame / result / error / status) via useEvents().
import {
  createContext, useContext, useEffect, useRef, useState, type ReactNode,
} from "react";

export interface JobEvent {
  type: "progress" | "log" | "frame" | "result" | "error" | "status" | string;
  payload: any;
}
type Handler = (e: JobEvent) => void;

interface EventsValue {
  connected: boolean;
  subscribe: (h: Handler) => () => void;
}

const EventsContext = createContext<EventsValue>({
  connected: false,
  subscribe: () => () => {},
});

export function EventsProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const handlers = useRef<Set<Handler>>(new Set());

  useEffect(() => {
    let stopped = false;
    let ws: WebSocket | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!stopped) timer = setTimeout(connect, 1500);
      };
      ws.onmessage = (m) => {
        const ev: JobEvent = JSON.parse(m.data);
        handlers.current.forEach((h) => {
          try { h(ev); } catch (e) { console.error(e); }
        });
      };
    };
    connect();
    return () => { stopped = true; if (timer) clearTimeout(timer); ws?.close(); };
  }, []);

  const subscribe = (h: Handler) => {
    handlers.current.add(h);
    return () => { handlers.current.delete(h); };
  };

  return (
    <EventsContext.Provider value={{ connected, subscribe }}>
      {children}
    </EventsContext.Provider>
  );
}

export const useEvents = () => useContext(EventsContext);
