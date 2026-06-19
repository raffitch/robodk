import { Routes, Route } from "react-router-dom";
import { EventsProvider } from "./api/events";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import ModuleRoute from "./components/ModuleRoute";

export default function App() {
  return (
    <EventsProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/m/:id" element={<ModuleRoute />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </Layout>
    </EventsProvider>
  );
}
