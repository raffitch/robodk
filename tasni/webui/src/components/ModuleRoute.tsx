import { useParams } from "react-router-dom";
import { MODULE_PAGES } from "../modules/registry";

// Renders the React page registered for the :id module, or a placeholder if the
// backend module exists but no UI has been written for it yet.
export default function ModuleRoute() {
  const { id } = useParams();
  const Page = id ? MODULE_PAGES[id] : undefined;
  if (!Page) {
    return (
      <div>
        <h1 className="page-title">{id}</h1>
        <p className="page-sub">
          This module's backend is registered but its UI isn't built yet.
          Add a page and register it in <code>src/modules/registry.ts</code>.
        </p>
      </div>
    );
  }
  return <Page />;
}
