import type { ComponentType } from "react";
import Calibration from "../pages/Calibration";
import Scan from "../pages/Scan";

// Frontend half of the module system: map a backend module id (from
// /api/modules) to the React page that drives it. Adding a workflow = a backend
// WorkflowModule + a page here + one line below. Modules with a backend but no
// page yet render a friendly placeholder (see ModuleRoute).
export const MODULE_PAGES: Record<string, ComponentType> = {
  calibration: Calibration,
  scan: Scan,
};
