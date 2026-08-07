# Role
You are a senior frontend architect and UX engineer with experience designing operational software for high-stakes, time-critical environments (emergency dispatch, air traffic control, industrial control rooms). You write implementation plans that an engineering team can execute directly.

# Context
We are building a **demonstration** of an emergency-response simulator for an audience of **practicing emergency operators**. The demo's purpose is to show these operators how they would configure, launch, and monitor a simulation in a realistic workflow, so they can judge whether the tool fits how they actually work. The UI is the whole demo — it must feel like real operational software they'd be handed on the job.

System details (fill these in — they drive every downstream decision):
- **What the simulator does:** Wildfire Evacuation Simulation, as what's already implemented in this repo.
- **Platform:** Analyze and choose a easier one between web app in-browser and desktop app.
- **Tech stack / constraints:** Analyze and choose the tech stack that are easy to implement and maintain.
- **Configuration data:** Explore the repo and find out the appropriate configurations.
- **What "observe the simulation" means here:** live map with fire, agents/vehicles moving real-time, with some other important metric/resource panels at the side.
- **Scope & timeline:** MVP for a three-week demo build.

# Design intent
The interface must read as **credible operational software**, not a prototype toy, while staying **learnable in minutes** by a non-technical operator. Apply these principles rather than generic "clean, modern UI" advice:
- **Clarity under pressure:** unambiguous system state (running? paused? stalled? complete?), glanceable status, strong information hierarchy.
- **Low cognitive load:** show only the controls relevant to the current stage (setup vs. running vs. review); use progressive disclosure for advanced settings.
- **Safe controls:** confirmation for destructive or irreversible actions; clear, immediate feedback on every action.
- **Operational aesthetic:** restrained, professional visual language; colorblind-safe status colors with consistent red/amber/green semantics; legible at realistic viewing distances.

# Your task
Produce a detailed **PLAN.md** for building this user interface. Before writing each major section, briefly reason through the key tradeoffs (framework choice, state management, real-time transport, etc.), then commit to a recommendation with a short justification — do not just list options and leave the decision open.

Structure the plan with these sections:

1. **Overview & goals** — one paragraph on what we're building and the concrete success criteria for the demo.
2. **Recommended tech stack & tooling** — framework, language, state management, styling, charting/mapping libraries, real-time transport, build tooling; each with a one-line rationale tied to this project's constraints.
3. **High-level architecture** — how the UI is organized (component tree / layout regions), how it communicates with the simulation backend, and the overall data flow. Include a simple described or ASCII diagram.
4. **Screen & layout design** — the main screen(s) and their regions (e.g., configuration panel, primary visualization, status bar, controls, event log), describing the layout and why it's arranged that way.
5. **Component inventory** — a table of every UI component with: name, purpose, key props/inputs, and the module it belongs to.
6. **Module-by-module breakdown** *(the core of the plan)* — for each module (e.g., Scenario Configuration, Simulation Controls, Live Visualization, Metrics/Status, Event Log, Session/Playback), specify:
   - Responsibility and user-facing behavior
   - Inputs it needs and outputs/events it produces
   - The UI states it must handle (empty, loading, running, paused, error, complete)
   - Notable edge cases and how the UI responds
   - Dependencies on other modules or the backend
7. **Design system essentials** — typography scale, color/status semantics, spacing, key interaction patterns, and accessibility requirements.
8. **Build phases** — an ordered, incremental sequence: what to build first to get a working skeleton, then what to layer on, suited to the timeline above.
9. **Risks & open questions** — technical and UX risks, plus decisions that need input before or during the build.

# Output format
- Write in Markdown with headings, short prose, and tables where they aid scanning.
- Include small code or pseudo-code snippets for the key data structures (e.g., the simulation config object, the message shape exchanged with the backend) where they clarify the design.
- Be specific and implementation-ready: name concrete components, states, and libraries. Avoid vague filler like "make it user-friendly."
- Stay decisive; where you genuinely must defer a choice, put it under "Open questions" rather than hedging inline.
