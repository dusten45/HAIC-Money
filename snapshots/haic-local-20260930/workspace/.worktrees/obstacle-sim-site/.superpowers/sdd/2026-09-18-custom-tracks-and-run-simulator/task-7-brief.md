### Task 7: Web custom-track editor, run controls, and automatic replay

**Files:**
- Modify: web_simulator/index.html
- Modify: web_simulator/styles.css
- Modify: web_simulator/app.js
- Modify: tests/test_web_simulator_assets.py

**Interfaces:**
- Existing window.HAICSimulator replay API remains available.
- Add client functions generateCustomMap(), applyCustomMap(payload), startRun(policy), sendManualAction(action), and finishRun().
- Use same-origin /api/* when available; if /api/health fails, keep file import/replay active and show file-only status.
- Keyboard controls: ArrowLeft/ArrowRight steering, ArrowUp gas, Space brake; key release returns that control to zero.

- [ ] Step 1: Write failing static asset tests.

~~~python
def test_site_contains_custom_generator_and_run_controls(self):
    html = Path("web_simulator/index.html").read_text(encoding="utf-8")
    for element_id in (
        "map-kind", "custom-template", "design-seed", "generate-custom-map",
        "start-agent-run", "start-manual-run", "finish-manual-run",
        "local-api-status",
    ):
        self.assertIn('id="%s"' % element_id, html)

def test_script_contains_api_fallback_and_manual_action_names(self):
    script = Path("web_simulator/app.js").read_text(encoding="utf-8")
    for token in (
        "/api/health", "generateCustomMap", "startRun",
        "sendManualAction", "custom-track",
    ):
        self.assertIn(token, script)
~~~

- [ ] Step 2: Run browser asset tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets -v
~~~

Expected: FAIL because the current page only edits official seed/track fields and imports existing logs.

- [ ] Step 3: Implement the UI flow.

Add a map-kind switch. Official mode preserves current fields. Custom mode shows template, design seed, width, map ID, generate, control-point editing, validation messages, and the existing obstacle table. Render custom centerline/boundaries using the existing world-coordinate projection and keep official/custom obstacle legends distinct.

Add an execution panel with map selector, policy selector, frame recording checkbox, start, pause, step, manual keyboard status, and finish/save. startRun("agent") uses the local API for a full Agent run; startRun("manual") opens a session and sends the current action at a bounded interval. On finish, parse the returned log, add it to comparison, and call the existing playback renderer.

Use AbortController for API requests, render errors in the existing status area, and stop the manual timer on pause, finish, page visibility loss, or error. Do not put Python paths or secrets in the page.

- [ ] Step 4: Run static tests and browser verification.

~~~powershell
node --check web_simulator/app.js
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets -v
~~~

Start the integrated launcher:

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m local_simulator.web --host 127.0.0.1 --port 8765 --project-root .
~~~

Expected: custom map creation, full road rendering, Agent/baseline run completion, manual keyboard action, automatic run.json replay, pause/resume/step, and file-only fallback with the plain static server.

- [ ] Step 5: Commit.

~~~powershell
git add web_simulator/index.html web_simulator/styles.css web_simulator/app.js tests/test_web_simulator_assets.py
git commit -m "feat: connect custom maps and run generation to the web app"
~~~

