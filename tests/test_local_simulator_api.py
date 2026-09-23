import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class TestLocalSimulatorApi(unittest.TestCase):
    def setUp(self):
        from local_simulator.track_generator import generate_custom_map
        from local_simulator.schema import map_to_dict
        from local_simulator.web import create_server

        self.root = tempfile.TemporaryDirectory()
        self.artifacts = tempfile.TemporaryDirectory()
        self.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=Path.cwd(),
            artifact_root=Path(self.artifacts.name),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = "http://127.0.0.1:%s" % self.server.server_address[1]
        self.custom_payload = map_to_dict(
            generate_custom_map("custom-track-api", 7, "oval", max_steps=2)
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.root.cleanup()
        self.artifacts.cleanup()

    def request(self, method, path, payload=None):
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_health_and_custom_map_generation(self):
        response = self.request("GET", "/api/health")
        self.assertEqual(response["status"], "ok")

        response = self.request(
            "POST",
            "/api/maps/generate",
            {
                "map_kind": "custom",
                "map_id": "custom-track-api",
                "design_seed": 7,
                "template": "oval",
            },
        )
        self.assertEqual(response["map_kind"], "custom")
        self.assertGreaterEqual(len(response["geometry"]["centerline"]), 12)

    def test_agent_catalog_lists_selectable_agent_roots(self):
        candidate = Path(self.artifacts.name) / "agents" / "demo-agent"
        candidate.mkdir(parents=True)
        (candidate / "agent.py").write_text("class Agent: pass\n", encoding="utf-8")
        (candidate / "model.pt").write_bytes(b"placeholder")

        response = self.request("GET", "/api/agents")

        selected = next(item for item in response["agents"] if item["id"] == "agents/demo-agent")
        self.assertEqual(selected["name"], "demo-agent")
        self.assertTrue(selected["ready"])
        self.assertTrue(selected["has_model"])

    def test_mutating_requests_require_json_and_same_origin_in_browsers(self):
        body = json.dumps({
            "map_kind": "custom",
            "map_id": "custom-track-origin-check",
            "design_seed": 7,
            "template": "oval",
        }).encode("utf-8")
        requests = (
            ({"Content-Type": "text/plain", "Origin": "http://attacker.example"}, 415),
            ({"Content-Type": "application/json", "Origin": "http://attacker.example"}, 403),
            ({"Content-Type": "application/json", "Origin": self.base_url}, 200),
        )
        for headers, expected_status in requests:
            with self.subTest(headers=headers):
                request = Request(
                    self.base_url + "/api/maps/generate",
                    data=body,
                    headers=headers,
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=10) as response:
                        status = response.status
                except HTTPError as error:
                    status = error.code
                self.assertEqual(status, expected_status)

    def test_server_rejects_non_loopback_bind_without_authentication(self):
        from local_simulator.web import create_server

        with self.assertRaisesRegex(ValueError, "loopback"):
            create_server(
                host="0.0.0.0",
                port=0,
                project_root=Path.cwd(),
                artifact_root=Path(self.artifacts.name),
            )

    def test_technical_map_generation(self):
        response = self.request(
            "POST",
            "/api/maps/generate",
            {
                "map_kind": "custom",
                "map_id": "custom-track-api-technical",
                "design_seed": 42,
                "template": "technical",
            },
        )
        self.assertEqual(response["schema_version"], 2)
        self.assertEqual(response["generator"]["template"], "technical")
        self.assertEqual(response["generator"]["generator_version"], 5)
        self.assertGreaterEqual(response["generator"]["corner_count"], 9)

    def test_extreme_technical_map_generation(self):
        response = self.request(
            "POST",
            "/api/maps/generate",
            {
                "map_kind": "custom",
                "map_id": "custom-track-api-extreme",
                "design_seed": 42,
                "template": "extreme_technical",
            },
        )
        self.assertEqual(response["schema_version"], 2)
        self.assertEqual(response["generator"]["template"], "extreme_technical")
        self.assertEqual(response["generator"]["generator_version"], 5)
        self.assertIn(response["generator"]["corner_count"], {12, 14, 16})
        self.assertGreaterEqual(response["generator"]["s_section_count"], 3)
        self.assertGreaterEqual(response["generator"]["near_90_corner_count"], 3)

    def test_generated_track_rejects_widths_outside_the_safe_limit(self):
        with self.assertRaises(HTTPError) as error:
            self.request(
                "POST",
                "/api/maps/generate",
                {
                    "map_kind": "custom",
                    "map_id": "custom-track-too-wide",
                    "design_seed": 42,
                    "template": "technical",
                    "width": 100,
                },
            )
        self.assertEqual(error.exception.code, 400)

    def test_manual_run_start_action_finish_writes_run(self):
        started = self.request(
            "POST",
            "/api/runs/start",
            {"map": self.custom_payload, "policy": "manual"},
        )
        self.assertEqual(started["track"]["width"], self.custom_payload["geometry"]["width"])
        self.assertEqual(
            len(started["track"]["points"]),
            len(self.custom_payload["geometry"]["centerline"]),
        )
        self.assertEqual(
            [point[2:] for point in started["track"]["points"]],
            self.custom_payload["geometry"]["centerline"],
        )
        stepped = self.request(
            "POST",
            "/api/runs/%s/action" % started["run_id"],
            {"action": [0, 1, 0]},
        )
        self.assertEqual(stepped["step"]["action"], [0.0, 1.0, 0.0])

        finished = self.request(
            "POST",
            "/api/runs/%s/finish" % started["run_id"],
            {},
        )

        self.assertTrue(Path(finished["output"]).exists())
        self.assertEqual(finished["run"]["map_ref"]["map_id"], "custom-track-api")

    def test_baseline_auto_run_returns_and_saves_run_log(self):
        response = self.request(
            "POST",
            "/api/runs/auto",
            {"map": self.custom_payload, "policy": "baseline"},
        )
        self.assertEqual(response["run"]["policy"]["kind"], "baseline")
        self.assertGreater(len(response["steps"]), 0)
        self.assertTrue(Path(response["output"]).exists())

    def test_unknown_session_returns_not_found(self):
        with self.assertRaises(HTTPError) as error:
            self.request("POST", "/api/runs/not-a-session/action", {"action": [0, 0, 0]})
        self.assertEqual(error.exception.code, 404)

    def test_unsafe_custom_geometry_is_rejected_before_save_or_run(self):
        unsafe_map = dict(self.custom_payload)
        unsafe_map["geometry"] = {
            "centerline": [
                [-30, -30], [-15, -30], [0, -30], [0, 0], [3, 0], [3, 3],
                [30, 3], [30, 15], [30, 30], [0, 30], [-30, 30], [-30, 0],
            ],
            "width": 8.0,
            "start_index": 0,
            "direction": 1,
        }
        for path, payload in (
            ("/api/maps/save", {"map": unsafe_map}),
            ("/api/runs/start", {"map": unsafe_map, "policy": "manual"}),
        ):
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as error:
                    self.request("POST", path, payload)
                self.assertEqual(error.exception.code, 400)
        self.assertFalse((Path(self.artifacts.name) / "maps" / "custom-track-api.json").exists())


if __name__ == "__main__":
    unittest.main()
