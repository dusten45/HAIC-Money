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

    def test_manual_run_start_action_finish_writes_run(self):
        started = self.request(
            "POST",
            "/api/runs/start",
            {"map": self.custom_payload, "policy": "manual"},
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

    def test_unknown_session_returns_not_found(self):
        with self.assertRaises(HTTPError) as error:
            self.request("POST", "/api/runs/not-a-session/action", {"action": [0, 0, 0]})
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
