from pathlib import Path
import unittest


class TestWebAssets(unittest.TestCase):
    def test_site_has_local_assets_and_required_controls(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        self.assertIn('id="map-canvas"', html)
        self.assertIn('id="play-button"', html)
        self.assertIn('id="log-file"', html)
        self.assertIn('src="app.js"', html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
