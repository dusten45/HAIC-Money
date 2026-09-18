import ast
import unittest


PARTICIPANT_ROOT = __file__.replace("\\", "/").rsplit("/", 2)[0]
SERVER_ROOT = PARTICIPANT_ROOT.rsplit("/", 1)[0] + "/2026-HAIC/simulator"
SHARED_FILES = (
    "core/finish_line.py",
    "core/track_variables.py",
    "core/obstacle_contacts.py",
    "core/vendor/car_racing.py",
    "core/vendor/car_dynamics.py",
)


def server_repository_available():
    try:
        with open(SERVER_ROOT + "/core/track_variables.py", "rb"):
            return True
    except OSError:
        return False


@unittest.skipUnless(server_repository_available(), "server repository is not available")
class TestServerParity(unittest.TestCase):
    def test_shared_environment_files_match_server(self):
        for relative_path in SHARED_FILES:
            with self.subTest(path=relative_path):
                with open(PARTICIPANT_ROOT + "/" + relative_path, "rb") as participant:
                    participant_content = participant.read()
                with open(SERVER_ROOT + "/" + relative_path, "rb") as server:
                    server_content = server.read()
                participant_tree = ast.parse(participant_content)
                server_tree = ast.parse(server_content)
                for tree in (participant_tree, server_tree):
                    for node in ast.walk(tree):
                        if hasattr(node, "body") and isinstance(node.body, list):
                            node.body[:] = [
                                item for item in node.body
                                if not (
                                    isinstance(item, ast.Expr)
                                    and isinstance(item.value, ast.Constant)
                                    and isinstance(item.value.value, str)
                                )
                            ]
                self.assertEqual(
                    ast.dump(participant_tree, include_attributes=False),
                    ast.dump(server_tree, include_attributes=False),
                    f"participant file differs from server: {relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
