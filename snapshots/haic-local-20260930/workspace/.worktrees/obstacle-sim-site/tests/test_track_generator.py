import unittest


class TestTrackGenerator(unittest.TestCase):
    def test_same_design_seed_has_same_geometry_and_fingerprint(self):
        from local_simulator.track_generator import (
            custom_geometry_fingerprint,
            generate_custom_map,
        )

        first = generate_custom_map("custom-track-0001", 90421, "hairpin")
        second = generate_custom_map("custom-track-0001", 90421, "hairpin")

        self.assertEqual(first, second)
        self.assertEqual(
            custom_geometry_fingerprint(first.geometry),
            custom_geometry_fingerprint(second.geometry),
        )

    def test_different_design_seed_changes_geometry(self):
        from local_simulator.track_generator import generate_custom_map

        first = generate_custom_map("custom-track-0001", 1, "oval")
        second = generate_custom_map("custom-track-0002", 2, "oval")

        self.assertNotEqual(first.geometry.centerline, second.geometry.centerline)

    def test_self_intersection_is_rejected(self):
        from local_simulator.track_generator import validate_custom_geometry
        from local_simulator.track_model import CustomTrackGeometry

        geometry = CustomTrackGeometry(
            centerline=(
                (0.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 10.0),
                (30.0, 10.0),
                (30.0, 0.0),
                (40.0, 0.0),
                (40.0, 10.0),
                (50.0, 10.0),
                (50.0, 0.0),
            ),
            width=8.0,
        )

        with self.assertRaisesRegex(ValueError, "self-intersection"):
            validate_custom_geometry(geometry)

    def test_batch_generation_uses_distinct_ids(self):
        from local_simulator.track_generator import generate_custom_maps

        maps = generate_custom_maps(20, 3, "chicane")

        self.assertEqual(
            [item.map_id for item in maps],
            ["custom-track-0020", "custom-track-0021", "custom-track-0022"],
        )
        self.assertEqual(len({item.geometry.centerline for item in maps}), 3)

    def test_all_templates_generate_valid_maps(self):
        from local_simulator.track_generator import generate_custom_map

        for template in ("oval", "s_curve", "hairpin", "chicane"):
            with self.subTest(template=template):
                result = generate_custom_map(
                    f"custom-track-{template}",
                    90421,
                    template,
                )
                self.assertGreaterEqual(len(result.geometry.centerline), 12)

    def test_generation_v5_records_distinct_corner_profiles(self):
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        self.assertEqual(
            TEMPLATES,
            frozenset({"oval", "s_curve", "hairpin", "chicane", "technical", "extreme_technical"}),
        )
        profiles = {}
        expected_ranges = {
            "oval": (4, 4),
            "s_curve": (6, 8),
            "hairpin": (6, 9),
            "chicane": (7, 10),
            "technical": (9, 12),
            "extreme_technical": (12, 16),
        }
        for template in sorted(TEMPLATES):
            first = generate_custom_map(f"custom-track-{template}", 90421, template)
            second = generate_custom_map(f"custom-track-{template}", 90421, template)
            first_meta = dict(first.generator)
            second_meta = dict(second.generator)
            self.assertEqual(first, second)
            self.assertEqual(first_meta["generator_version"], 5)
            self.assertEqual(first_meta["corner_count"], len(first_meta["corner_sequence"]))
            self.assertGreaterEqual(first_meta["corner_count"], expected_ranges[template][0])
            self.assertLessEqual(first_meta["corner_count"], expected_ranges[template][1])
            if template == "extreme_technical":
                self.assertIn(first_meta["corner_count"], {12, 14, 16})
            sequence = first_meta["corner_sequence"]
            self.assertEqual(len(sequence), first_meta["corner_count"])
            self.assertTrue(all(token.split(":", 1)[0] in {"left", "right"} for token in sequence))
            if template == "s_curve":
                self.assertTrue(
                    any(
                        sequence[index].split(":", 1)[0]
                        != sequence[index + 1].split(":", 1)[0]
                        for index in range(len(sequence) - 1)
                    )
                )
            elif template == "hairpin":
                self.assertGreaterEqual(sum(token.endswith(":hairpin") for token in sequence), 2)
            elif template == "chicane":
                switches = sum(
                    sequence[index].split(":", 1)[0]
                    != sequence[index + 1].split(":", 1)[0]
                    for index in range(len(sequence) - 1)
                )
                self.assertGreaterEqual(switches, 4)
            elif template == "technical":
                self.assertGreaterEqual(len({token.split(":", 1)[1] for token in sequence}), 3)
            profiles[template] = tuple(first_meta["corner_sequence"])
        self.assertNotEqual(profiles["oval"], profiles["technical"])

    def test_measured_corner_turns_match_the_advertised_profile(self):
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        radius_ranges = {
            "wide": (1.45, 3.2),
            "medium": (1.1, 2.7),
            "tight": (1.45, 2.0),
            "hairpin": (1.45, 1.9),
        }
        for width in (8.0, 9.0):
            for template in sorted(TEMPLATES):
                for seed in (0, 42, 73, 4294967295):
                    generated = generate_custom_map(
                        f"custom-track-turns-{template}",
                        seed,
                        template,
                        width=width,
                    )
                    metadata = dict(generated.generator)
                    sequence = metadata["corner_sequence"]
                    measured_turns = metadata["corner_turn_degrees"]
                    measured_radii = metadata["corner_radius_widths"]
                    self.assertEqual(len(measured_turns), len(sequence))
                    self.assertEqual(len(measured_radii), len(sequence))
                    for token, turn_degrees, radius_widths in zip(sequence, measured_turns, measured_radii):
                        direction, corner_class = token.split(":", 1)
                        low_radius, high_radius = radius_ranges[corner_class]
                        with self.subTest(template=template, seed=seed, width=width, token=token, turn=turn_degrees):
                            self.assertGreater(abs(turn_degrees), 20.0)
                            self.assertGreater(turn_degrees if direction == "left" else -turn_degrees, 0.0)
                            self.assertGreaterEqual(radius_widths, low_radius)
                            self.assertLessEqual(radius_widths, high_radius)
                            if corner_class == "hairpin":
                                self.assertGreaterEqual(abs(turn_degrees), 90.0)

    def test_boundary_seeds_are_repeatable_and_each_template_varies(self):
        from local_simulator.track_generator import generate_custom_map

        for seed in (0, 4294967295):
            first = generate_custom_map(f"custom-track-boundary-{seed}", seed, "technical")
            second = generate_custom_map(f"custom-track-boundary-{seed}", seed, "technical")
            self.assertEqual(first, second)

        for template in ("oval", "s_curve", "hairpin", "chicane", "technical"):
            geometries = {
                generate_custom_map(f"custom-track-{template}-{seed}", seed, template).geometry.centerline
                for seed in range(16)
            }
            with self.subTest(template=template):
                self.assertGreaterEqual(len(geometries), 8)

    def test_sampled_centerline_respects_segment_limit(self):
        import math
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        for template in sorted(TEMPLATES):
            geometry = generate_custom_map(f"custom-track-spacing-{template}", 42, template).geometry
            for index, point in enumerate(geometry.centerline):
                following = geometry.centerline[(index + 1) % len(geometry.centerline)]
                self.assertLessEqual(math.dist(point, following), 4.00001)

    def test_invalid_design_seeds_are_rejected(self):
        from local_simulator.track_generator import generate_custom_map

        for seed in (True, -1, 4294967296, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "design_seed"):
                    generate_custom_map("custom-track-invalid-seed", seed, "oval")

    def test_unknown_template_is_rejected(self):
        from local_simulator.track_generator import generate_custom_map

        with self.assertRaisesRegex(ValueError, "template"):
            generate_custom_map("custom-track-invalid", 1, "unknown")

    def test_rejects_turn_radius_smaller_than_track_half_width(self):
        from local_simulator.track_generator import validate_custom_geometry
        from local_simulator.track_model import CustomTrackGeometry

        points = (
            (-30.0, -30.0), (-15.0, -30.0), (0.0, -30.0), (0.0, 0.0),
            (3.0, 0.0), (3.0, 3.0), (30.0, 3.0), (30.0, 15.0),
            (30.0, 30.0), (0.0, 30.0), (-30.0, 30.0), (-30.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "turn radius"):
            validate_custom_geometry(CustomTrackGeometry(centerline=points, width=8.0))

    def test_generator_handles_width_limits_and_rejects_out_of_range_width(self):
        from local_simulator.track_generator import generate_custom_map

        for width in (0.5, 9.0):
            document = generate_custom_map("custom-track-width", 73, "technical", width=width)
            self.assertTrue(
                all(
                    abs(value) <= 100000
                    for point in document.geometry.centerline
                    for value in point
                )
            )
            for seed in (0, 42, 73):
                generate_custom_map(
                    f"custom-track-extreme-width-{seed}", seed, "extreme_technical", width=width
                )
        for width in (9.01, 100.0):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "width"):
                generate_custom_map("custom-track-invalid-width", 73, "technical", width=width)

    def test_extreme_route_uses_closed_orthogonal_skeleton(self):
        import math
        from local_simulator.track_generator import _build_extreme_route

        observed_counts = set()
        for width in (8.0, 9.0):
            for seed in (0, 1, 42, 73, 300, 1200, 4294967295):
                route = _build_extreme_route(seed, width, attempt_index=0)
                repeated_route = _build_extreme_route(seed, width, attempt_index=0)
                retry_route = _build_extreme_route(seed, width, attempt_index=1)
                count = len(route.vertices)
                observed_counts.add(count)
                self.assertEqual(route, repeated_route)
                self.assertIn(count, {12, 14, 16})
                self.assertEqual(len(route.corner_sequence), count)
                self.assertTrue(all(token.endswith(":tight") for token in route.corner_sequence))
                self.assertEqual(len(retry_route.vertices), count)
                self.assertEqual(len(retry_route.s_section_pairs), len(route.s_section_pairs))

                for index, point in enumerate(route.vertices):
                    following = route.vertices[(index + 1) % count]
                    dx, dy = following[0] - point[0], following[1] - point[1]
                    self.assertGreater(math.hypot(dx, dy), 0.0)
                    self.assertTrue(abs(dx) < 1e-7 or abs(dy) < 1e-7)

                directions = [token.split(":", 1)[0] for token in route.corner_sequence]
                used_corners = set()
                self.assertGreaterEqual(len(route.s_section_pairs), 3)
                for first, second in route.s_section_pairs:
                    self.assertEqual(second, (first + 1) % count)
                    self.assertNotIn(first, used_corners)
                    self.assertNotIn(second, used_corners)
                    self.assertNotEqual(directions[first], directions[second])
                    radius_target = 1.7 * width * (1.0 + 0.4 * max(0.0, min(1.0, (width - 8.0) / 92.0)))
                    tangent_trim = radius_target * math.sqrt(2.0)
                    remaining_straight = math.dist(route.vertices[first], route.vertices[second]) - 2.0 * tangent_trim
                    self.assertGreaterEqual(remaining_straight, 0.6 * width - 1e-5)
                    self.assertLessEqual(remaining_straight, 2.2 * width + 1e-5)
                    used_corners.update((first, second))
        self.assertEqual(observed_counts, {12, 14, 16})

    def test_extreme_generation_reports_measured_corner_metadata(self):
        import math
        from local_simulator.track_generator import (
            _build_extreme_route,
            _generate_extreme_geometry,
            _round_coordinate,
            _round_extreme_route,
            generate_custom_map,
        )

        def tangent_trim(route, index, width):
            vertices = route.vertices
            previous = vertices[index - 1]
            anchor = vertices[index]
            following = vertices[(index + 1) % len(vertices)]
            incoming_length = math.dist(previous, anchor)
            outgoing_length = math.dist(anchor, following)
            incoming = (
                (previous[0] - anchor[0]) / incoming_length,
                (previous[1] - anchor[1]) / incoming_length,
            )
            outgoing = (
                (following[0] - anchor[0]) / outgoing_length,
                (following[1] - anchor[1]) / outgoing_length,
            )
            cosine = max(-1.0, min(1.0, incoming[0] * outgoing[0] + incoming[1] * outgoing[1]))
            half_deflection = (math.pi - math.acos(cosine)) / 2.0
            radius = 1.7 * width * (1.0 + 0.4 * max(0.0, min(1.0, (width - 8.0) / 92.0)))
            trim = radius * math.sin(half_deflection) / math.cos(half_deflection) ** 2
            return min(trim, 0.45 * incoming_length, 0.45 * outgoing_length)

        observed_counts = set()
        for width in (8.0, 9.0):
            for seed in (0, 1, 42, 73, 300, 1200, 4294967295):
                candidate = _generate_extreme_geometry(seed, width)
                self.assertGreaterEqual(len(candidate.s_section_connector_lengths), 3)
                self.assertTrue(all(0.6 * width - 1e-5 <= value <= 2.2 * width + 1e-5 for value in candidate.s_section_connector_lengths))

                route = None
                rounded = None
                for attempt_index in range(64):
                    attempted_route = _build_extreme_route(seed, width, attempt_index)
                    attempted_result = _round_extreme_route(attempted_route, width)
                    if attempted_result.geometry == candidate.geometry:
                        route = attempted_route
                        rounded = attempted_result
                        break
                self.assertIsNotNone(route)
                self.assertIsNotNone(rounded)
                for (first, second), reported_length in zip(
                    rounded.s_section_pairs,
                    rounded.s_section_connector_lengths,
                ):
                    edge_length = math.dist(route.vertices[first], route.vertices[second])
                    first_trim = tangent_trim(route, first, width)
                    second_trim = tangent_trim(route, second, width)
                    first_endpoint = tuple(
                        _round_coordinate(
                            route.vertices[first][axis]
                            + (route.vertices[second][axis] - route.vertices[first][axis])
                            * first_trim / edge_length
                        )
                        for axis in (0, 1)
                    )
                    second_endpoint = tuple(
                        _round_coordinate(
                            route.vertices[second][axis]
                            + (route.vertices[first][axis] - route.vertices[second][axis])
                            * second_trim / edge_length
                        )
                        for axis in (0, 1)
                    )
                    self.assertIn(first_endpoint, candidate.geometry.centerline)
                    self.assertIn(second_endpoint, candidate.geometry.centerline)
                    measured_length = math.dist(first_endpoint, second_endpoint)
                    self.assertAlmostEqual(reported_length, measured_length, places=9)
                    self.assertGreaterEqual(measured_length, 0.6 * width - 1e-5)
                    self.assertLessEqual(measured_length, 2.2 * width + 1e-5)

                document = generate_custom_map(f"custom-track-extreme-{seed}", seed, "extreme_technical", width=width)
                metadata = dict(document.generator)
                count = metadata["corner_count"]
                observed_counts.add(count)
                self.assertIn(count, {12, 14, 16})
                self.assertEqual(len(metadata["corner_sequence"]), count)
                self.assertEqual(len(metadata["corner_turn_degrees"]), count)
                self.assertEqual(len(metadata["corner_radius_widths"]), count)
                self.assertGreaterEqual(metadata["s_section_count"], 3)
                measured_near_90 = sum(75.0 <= abs(turn) <= 105.0 for turn in metadata["corner_turn_degrees"])
                self.assertEqual(metadata["near_90_corner_count"], measured_near_90)
                self.assertGreaterEqual(measured_near_90, 3)
        self.assertEqual(observed_counts, {12, 14, 16})
        repeated = generate_custom_map("custom-track-extreme-repeat", 73, "extreme_technical")
        repeated_again = generate_custom_map("custom-track-extreme-repeat", 73, "extreme_technical")
        self.assertEqual(repeated, repeated_again)
        geometries = {generate_custom_map(f"custom-track-extreme-diversity-{seed}", seed, "extreme_technical").geometry.centerline for seed in range(16)}
        self.assertGreaterEqual(len(geometries), 8)

    def test_generator_retries_enough_candidates_at_maximum_supported_width(self):
        from local_simulator.track_generator import generate_custom_map

        generated = generate_custom_map(
            "custom-track-retry-boundary",
            58,
            "technical",
            width=9.0,
        )
        self.assertEqual(dict(generated.generator)["generator_version"], 5)

    def test_generated_road_boundaries_do_not_intersect(self):
        from local_simulator.track_generator import (
            TEMPLATES,
            _road_edges_intersect,
            generate_custom_map,
        )

        for template in sorted(TEMPLATES):
            document = generate_custom_map(f"custom-track-clearance-{template}", 73, template)
            self.assertFalse(_road_edges_intersect(document.geometry))

    def test_detects_overlapping_road_boundaries(self):
        from local_simulator.track_generator import _road_edges_intersect
        from local_simulator.track_model import CustomTrackGeometry

        narrow_loop = CustomTrackGeometry(
            centerline=(
                (-20.0, -3.0), (-10.0, -3.0), (0.0, -3.0), (10.0, -3.0),
                (20.0, -3.0), (23.0, 0.0), (20.0, 3.0), (10.0, 3.0),
                (0.0, 3.0), (-10.0, 3.0), (-20.0, 3.0), (-23.0, 0.0),
            ),
            width=8.0,
        )
        self.assertTrue(_road_edges_intersect(narrow_loop))

    def test_generated_profiles_validate_for_a_fixed_seed_range(self):
        from local_simulator.track_generator import (
            TEMPLATES,
            generate_custom_map,
            validate_custom_geometry,
        )

        for template in sorted(TEMPLATES):
            for seed in range(100):
                with self.subTest(template=template, seed=seed):
                    document = generate_custom_map(f"custom-track-{template}-{seed}", seed, template)
                    validate_custom_geometry(document.geometry)
                    if template == "extreme_technical":
                        metadata = dict(document.generator)
                        self.assertIn(metadata["corner_count"], {12, 14, 16})
                        self.assertGreaterEqual(metadata["s_section_count"], 3)
                        measured_near_90 = sum(
                            75.0 <= abs(turn) <= 105.0
                            for turn in metadata["corner_turn_degrees"]
                        )
                        self.assertGreaterEqual(measured_near_90, 3)
                        self.assertEqual(metadata["near_90_corner_count"], measured_near_90)


if __name__ == "__main__":
    unittest.main()
