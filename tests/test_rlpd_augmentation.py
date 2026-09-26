import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from haic.algorithms.rlpd.augment import random_shift


class TestRandomShift(unittest.TestCase):
    def _stacked_coordinates(self, batch=2):
        coordinates = torch.arange(84 * 84, dtype=torch.int64).reshape(84, 84)
        channels = torch.stack(
            [(coordinates + 37 * channel).remainder(256) for channel in range(4)]
        ).to(torch.uint8)
        return channels.unsqueeze(0).expand(batch, -1, -1, -1).clone()

    def test_pad_four_includes_offset_eight_and_keeps_stacked_frames_aligned(self):
        observations = self._stacked_coordinates(batch=2)
        padded = F.pad(observations, (4, 4, 4, 4), mode="replicate")
        expected = torch.stack(
            (
                padded[0, :, 8:92, 0:84],
                padded[1, :, 0:84, 8:92],
            )
        )
        draws = iter((torch.tensor([0, 8]), torch.tensor([8, 0])))

        def draw(low, high, size, *, device, generator):
            self.assertEqual((low, high), (0, 9))
            self.assertEqual(size, (2,))
            self.assertIsNone(generator)
            return next(draws).to(device=device)

        with mock.patch("haic.algorithms.rlpd.augment.torch.randint", side_effect=draw) as randint:
            actual = random_shift(observations, pad=4)

        self.assertEqual(randint.call_count, 2)
        self.assertTrue(torch.equal(actual, expected))

    def test_supplied_generator_is_reproducible_without_changing_global_rng(self):
        observations = self._stacked_coordinates()
        global_state = torch.random.get_rng_state().clone()
        first_generator = torch.Generator().manual_seed(731)
        second_generator = torch.Generator().manual_seed(731)

        first = random_shift(observations, generator=first_generator)
        second = random_shift(observations, generator=second_generator)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_preserves_shape_device_dtype_and_values_for_supported_dtypes(self):
        uint8_input = self._stacked_coordinates()
        float_input = uint8_input.to(torch.float32) / 255.0

        for observations in (uint8_input, float_input):
            with self.subTest(dtype=observations.dtype):
                shifted = random_shift(
                    observations,
                    generator=torch.Generator().manual_seed(52),
                )
                self.assertEqual(shifted.shape, observations.shape)
                self.assertEqual(shifted.device, observations.device)
                self.assertEqual(shifted.dtype, observations.dtype)
                self.assertTrue(torch.isin(shifted, observations.flatten()).all())
                self.assertTrue(torch.all(shifted >= observations.min()))
                self.assertTrue(torch.all(shifted <= observations.max()))
                if observations.dtype == torch.float32:
                    self.assertGreaterEqual(float(shifted.min()), 0.0)
                    self.assertLessEqual(float(shifted.max()), 1.0)

    def test_float_input_remains_differentiable(self):
        observations = torch.linspace(0.0, 1.0, 4 * 84 * 84).reshape(1, 4, 84, 84)
        observations.requires_grad_()

        shifted = random_shift(
            observations,
            generator=torch.Generator().manual_seed(83),
        )
        shifted.sum().backward()

        self.assertIsNotNone(observations.grad)
        self.assertEqual(observations.grad.shape, observations.shape)
        self.assertTrue(torch.isfinite(observations.grad).all())
        self.assertEqual(float(observations.grad.sum()), shifted.numel())

    def test_rejects_wrong_shapes(self):
        for shape in ((4, 84, 84), (1, 3, 84, 84), (1, 4, 83, 84)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                random_shift(torch.zeros(shape, dtype=torch.float32))

    def test_rejects_invalid_padding(self):
        observations = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
        for pad in (-1,):
            with self.subTest(pad=pad), self.assertRaises(ValueError):
                random_shift(observations, pad=pad)
        for pad in (1.5, True, "4"):
            with self.subTest(pad=pad), self.assertRaises(TypeError):
                random_shift(observations, pad=pad)

    def test_rejects_unsupported_dtypes_and_invalid_float_values(self):
        with self.assertRaises(TypeError):
            random_shift(torch.zeros((1, 4, 84, 84), dtype=torch.float64))
        with self.assertRaises(TypeError):
            random_shift(torch.zeros((1, 4, 84, 84), dtype=torch.int16))

        for invalid in (float("nan"), float("inf"), -0.01, 1.01):
            observations = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
            observations[0, 0, 0, 0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                random_shift(observations)


if __name__ == "__main__":
    unittest.main()
