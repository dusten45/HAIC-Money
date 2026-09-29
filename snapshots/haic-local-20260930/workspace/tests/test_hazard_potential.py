import pytest


class TestHazardPotentialShaping:
    def test_potential_is_negative_pixel_hazard_risk(self):
        from training.hazard_potential import hazard_potential

        assert hazard_potential(0.7) == pytest.approx(-0.7)

    def test_reducing_risk_yields_the_discounted_potential_bonus(self):
        from training.hazard_potential import hazard_potential_shaping_reward

        reward = hazard_potential_shaping_reward(
            current_risk=0.8,
            next_risk=0.3,
            gamma=0.9,
            scale=2.0,
        )

        # 2 * (0.9 * -0.3 - (-0.8))
        assert reward == pytest.approx(1.06)

    def test_increasing_risk_yields_a_potential_penalty(self):
        from training.hazard_potential import hazard_potential_shaping_reward

        reward = hazard_potential_shaping_reward(
            current_risk=0.2,
            next_risk=0.7,
            gamma=0.9,
            scale=2.0,
        )

        # 2 * (0.9 * -0.7 - (-0.2))
        assert reward == pytest.approx(-0.86)

    @pytest.mark.parametrize("boundary", ["terminated", "truncated"])
    def test_terminal_boundary_zeroes_the_next_potential(self, boundary):
        from training.hazard_potential import hazard_potential_shaping_reward

        reward = hazard_potential_shaping_reward(
            current_risk=0.6,
            next_risk=0.9,
            gamma=0.9,
            scale=2.0,
            **{boundary: True},
        )

        assert reward == pytest.approx(1.2)

    @pytest.mark.parametrize("risk", [-0.001, 1.001, float("nan"), float("inf")])
    @pytest.mark.parametrize("argument", ["current_risk", "next_risk"])
    def test_rejects_risk_outside_unit_interval(self, argument, risk):
        from training.hazard_potential import hazard_potential_shaping_reward

        values = {"current_risk": 0.2, "next_risk": 0.4}
        values[argument] = risk
        with pytest.raises(ValueError, match="risk"):
            hazard_potential_shaping_reward(**values, gamma=0.9, scale=1.0)

    @pytest.mark.parametrize("gamma", [-0.01, 1.01, float("nan"), float("inf")])
    def test_rejects_discount_outside_unit_interval(self, gamma):
        from training.hazard_potential import hazard_potential_shaping_reward

        with pytest.raises(ValueError, match="gamma"):
            hazard_potential_shaping_reward(0.2, 0.4, gamma=gamma, scale=1.0)

    @pytest.mark.parametrize("scale", [-0.01, float("nan"), float("inf")])
    def test_rejects_negative_or_non_finite_scale(self, scale):
        from training.hazard_potential import hazard_potential_shaping_reward

        with pytest.raises(ValueError, match="scale"):
            hazard_potential_shaping_reward(0.2, 0.4, gamma=0.9, scale=scale)
