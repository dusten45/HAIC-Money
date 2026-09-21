from dataclasses import dataclass

DAMAGE_PER_COLLISION = 0.2
MAX_DAMAGE = 1.0
MIN_GRIP_MULTIPLIER = 0.6
MIN_ENGINE_MULTIPLIER = 0.8
MIN_STEERING_MULTIPLIER = 0.8


@dataclass(frozen=True)
class DamageEffects:
    grip_multiplier: float
    engine_multiplier: float
    steering_multiplier: float


def calculate_damage_effects(damage: float) -> DamageEffects:
    return DamageEffects(
        grip_multiplier=max(MIN_GRIP_MULTIPLIER, 1.0 - damage * 0.5),
        engine_multiplier=max(MIN_ENGINE_MULTIPLIER, 1.0 - damage * 0.25),
        steering_multiplier=max(MIN_STEERING_MULTIPLIER, 1.0 - damage * 0.25),
    )


class CollisionDamage:

    def __init__(self) -> None:
        self.damage = 0.0

    def reset(self) -> None:
        self.damage = 0.0

    def update(self, collision: bool) -> bool:
        if collision:
            self.damage = min(MAX_DAMAGE, self.damage + DAMAGE_PER_COLLISION)
        return self.damage >= MAX_DAMAGE

    @property
    def effects(self) -> DamageEffects:
        return calculate_damage_effects(self.damage)
