import logging
import math
import time
from typing import Dict, Any, Tuple, Optional
import tensorflow as tf

logger = logging.getLogger("forecast_system.lr_policy")

class PerformanceLRDecayPolicy:
    """
    Configurable Policy for Learning Rate Decay and Cyclic Learning Rate (CLR).

    Monitors performance metrics or applies predefined cyclic / plateau decay policies:
    - 'plateau': Multiplies current learning rate by `factor` on performance stagnation.
    - 'plateau_restart': Decays on plateau and triggers a Warm Restart Bump when flat at min_lr.
    - 'exponential_plateau': Exponentially decays learning rate based on decay step count.
    - 'step_plateau': Subtracts a fixed step size from current learning rate.
    - 'cyclic' / 'clr': Cyclical Learning Rate (triangular, triangular2, exp_range) between min_lr and max_lr.
    """
    def __init__(
        self,
        enabled: bool = True,
        policy: str = 'plateau',
        factor: float = 0.5,
        patience: int = 2,
        min_lr: float = 1e-6,
        min_delta: float = 1e-4,
        cooldown: int = 0,
        restart_patience: int = 6,
        restart_factor: float = 0.5,
        max_lr: Optional[float] = None,
        cycle_step_size: int = 5,
        cyclic_mode: str = 'triangular',
        gamma: float = 0.999,
        monitor: str = 'val_loss',
        mode: str = 'min',
        initial_lr: Optional[float] = None
    ):
        self.enabled = enabled
        self.policy = policy.lower() if policy else 'plateau'
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.min_delta = float(min_delta)
        self.cooldown = int(cooldown)
        self.restart_patience = int(restart_patience)
        self.restart_factor = float(restart_factor)
        self.max_lr = float(max_lr) if max_lr is not None else None
        self.cycle_step_size = int(cycle_step_size)
        self.cyclic_mode = cyclic_mode.lower() if cyclic_mode else 'triangular'
        self.gamma = float(gamma)
        self.monitor = monitor
        self.mode = mode.lower()
        self.initial_lr = initial_lr

        self.best_metric: float = float('inf') if self.mode == 'min' else float('-inf')
        self.wait_count: int = 0
        self.floor_wait_count: int = 0
        self.cooldown_counter: int = 0
        self.decay_count: int = 0
        self.decay_events: list = []

    def reset(self, initial_lr: Optional[float] = None):
        """Resets tracking state for a new training run."""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.wait_count = 0
        self.floor_wait_count = 0
        self.cooldown_counter = 0
        self.decay_count = 0
        self.decay_events = []
        if initial_lr is not None:
            self.initial_lr = initial_lr

    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < (self.best_metric - self.min_delta)
        else:
            return current > (self.best_metric + self.min_delta)

    def step(
        self,
        current_metric: float,
        current_lr: float,
        model_name: str = "Model",
        step_idx: int = 0,
        step_type: str = "Epoch"
    ) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
        """
        Evaluates current performance metric and computes updated learning rate.

        Returns:
            Tuple of (new_lr, lr_changed, event_details_dict)
        """
        if not self.enabled:
            return current_lr, False, None

        if self.initial_lr is None:
            self.initial_lr = current_lr

        # Handle Cyclic Learning Rate (CLR)
        if self.policy in ('cyclic', 'clr', 'triangular_cyclic'):
            max_lr = self.max_lr if self.max_lr is not None else (self.initial_lr or current_lr)
            base_lr = self.min_lr
            step_size = max(1, self.cycle_step_size)

            cycle = math.floor(1 + step_idx / (2 * step_size))
            x = abs(step_idx / step_size - 2 * cycle + 1)
            scale_factor = max(0.0, 1.0 - x)

            if self.cyclic_mode == 'triangular2':
                scale_factor = scale_factor / (2.0 ** (cycle - 1))
            elif self.cyclic_mode == 'exp_range':
                scale_factor = scale_factor * (self.gamma ** step_idx)

            new_lr = base_lr + (max_lr - base_lr) * scale_factor

            if abs(current_lr - new_lr) > 1e-12:
                event_dict = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_name': model_name,
                    'step_type': step_type,
                    'step': step_idx,
                    'monitor_metric': self.monitor,
                    'metric_value': round(float(current_metric), 5),
                    'old_lr': float(current_lr),
                    'new_lr': float(new_lr),
                    'policy': 'cyclic',
                    'reason': f"Cyclic LR step ({self.cyclic_mode}): {new_lr:.6e}"
                }
                return new_lr, True, event_dict
            return current_lr, False, None

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait_count = 0
            logger.debug(
                f"[{model_name}] {step_type} {step_idx}: In LR decay cooldown ({self.cooldown_counter} steps left). "
                f"Current LR: {current_lr:.6f}"
            )
            return current_lr, False, None

        if self._is_improvement(current_metric):
            old_best = self.best_metric
            self.best_metric = current_metric
            self.wait_count = 0
            self.floor_wait_count = 0
            logger.debug(
                f"[{model_name}] {step_type} {step_idx}: Performance improved ({self.monitor}: {old_best:.5f} -> {current_metric:.5f}). "
                f"Reset wait count."
            )
            return current_lr, False, None

        # Performance stagnated
        self.wait_count += 1
        logger.info(
            f"[{model_name}] {step_type} {step_idx}: Performance metric '{self.monitor}' stagnated at {current_metric:.5f} "
            f"(Best: {self.best_metric:.5f}). Wait count: {self.wait_count}/{self.patience}."
        )

        if self.wait_count >= self.patience:
            # Trigger LR Decay
            self.decay_count += 1
            if self.policy == 'exponential_plateau':
                new_lr = self.initial_lr * (self.factor ** self.decay_count)
            elif self.policy == 'step_plateau':
                step_val = self.factor * self.initial_lr
                new_lr = current_lr - step_val
            else:  # 'plateau'
                new_lr = current_lr * self.factor

            new_lr = max(new_lr, self.min_lr)

            if current_lr - new_lr > 1e-12:
                self.cooldown_counter = self.cooldown
                self.wait_count = 0
                self.floor_wait_count = 0

                reason = (
                    f"Performance metric '{self.monitor}' failed to improve by >={self.min_delta} "
                    f"for {self.patience} consecutive {step_type.lower()}s. Policy: '{self.policy}'."
                )

                event_dict = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_name': model_name,
                    'step_type': step_type,
                    'step': step_idx,
                    'monitor_metric': self.monitor,
                    'metric_value': round(float(current_metric), 5),
                    'best_metric_value': round(float(self.best_metric), 5),
                    'old_lr': float(current_lr),
                    'new_lr': float(new_lr),
                    'policy': self.policy,
                    'decay_count': self.decay_count,
                    'reason': reason
                }

                self.decay_events.append(event_dict)

                logger.info(
                    f"\n🔻 [LR DECAY TRIGGERED] [{model_name}] {step_type} {step_idx}: "
                    f"Learning rate reduced from {current_lr:.6e} to {new_lr:.6e}! "
                    f"Reason: {reason}\n"
                )

                return new_lr, True, event_dict
            else:
                # Learning rate reached floor limit (min_lr)
                self.floor_wait_count += 1
                logger.info(
                    f"[{model_name}] {step_type} {step_idx}: Learning rate reached floor limit (min_lr = {self.min_lr:.6e}). "
                    f"Stagnation count at floor: {self.floor_wait_count}/{self.restart_patience}."
                )

                if self.restart_patience > 0 and self.floor_wait_count >= self.restart_patience:
                    # TRIGGER WARM RESTART BUMP
                    bump_lr = max(self.initial_lr * self.restart_factor, self.min_lr * 2.0)
                    self.decay_count = 0
                    self.wait_count = 0
                    self.floor_wait_count = 0
                    self.cooldown_counter = max(self.cooldown * 2, 2)

                    reason = (
                        f"Performance metric '{self.monitor}' flattened at LR floor limit ({self.min_lr:.6e}) "
                        f"for {self.restart_patience} consecutive {step_type.lower()}s. "
                        f"Triggered Warm Restart Bump to shake optimizer state."
                    )

                    event_dict = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'model_name': model_name,
                        'step_type': step_type,
                        'step': step_idx,
                        'monitor_metric': self.monitor,
                        'metric_value': round(float(current_metric), 5),
                        'best_metric_value': round(float(self.best_metric), 5),
                        'old_lr': float(current_lr),
                        'new_lr': float(bump_lr),
                        'policy': 'warm_restart_bump',
                        'decay_count': 0,
                        'reason': reason
                    }

                    self.decay_events.append(event_dict)

                    logger.info(
                        f"\n⚡ [LR RESTART BUMP TRIGGERED] [{model_name}] {step_type} {step_idx}: "
                        f"Learning rate boosted from {current_lr:.6e} up to {bump_lr:.6e}! "
                        f"Reason: {reason}\n"
                    )

                    return bump_lr, True, event_dict

        return current_lr, False, None


class PerformanceLRDecayCallback(tf.keras.callbacks.Callback):
    """
    Keras Callback wrapping PerformanceLRDecayPolicy for deep learning models.
    Automatically updates model optimizer learning rate at the end of each epoch.
    """
    def __init__(
        self,
        policy: PerformanceLRDecayPolicy,
        model_name: str = "DLModel",
        epoch_offset: int = 0
    ):
        super().__init__()
        self.policy = policy
        self.model_name = model_name
        self.epoch_offset = epoch_offset

    def _get_lr(self) -> float:
        optimizer = self.model.optimizer
        if hasattr(optimizer.learning_rate, 'numpy'):
            return float(optimizer.learning_rate.numpy())
        elif hasattr(optimizer, 'lr'):
            return float(tf.keras.backend.get_value(optimizer.lr))
        else:
            return float(optimizer.learning_rate)

    def _set_lr(self, new_lr: float):
        optimizer = self.model.optimizer
        if hasattr(optimizer.learning_rate, 'assign'):
            optimizer.learning_rate.assign(new_lr)
        else:
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = self._get_lr()
        metric_val = logs.get(self.policy.monitor)
        if metric_val is None:
            # Fallback to loss or val_loss if monitor metric not found in logs
            metric_val = logs.get('val_loss', logs.get('loss', 0.0))

        actual_epoch = epoch + 1 + self.epoch_offset
        new_lr, changed, event = self.policy.step(
            current_metric=float(metric_val),
            current_lr=current_lr,
            model_name=self.model_name,
            step_idx=actual_epoch,
            step_type="Epoch"
        )

        if changed:
            self._set_lr(new_lr)
