# Reference: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L32
seed: 42

n_timesteps: !!float 1e6
policy: 'MlpPolicy'
n_steps: 16
batch_size: 4096
gae_lambda: 0.95
gamma: 0.99
n_epochs: 20
ent_coef: 0.01
learning_rate: !!float 3e-4
clip_range: !!float 0.2
policy_kwargs: "dict(
                  activation_fn=nn.ELU,
                  net_arch=[32, 32],
                  squash_output=False,
                )"
vf_coef: 1.0
max_grad_norm: 1.0
device: "cuda:0"



# Tuned
Pendulum-v1:
  n_envs: 4
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  n_steps: 1024
  gae_lambda: 0.95
  gamma: 0.9
  n_epochs: 10
  ent_coef: 0.0
  learning_rate: !!float 1e-3
  clip_range: 0.2
  use_sde: True
  sde_sample_freq: 4

# Tuned
CartPole-v1:
  n_envs: 8
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  n_steps: 32
  batch_size: 256
  gae_lambda: 0.8
  gamma: 0.98
  n_epochs: 20
  ent_coef: 0.0
  learning_rate: lin_0.001
  clip_range: lin_0.2


# Tuned
Pendulum-v1:
  n_envs: 4
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  n_steps: 1024
  gae_lambda: 0.95
  gamma: 0.9
  n_epochs: 10
  ent_coef: 0.0
  learning_rate: !!float 1e-3
  clip_range: 0.2
  use_sde: True
  sde_sample_freq: 4

ActionsCfg scale 50 => 
joint_efforts: tensor([[-2606.2766]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>
[Env 0]: Pole joint:  -100.0
joint_efforts: tensor([[3451.4514]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>
[Env 0]: Pole joint:  100.0001449584961
joint_efforts: tensor([[-2606.6353]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>
[Env 0]: Pole joint:  -100.0

ActionsCfg scale 5 => 
joint_efforts: tensor([[-3005.1240]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>
[Env 0]: Pole joint:  -100.00004577636719
joint_efforts: tensor([[3005.1707]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>
[Env 0]: Pole joint:  100.00000762939453
joint_efforts: tensor([[-3005.1245]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>
[Env 0]: Pole joint:  -99.99996948242188
joint_efforts: tensor([[3005.1660]], device='cuda:0') torch.Size([1, 1]) <class 'torch.Tensor'>

ActionsCfg scale 1 => 
Good

scale=10.0 # RL => too large
scale=8.0 # RL => too large
scale=6.0 # RL => too large

# Rotate only one direction

from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

```python
@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)
```

```python
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
```

TODO: set_joint_effort_target => my_create_interactive_scene.py

open_ai_gym
self.max_action = 50.0 / act_scale = -0.005 => slight movement
self.max_action = 50.0 / act_scale = -0.0 => 

soft_binary_with_repellor

max_action 25.0 / after clamp 
actions=tensor([[ 0.2389],
        [-0.5853],
        [-1.0913],
        [-0.6305],
        [-0.5190],
        [ 0.3964],
        [-0.5412],
        [-1.5243],
        [-0.8362],
        [-1.4127],
        [-1.2904],
        [-0.3930],
        [-1.5421],
        [-0.4943],
        [-2.2728],
        [-0.8161]],

max_action 25.0 / no clamp 
actions=tensor([[-1.0580],
        [-0.1515],
        [-2.5391],
        [-0.9601],
        [-2.3541],
        [ 0.7628],
        [ 0.0580],
        [-1.3803],
        [-0.9785],
        [-0.6852],
        [ 0.4753],
        [ 0.2439],
        [-1.9257],
        [ 1.5984],
        [-0.4336],
        [-0.5499]],

_pre_physics_step digging => 