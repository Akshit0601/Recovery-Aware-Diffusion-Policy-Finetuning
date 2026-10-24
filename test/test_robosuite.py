"""
Sample test run: Panda robot on the Lift task (pick up a cube).
Runs a short episode with random actions and prints observations.
No display needed — runs fully headless.
"""

import numpy as np
import robosuite as suite

def main():
    # ---------- Create environment ----------
    env = suite.make(
        env_name="Lift",                 # pick-up-the-cube task
        robots="Panda",                  # Franka Panda arm
        has_renderer=True,              # no on-screen window
        has_offscreen_renderer=False,    # skip offscreen rendering too (faster)
        use_camera_obs=False,            # no image observations
        horizon=200,                     # max steps per episode
        reward_shaping=True,             # dense reward signal
    )

    env_name = getattr(env, 'name', env.__class__.__name__)
    robot_name = getattr(env.robots[0], 'name', env.robots[0].__class__.__name__)

    print("=" * 60)
    print(f"Environment : {env_name}")
    print(f"Robot       : {robot_name}")
    print(f"Action dim  : {env.action_dim}")
    print(f"Horizon     : {env.horizon}")
    print("=" * 60)

    # ---------- Reset and run an episode ----------
    obs = env.reset()

    # Show observation keys
    print(f"\nObservation keys ({len(obs)}):")
    for k, v in sorted(obs.items()):
        shape = v.shape if hasattr(v, "shape") else type(v).__name__
        print(f"  {k:40s} {shape}")

    print("\n--- Running episode with random actions ---\n")

    total_reward = 0.0
    success = False

    for step in range(env.horizon):
        action = np.random.uniform(-1, 1, size=env.action_dim)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if reward > 0 and step % 20 == 0:
            print(f"  step {step:4d}  reward={reward:.4f}  cumulative={total_reward:.4f}")

        if done:
            # Check if task was actually solved
            success = env._check_success()
            print(f"  Episode ended at step {step}")
            break

    print(f"\n{'=' * 60}")
    print(f"Total reward : {total_reward:.4f}")
    print(f"Task success : {success}")
    print(f"Steps taken  : {step + 1}")
    print(f"{'=' * 60}")

    # ---------- Quick MuJoCo model info ----------
    model = env.sim.model
    print(f"\nMuJoCo model stats:")
    print(f"  Bodies : {model.nbody}")
    print(f"  Joints : {model.njnt}")
    print(f"  DOFs   : {model.nv}")
    print(f"  Geoms  : {model.ngeom}")

    env.close()
    print("\n✅ Test run complete — robosuite + MuJoCo stack is working!")


if __name__ == "__main__":
    main()
