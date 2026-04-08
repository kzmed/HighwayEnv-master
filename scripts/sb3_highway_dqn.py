import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env


TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-v0", render_mode="rgb_array")
    obs, info = env.reset()

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="../data/highway_dqn/",
    )


    if TRAIN:
        model.learn(total_timesteps=int(1e4))
        model.save("../data/highway_dqn/model")
        del model


    model = DQN.load("../data/highway_dqn/model", env=env)
    env = RecordVideo(
        env, video_folder="../data/highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
