import argparse
import numpy as np
import pandas as pd
from env import CloudEnv
from agent import QAgent
from config import ALPHA, GAMMA, EPSILON


def train(env, agent, episodes, max_steps, epsilon_decay):
    """Train Q-agent on the cloud scheduling task."""
    for ep in range(1, episodes + 1):
        state = env.get_state()
        for step in range(max_steps):
            action = agent.choose_action(state)
            metrics = env.step(action)
            next_state = env.get_state()
            reward = env.compute_reward(state, action, metrics)
            agent.learn(state, action, reward, next_state)
            state = next_state

        # optional: decay exploration
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)

        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} complete, ε={agent.epsilon:.3f}")

    # save learned Q-table
    agent.save("qtable.pkl")


def round_robin(env, num_jobs):
    """Baseline: cycle through providers."""
    results = []
    for i in range(num_jobs):
        action = i % len(env.aws.client.meta.service_model.operation_names)  # or simply 3
        metrics = env.step(action)
        results.append(metrics)
    return results

def least_connection(env, num_jobs):
    """
    Baseline: pick the provider with the fewest ongoing connections.
    Here we proxy 'load' by the most recent latency metric from get_state().
    When you plug in a real connection count, swap loads = [...] accordingly.
    """
    results = []
    for _ in range(num_jobs):
        state = env.get_state()
        # state = [aws_energy, aws_latency, gcp_energy, gcp_latency, az_energy, az_latency]
        loads = [
            state[1],  # AWS latency
            state[3],  # GCP latency
            state[5],  # Azure latency
        ]
        action = int(np.argmin(loads))
        metrics = env.step(action)
        results.append(metrics)
    return results



def evaluate(env, agent, num_jobs):
    """Run the trained agent and baselines on a fixed workload."""
    print("\n=== Evaluating Schedulers ===")
    schedulers = {
        "Q-Learner": lambda: [env.step(agent.choose_action(env.get_state())) for _ in range(num_jobs)],
        "Round-Robin": lambda: round_robin(env, num_jobs),
        "Least-Conn":  lambda: least_connection(env, num_jobs),
    }

    summary = []
    for name, fn in schedulers.items():
        metrics = fn()
        # aggregate
        avg_latency = np.mean([m["latency"] for m in metrics])
        avg_energy  = np.mean([m["energy"]  for m in metrics])
        avg_cost    = np.mean([m["cost"]    for m in metrics])
        summary.append({
            "Scheduler": name,
            "Latency":   avg_latency,
            "Energy":    avg_energy,
            "Cost":      avg_cost
        })

    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    df.to_csv("evaluation_summary.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,   default=100,   help="Training episodes")
    p.add_argument("--max_steps",   type=int,   default=50,    help="Steps per episode")
    p.add_argument("--decay",       type=float, default=0.995, help="ε decay per episode")
    p.add_argument("--test_jobs",   type=int,   default=30,    help="Jobs for evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # init env & agent
    env = CloudEnv(
        aws_func_name=os.getenv("AWS_LAMBDA_NAME"),
        gcp_project=os.getenv("GCP_PROJECT"),
        gcp_func_name=os.getenv("GCP_FUNC_NAME"),
        azure_subscription_id=os.getenv("AZURE_SUB_ID"),
        azure_resource_group=os.getenv("AZURE_RG"),
        azure_func_app=os.getenv("AZURE_FUNC_APP")
    )
    agent = QAgent(actions=[0, 1, 2])

    # 1) Train
    train(env, agent, args.episodes, args.max_steps, args.decay)

    # 2) Evaluate
    evaluate(env, agent, args.test_jobs)
