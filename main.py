from deepQ import deepQ_simple
import os

# --- Demo file for showing how you would use this --- #

# specify hyperparams
env_name = "MountainCar-v0"

state_shape = 2
n_actions = 3

n_episodes = 2000

lr = 0.00075

batch_size = 64 
buffer_size = 100000

epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997

gamma = 0.96

target_net_update_freq = 20
polyak_average = False
tau = 0.001

checkpoint_freq = 100

checkpoint_dir = "checkpoints"
log_dir = "runs"
base_name = f"DeepQ_{env_name}_e{epsilon_decay}_lr{lr}_batchsize{batch_size}_buffersize{buffer_size}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# this is the main algorithm object, which you can use to either train or test a model
q_alg = deepQ_simple(
    env_id=env_name,
    state_shape=state_shape,
    n_actions=n_actions,
    lr=lr,
    batch_size=batch_size,
    buffer_size=buffer_size,
    checkpoint_dir=checkpoint_dir,
    base_name=base_name,
    log_dir=log_dir,
)

TRAIN = True

if TRAIN:
    q_alg.train(
        n_episodes=n_episodes,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        target_net_update_freq=target_net_update_freq,
        polyak_average=polyak_average,
        tau=tau,
        checkpoint_freq=checkpoint_freq,
    )



# checkpoint_path = "checkpoints/DeepQ_MountainCar-v0_e0.997_lr0.00075_batchsize64_buffersize150000_checkpoint700"
# q_alg.load_checkpoint(checkpoint_path)
# q_alg.visual_demo()
q_alg.record_demo("demos", demo_name="cartpole_demo")
