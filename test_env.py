# test_env.py  (Create this new file in the project root)
from tarware.warehouse import Warehouse, RewardType

env = Warehouse(
    shelf_columns=3,
    column_height=8,
    shelf_rows=1,
    num_agvs=2,
    num_pickers=2,
    request_queue_size=20,
    max_inactivity_steps=None,
    max_steps=500,
    reward_type=RewardType.INDIVIDUAL,
    observation_type="global"
)

print("Environment created successfully!")
obs = env.reset()
print("Environment reset successfully!")