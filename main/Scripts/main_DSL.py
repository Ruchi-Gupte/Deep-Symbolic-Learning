import json
from Stage0 import Stage0
from Stage1 import Stage1
from Stage2 import Stage2
from Stage3 import Stage3
from Stage4 import Stage4
from Stage5 import Stage5


stage3_load_condition=stage2_load_condition=1

with open('config.json', 'r') as f:
        config = json.load(f)

Stage0(config)
Stage1(config)
Stage2(config,stage2_load_condition)
Stage3(config,stage3_load_condition)
Stage4(config)
Stage5(config)
