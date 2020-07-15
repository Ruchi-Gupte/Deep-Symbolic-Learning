import json
from Stage0 import Stage0
from Stage1 import Stage1
from Stage2 import Stage2
from Stage3 import Stage3
from Stage4 import Stage4
from Stage5 import Stage5


stage3_load_condition=stage2_load_condition=1 #If you want to run code on a new dataset then set as 0
                                              #If load_condition=1 then it will load the model and symbolic list from file

with open('config.json', 'r') as f:
        config = json.load(f)

Stage0(config)  #Reading dataset Stage
Stage1(config) #Feature Extraction/ Patch Creation Stage
Stage2(config,stage2_load_condition) #Symbolic ID Assignment Stage
Stage3(config,stage3_load_condition) #Feature Scoring Stage
Stage4(config) #Hardcoding Directional Matrix Stage
Stage5(config) #Patch-Patch Co-Occurence Stage


#Community Detection Stage