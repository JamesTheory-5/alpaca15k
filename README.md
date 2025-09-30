# alpaca15k
```python
from datasets import load_dataset
import json

# Load datasets
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

# Convert to lists of dicts
alpaca_data = [dict(row) for row in alpaca]
dolly_data = [dict(row) for row in dolly]



temp = []
for i in alpaca_data:
    x = {}
    x["instruction"] = i["instruction"]
    if("input" in i):
        if(i["input"] != ''):
            x["input"] = i["input"]
    x["output"] = i["output"]
    temp.append(x)

for i in dolly_data:
    x = {}
    x["instruction"] = i["instruction"]
    if("context" in i):
        if(i["context"] != ''):
            x["input"] = i["context"]
    x["output"] = i["response"]
    temp.append(x)    

with open("alpaca15k.json","w") as jf:
    json.dump(temp,jf,indent=4)

print("Saved alpaca_dolly.json")

```

### self-instruct
```python
from datasets import load_dataset
import json

dataset_name = "HuggingFaceH4/self_instruct"
# Load datasets
dataset = load_dataset(dataset_name, split="train")

# Convert to lists of dicts
dataset_data = [dict(row) for row in dataset]


temp = []
for i in dataset_data:
    x = {}
    x["instruction"] = i["prompt"]
    x["output"] = i["completion"]
    temp.append(x)

with open("self-instruction.json","w") as jf:
    json.dump(temp,jf,indent=4)

print("Saved alpaca_dolly.json")

```
