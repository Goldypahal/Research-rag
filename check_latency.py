import json
import os

path = r"g:\Desktop\RAG\research_rag\data\traces\query_logs.jsonl"
if os.path.exists(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[-5:]:
            try:
                data = json.loads(line)
                print(f"Query: {data.get('query')}")
                print(f"Latency: {data.get('latency'):.2f}s")
                print("-" * 20)
            except:
                pass
else:
    print("File not found")
