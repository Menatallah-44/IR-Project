import os
import sys
import json

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

BASE = os.path.dirname(os.path.abspath(__file__))

# 1. read the 10 text files
docs = []
for i in range(1, 11):
    name = "d" + str(i)
    with open(os.path.join(BASE, "docs", name + ".txt"), "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    docs.append((name, text))
    print(name, "|", text)

# 2. start spark
spark = SparkSession.builder.master("local[1]").appName("index").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")


# 3. give every word in a file its position (starting from 1)
def get_words(item):
    name = item[0]
    words = item[1].lower().split()
    result = []
    position = 1
    for w in words:
        result.append((w, (name, position)))
        position = position + 1
    return result


# 4. for one word, collect the files and positions
def make_postings(item):
    word = item[0]
    postings = {}
    for name, pos in item[1]:
        if name not in postings:
            postings[name] = []
        postings[name].append(pos)
    for name in postings:
        postings[name].sort()
    return (word, postings)


rdd = sc.parallelize(docs)
pairs = rdd.flatMap(get_words)
grouped = pairs.groupByKey()
index = dict(grouped.map(make_postings).collect())
spark.stop()

# 5. print and save the index
out_dir = os.path.join(BASE, "outputs")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as f:
    for word in sorted(index):
        parts = []
        for i in range(1, 11):
            name = "d" + str(i)
            if name in index[word]:
                positions = ", ".join(str(p) for p in index[word][name])
                parts.append(name + ": " + positions)
        line = word + " " + "; ".join(parts)
        print(line)
        f.write(line + "\n")

with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f2:
    json.dump(index, f2)

print("Unique terms:", len(index))