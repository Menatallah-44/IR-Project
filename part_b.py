import json
import math
import os

# make sure we run from the project folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

N = 10
with open("outputs/index.json") as f:
    index = json.load(f)

names = []
for i in range(1, N + 1):
    names.append("d" + str(i))
words = sorted(index)

# keep everything we print so we can save it in results.txt
lines = []


def show(text=""):
    print(text)
    lines.append(str(text))


def weighted_tf(tf):
    if tf > 0:
        return 1 + math.log10(tf)
    return 0


def print_table(title, table):
    show()
    show(title)
    for w in words:
        line = w.ljust(13)
        for d in names:
            line = line + str(round(table[w][d], 4)).ljust(8)
        show(line)


# raw tf, weighted tf, df, idf, tf-idf
raw = {}
weighted = {}
tfidf = {}
df = {}
idf = {}
for w in words:
    df[w] = len(index[w])
    idf[w] = math.log10(N / df[w])
    raw[w] = {}
    weighted[w] = {}
    tfidf[w] = {}
    for d in names:
        tf = len(index[w].get(d, []))
        raw[w][d] = tf
        weighted[w][d] = weighted_tf(tf)
        tfidf[w][d] = weighted[w][d] * idf[w]

# length of each document
length = {}
for d in names:
    total = 0
    for w in words:
        total = total + tfidf[w][d] ** 2
    length[d] = math.sqrt(total)

# normalized tf-idf
norm = {}
for w in words:
    norm[w] = {}
    for d in names:
        norm[w][d] = tfidf[w][d] / length[d]

print_table("RAW TF", raw)
print_table("WEIGHTED TF", weighted)
print_table("TF-IDF", tfidf)
print_table("NORMALIZED TF-IDF", norm)

show()
show("DF and IDF")
for w in words:
    show(w.ljust(13) + " " + str(df[w]) + " " + str(round(idf[w], 4)))

show()
show("DOCUMENT LENGTHS")
for d in names:
    show(d + " " + str(round(length[d], 4)))


# phrase search: word2 must come right after word1
# returns a dictionary: document -> list of starting positions
def phrase(w1, w2):
    found = {}
    for d in names:
        if d in index[w1] and d in index[w2]:
            starts = []
            for p in index[w1][d]:
                if p + 1 in index[w2][d]:
                    starts.append(p)
            if len(starts) > 0:
                found[d] = starts
    return found


def show_phrase(title, found):
    show()
    show("PHRASE QUERY: " + title)
    show("Matched document | Starting positions | Token ranges")
    for d in found:
        starts = ""
        ranges = ""
        for p in found[d]:
            starts = starts + str(p) + ", "
            ranges = ranges + str(p) + "-" + str(p + 1) + ", "
        show(d.ljust(17) + " | " + starts[:-2].ljust(18) + " | " + ranges[:-2])


# rank documents for a query (each query word appears once)
def rank(title, query_words, docs_list):
    q = {}
    for w in query_words:
        q[w] = idf[w]
    q_len = 0
    for w in q:
        q_len = q_len + q[w] ** 2
    q_len = math.sqrt(q_len)
    for w in q:
        q[w] = q[w] / q_len

    scores = {}
    for d in docs_list:
        s = 0
        for w in q:
            s = s + q[w] * norm[w][d]
        scores[d] = s
    ranking = sorted(scores, key=scores.get, reverse=True)

    show()
    show(title)
    show("query length: " + str(round(q_len, 6)))
    for w in q:
        show("weight " + w + " " + str(round(q[w], 4)))
    for d in names:
        if d in scores:
            show(d + " " + str(round(scores[d], 4)))
    show("rank: " + " > ".join(ranking))


ml = phrase("machine", "learning")
ds = phrase("data", "science")

show_phrase('"machine learning"', ml)
rank('1. "machine learning"', ["machine", "learning"], list(ml))

show_phrase('"data science"', ds)
rank('2. "data science"', ["data", "science"], list(ds))

# boolean queries
both = []
only_first = []
for d in ml:
    if d in ds:
        both.append(d)
    else:
        only_first.append(d)

show()
show('BOOLEAN AND: "machine learning" AND "data science"')
show("Matched documents: " + ", ".join(both))
rank('3. "machine learning" AND "data science"',
     ["machine", "learning", "data", "science"], both)

show()
show('BOOLEAN AND NOT: "machine learning" AND NOT "data science"')
show("Matched documents: " + ", ".join(only_first))
rank('4. "machine learning" AND NOT "data science"',
     ["machine", "learning"], only_first)

# save everything to a file
f = open("outputs/results.txt", "w")
f.write("\n".join(lines))
f.close()