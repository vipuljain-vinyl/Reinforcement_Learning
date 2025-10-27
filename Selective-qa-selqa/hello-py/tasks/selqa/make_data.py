
# tasks/selqa/make_data.py
import json, random, re
random.seed(1337)

FACTS = [
  ("paris", "The capital of France is Paris.", ["Paris"]),
  ("canberra", "Australia's capital city is Canberra.", ["Canberra"]),
  ("nile", "The Nile is the longest river in Africa.", ["Nile"]),
  ("everest", "Mount Everest is Earth's highest mountain.", ["Mount Everest","Everest"]),
  ("mercury", "Mercury is the closest planet to the Sun.", ["Mercury"]),
  ("einstein", "Albert Einstein developed the theory of relativity.", ["Albert Einstein","Einstein"]),
  ("python", "Python is a programming language created by Guido van Rossum.", ["Python"]),
  ("pandas", "pandas is a Python library for data analysis.", ["pandas"]),
  ("h2o", "Water has the chemical formula H2O.", ["H2O","water"]),
  ("beethoven", "Beethoven composed nine symphonies.", ["Beethoven"]),
  ("amazon", "Amazon River flows in South America.", ["Amazon River","Amazon"]),
  ("sahara", "The Sahara is a large desert in Africa.", ["Sahara"]),
  ("venus", "Venus is sometimes called Earth's sister planet.", ["Venus"]),
  ("tokyo", "Tokyo is the capital of Japan.", ["Tokyo"]),
  ("apple", "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.", ["Apple","Apple Inc."]),
  ("tesla", "Tesla, Inc. was founded by a group of engineers including Martin Eberhard and Marc Tarpenning.", ["Tesla","Tesla, Inc."]),
]

NEG_TEMPL = [
  "Which city is the capital of {C}?","Name the capital of {C}.",
  "Who developed relativity?","What is the chemical formula of water?",
  "Which river is the longest in Africa?","Name Earth's highest mountain."
]

COUNTRIES = {"France":"Paris","Australia":"Canberra","Japan":"Tokyo"}

def normalize(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()

def build_corpus():
    docs = []
    # add some distractors
    distractors = [
      "The blue whale is the largest animal.", "Copper has symbol Cu.",
      "Mars is the fourth planet.", "The Danube flows in Europe.",
    ]
    did = 0
    for key, text, _ in FACTS:
        docs.append({"doc_id": f"d{did}", "text": text}); did += 1
    for t in distractors:
        docs.append({"doc_id": f"d{did}", "text": t}); did += 1
    return docs

def q_variants(key, text, golds):
    CANDS = []
    if "capital of" in text:
        # map gold -> country if we can
        for country, cap in COUNTRIES.items():
            if cap in golds:
                CANDS += [f"Which city is the capital of {country}?",
                          f"Name the capital of {country}.",
                          f"What is {country}'s capital?"]
                break
        else:
            CANDS += [f"Which city is the capital of France?",
                      f"Name the capital of France.",
                      f"What is France's capital?"]
    elif "relativity" in text:
        CANDS += ["Who developed the theory of relativity?",
                  "Name the scientist who created relativity."]
    elif "H2O" in text:
        CANDS += ["What is water's chemical formula?",
                  "Write the chemical formula for water."]
    elif "longest river" in text:
        CANDS += ["Which river is the longest in Africa?",
                  "Name Africa's longest river."]
    elif "highest mountain" in text:
        CANDS += ["What is Earth's highest mountain?",
                  "Name Earth's tallest mountain."]
    else:
        # generic backups
        CANDS += [f"What does this refer to: {key}?",
                  f"Give the key fact about {key}."]

    return list(dict.fromkeys(CANDS))  # dedupe, keep order

def make_qa(docs, variants_per_fact=2, seed=1337):  # fewer variants => fewer Qs
    random.seed(seed)
    items = []
    qid = 0
    for key, text, golds in FACTS:
        vs = q_variants(key, text, golds)
        random.shuffle(vs)
        # # create a question roughly tied to the fact
        # if "capital of" in text:
        #     C = [c for c in COUNTRIES if COUNTRIES[c] in golds][0] if any(COUNTRIES.get(c) in golds for c in COUNTRIES) else "France"
        #     q = random.choice(NEG_TEMPL[:2]).format(C=C)
        # elif "relativity" in text: q = "Who developed the theory of relativity?"
        # elif "H2O" in text: q = "What is water's chemical formula?"
        # elif "longest river" in text: q = "Which river is the longest in Africa?"
        # elif "highest mountain" in text: q = "What is Earth's highest mountain?"
        # else:
        #     q = f"What does this refer to: {key}?"
        for q in vs[:variants_per_fact]:
            items.append({"qid": f"q{qid}", "question": q, "answers": golds}); qid += 1
    random.shuffle(items)

    # split
    # Larger splits: ~40% calib, 60% test (ensure at least 10 test)
    calib_n = max(12, int(0.4 * len(items)))
    test_n  = max(10, len(items) - calib_n)
    calib = items[:calib_n]
    test  = items[calib_n:calib_n + test_n]
    return calib, test

def main():
    docs = build_corpus()
    calib, test = make_qa(docs)
    with open("corpus.jsonl","w") as f:
        for d in docs: f.write(json.dumps(d)+"\n")
    with open("train_calib.jsonl","w") as f:
        for x in calib: f.write(json.dumps(x)+"\n")
    # hide gold answers in test for realism
    with open("test.jsonl","w") as f:
        for x in test: f.write(json.dumps({"qid":x["qid"],"question":x["question"]})+"\n")

if __name__ == "__main__":
    main()
