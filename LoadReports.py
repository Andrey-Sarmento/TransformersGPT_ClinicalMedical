# Package
import glob
import os

# Load the reports
arquivos = glob.glob(os.path.join("Reports", "Pront[0-9]*.txt"))
arquivos.sort(key=lambda x: int(x.rsplit("Pront", 1)[1].split(".")[0]))
corpus = []

for i, f in enumerate(arquivos):
    with open(f, "r", encoding="utf-8") as fh:
        txt = fh.read()
        corpus.append("== Prontuário ==\n\n" + txt)

corpus = "\n\n".join(corpus)

# índices onde aparece o caractere µ
#indices = [i for i, ch in enumerate(corpus) if ch == "µ"]
#len(indices)
#for idx in indices:
#    print(corpus[max(0, idx-20): idx+20])
#    print("-"*60)

# Contagem de palavras diferentes
#from collections import Counter
#corpus_words = corpus.split()
#cont = Counter(corpus_words)
#len(cont)
