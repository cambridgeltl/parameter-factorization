import os
import sys
import math
import random
random.seed(313)

basedir = sys.argv[1]

def printout(split, outdir, sentences):
    fout = open(outdir + '.' + split + '.bio', 'w')
    for sent in sentences:
       for l in sent:
           fout.write(l)
       fout.write('\n')
    fout.close()

for filename in os.listdir(basedir):
    if filename.endswith(".bio"): 
        print("Splitting", filename)
        sentences = []
        current = []
        fin = open(os.path.join(basedir, filename))
        for l in fin:
            if not l.strip():
                sentences.append(current)
                current = []
            else:
                current.append(l)
        if current:
            sentences.append(current)
        random.shuffle(sentences)
        if len(sentences) >= 10:
            print('Found {} sentences'.format(len(sentences)))
            delim1 = int(math.floor(len(sentences) * 0.8))
            delim2 = int(math.floor(len(sentences) * 0.9))
            print('Split sizes: train {}, dev {}, test {}.'.format(delim1, delim2-delim1, len(sentences)-delim2))
            os.mkdir(os.path.join(basedir, filename[8:-4]))
            outdir = os.path.join(basedir, filename[8:-4], filename[8:-4])
            printout('train', outdir, sentences[:delim1])
            printout('dev', outdir, sentences[delim1:delim2])
            printout('test', outdir, sentences[delim2:])
