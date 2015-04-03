import random
import gzip

train_path = 'data/labeledTrainData.tsv.gz'


validation = random.sample(range(25000), 2500)

with gzip.open(train_path) as f, \
        gzip.open('data/train.tsv.gz', 'w') as train, \
        gzip.open('data/validation.tsv.gz', 'w') as val:

    header = f.next()
    train.write(header)
    val.write(header)
    
    for i, line in enumerate(f):
        if i in validation:
            val.write(line)
        else:
            train.write(line)