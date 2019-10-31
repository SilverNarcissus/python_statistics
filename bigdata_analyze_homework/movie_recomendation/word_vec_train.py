from gensim.models import Word2Vec

with open('./data/text8.txt') as train:
    sentences = train.readline()

    model = Word2Vec(sentences, size=100)

    model.save('word2vec.model')