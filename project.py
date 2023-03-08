import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from mpi4py import MPI
import pdfplumber as pdf

download = False


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if not download:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        download = True


    with pdf.open(r'book2.pdf') as file:
        n = len(file.pages)
        print(f"Process {rank} will begin extracting text from a total {n} pdf pages.")

        data = [{'text': file.pages[i].extract_text(), 'number': i+1} for i in range(n)]

        # split data according to rank size
        data1 = data[0:int(n/3)]
        data2 = data[int(n/3):int(2*n/3)]
        data3 = data[int(2*n/3):n]

        message = [data1, data2, data3]
        requests = []

        for r in range(1, size):
            print(f"Process {rank} sending data to process {r-1}")
            requests.append(comm.isend(message[r-1], dest=r))

    MPI.Request.waitall(requests)

    print(f"Process: {rank} done sending text to 3 other processes\n")

    result = []
    result = result + comm.recv(source=1)
    result = result + comm.recv(source=2)
    result = result + comm.recv(source=3)

    print(f"Process {rank} received results from all processes")
    print("Begin combining and displaying data\n\n")

    result.sort(key=lambda x: x['score'], reverse=True)

    print("Top 5 ranked sentences are:\n")
    for sen in result[0:5]:
        print(f"From page {sen['page']} with a score of {sen['score']}:")
        print(sen['text'], end='\n\n')



   #for sentence in result:
        #print(sentence['score'])


for r in range(1, size):
    if rank == r:
        data = comm.recv(source=0)
        print(f"I'm rank {r} and I received {len(data)} pages beginning with page {data[0]['number']}")
        print("Begin NLP processing")

        def rank_sentences(text):
            # tokenize sentences
            sentences = sent_tokenize(text)
            num_sentences = len(sentences)

            # create lemmatizer and stop words
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            # create a dictionary to hold sentence scores
            sentence_scores = {}

            # iterate through each sentence
            for i, sentence in enumerate(sentences):
                # split sentence into words
                words = nltk.word_tokenize(sentence)

                # create a list to hold part of speech tags
                pos_tags = nltk.pos_tag(words)

                # create a dictionary to hold word scores
                word_scores = {}

                # iterate through each word and its part of speech tag
                for word, pos in pos_tags:
                # only consider nouns, verbs, and adjectives
                    if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']:
                        # lemmatize word
                        lemma = lemmatizer.lemmatize(word)
                        # ignore stop words
                        if lemma not in stop_words:
                        # increase word score if it is a named entity
                            if pos in ['NNP', 'NNPS']:
                                word_scores[lemma] = 2
                            else:
                                word_scores[lemma] = 1

                    # sum word scores to get sentence score
                    sentence_scores[i] = sum(word_scores.values())

            # sort sentences by score
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

            # return list of sentence scores and indices
            return [(sentences[index], score) for index, score in sorted_sentences]

        result = []
        for page in data:
             sentences = rank_sentences(page['text'])
             for sentence in sentences:
                sen = {}
                sen['text'] = sentence[0]
                sen['score'] = sentence[1]
                sen['page'] = page['number']
                result.append(sen)

        print(f"Process {rank} completed NLP ranking, sending result process 0")
        comm.send(result, dest=0)

MPI.Finalize()
