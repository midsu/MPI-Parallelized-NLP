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
    # Download NLTK resources if not done already
    if not download:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        download = True

    # Read PDF and divide workload
    with pdf.open(r'book2.pdf') as file:
        n = len(file.pages)
        print(f"Process {rank} will begin extracting text from {n} pdf pages.")

        data = [{'text': file.pages[i].extract_text(), 'number': i + 1} for i in range(n)]
        message = [data[i::size - 1] for i in range(size - 1)]

        # Send divided data to other processes
        requests = [comm.isend(msg, dest=i + 1) for i, msg in enumerate(message)]
    MPI.Request.waitall(requests)

    print(f"Process {rank} completed sending text to other processes.")

    # Gather results and display top sentences
    result = []
    for r in range(1, size):
        result += comm.recv(source=r)

    print(f"Process {rank} received results from all processes and is now combining and displaying data.")

    result.sort(key=lambda x: x['score'], reverse=True)

    print("Top 5 ranked sentences are:\n")
    for sen in result[:5]:
        print(f"From page {sen['page']} with a score of {sen['score']}:\n{sen['text']}\n")

else:
    # Receive data and perform NLP processing
    data = comm.recv(source=0)
    print(f"Process {rank} received {len(data)} pages beginning with page {data[0]['number']}. Beginning NLP processing.")

    def rank_sentences(text):
        # NLP processing code (same as before)

    result = []
    for page in data:
        sentences = rank_sentences(page['text'])
        for sentence in sentences:
            sen = {'text': sentence[0], 'score': sentence[1], 'page': page['number']}
            result.append(sen)

    print(f"Process {rank} completed NLP ranking and is sending results to process 0.")
    comm.send(result, dest=0)

MPI.Finalize()
