from flask import Flask, render_template, request, redirect
from nltk.corpus import reuters
import nltk
import os
import pickle
import numpy as np

HEADLINES_PATH = "data/headlines.pkl"
REPRESENTATIONS_PATH = "data/model-20171214-121501-representations.pkl"

headlines = {}
representations = None
fileids = None
app = Flask(__name__)

@app.route('/')
def show_index():
    return render_template("index.html", reuters=reuters)

@app.route('/choose', methods=['POST'])
def redirect_after_category_choose():
    category = request.form.get("category")
    return redirect("/category/{}".format(category))

@app.route('/category/<category>', methods=['GET'])
def show_category(category):
    category_headlines = {fileid.replace('/', '_'): headlines[fileid] for fileid in reuters.fileids(category)}
    return render_template("choose.html", headlines=category_headlines)


@app.route('/show', methods=['POST'])
def redirect_after_text_choose():
    fileid = request.form.get("fileid")
    return redirect("/text/{}".format(fileid))


@app.route('/text/<fileid>', methods=['GET'])
def show_text(fileid):
    real_fileid = fileid.replace('_', '/')
    text = reuters.raw(real_fileid)
    similar_fileids = get_similar_texts(real_fileid)
    similar_texts = [(f.replace('/', '_'), headlines[f]) for f in similar_fileids]
    return render_template("show.html", text=text, similar_texts=similar_texts)


def get_similar_texts(fileid, count=3):
    """Get similar texts to `fileid` using norm 2 in the document representation space"""
    i = np.where(fileids == fileid)[0][0]
    dists = distance(representations, representations[i:i+1])
    similar_i = np.argpartition(dists[0], count+1)[0:count+1]
    # remove the given text from the similar recommendations
    similar_fileids = [fileids[j] for j in similar_i if i != j]
    return similar_fileids


def main():
    global representations, fileids

    data_dir = os.path.join(os.getcwd(), "data")
    nltk.data.path.insert(0, data_dir)

    # preload the reuters data at the server launch
    reuters._init()

    try:
        headlines.update(pickle.load(open(HEADLINES_PATH, "rb")))
    except (OSError, IOError):
        for fileid in reuters.fileids():
            headlines[fileid] = reuters.raw(fileid).split('\n', maxsplit=1)[0]
        pickle.dump(headlines, open(HEADLINES_PATH, "wb"))

    # Load the documents informations (continuous vector repr, etc)
    (representations, fileids) = pickle.load(open(REPRESENTATIONS_PATH, "rb"))

    print(get_similar_texts("test/19892"))

    # Start the app
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0', port=port)


def distance(a1, a2, norm_ord=2):
    """Distance, using norm `ord`(2, 3, inf, ...), between all
    N points of `a1` and M points of `a2` resulting in a N*M distance matrix"""
    dim = a1.shape[-1]
    assert dim == a2.shape[-1] # same dim of points of a1 and a2
    diff = a1.reshape(1, -1, dim) - a2.reshape(-1, 1, dim)
    distances = np.linalg.norm(diff, ord=norm_ord, axis=2)
    return distances

if __name__ == "__main__":
    main()
