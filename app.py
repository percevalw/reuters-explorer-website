from flask import Flask, render_template, request
from nltk.corpus import reuters
import nltk
import os
import pickle

HEADLINES_PATH = "data/headlines.pkl"

headlines = {}
app = Flask(__name__)

@app.route('/')
def show_category_form():
    return render_template("index.html", reuters=reuters)


@app.route('/choose', methods=['POST'])
def show_text_form():
    category = request.form.get("category")
    category_headlines = {fileid: headlines[fileid] for fileid in reuters.fileids(category)}
    return render_template("choose.html", headlines=category_headlines)


@app.route('/show', methods=['POST'])
def show_text():
    fileid = request.form.get("fileid")
    text = reuters.raw(fileid)
    return render_template("show.html", text=text)


def main():
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

    # Start the app
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
