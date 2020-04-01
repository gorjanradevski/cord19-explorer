from flask import render_template
from app import app
from app.forms import SearchForm
from app.modelling import SentenceMappingsProducer
from transformers import BertConfig
import torch
from torch import nn
from transformers import BertTokenizer
import json
from app.article import Article

# https://stackoverflow.com/questions/59122308/heroku-slug-size-too-large-after-installing-pytorch


@app.route("/", methods=["GET", "POST"])
def index():
    form = SearchForm()
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    if form.query.data is not None:
        config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
        model = nn.DataParallel(
            SentenceMappingsProducer("google/bert_uncased_L-4_H-256_A-4", 512, config)
        )
        model.load_state_dict(
            torch.load(
                "app/static/models/sentence_mapping_reg_L4H256A4",
                map_location=torch.device("cpu"),
            )
        )
        model.train(False)
        with torch.no_grad():
            preds = model(
                torch.tensor(tokenizer.encode(form.query.data)).unsqueeze(0)
            ).numpy()
        articles = [
            Article(article_json, preds)
            for article_json in json.load(open("app/static/articles_coords.json"))
        ]
        articles_sorted = sorted(articles, key=lambda x: x.distance_to_query)

        return render_template("index.html", documents=articles_sorted[:20], form=form)
    return render_template("index.html", form=form)
