from flask import render_template
from app import app
from app.forms import SearchForm
from app.modelling import SentenceMappingsProducer
from transformers import BertConfig
import torch
from torch import nn
from transformers import BertTokenizer
import json

import gdown
import os
from app.article import Article

# https://stackoverflow.com/questions/59122308/heroku-slug-size-too-large-after-installing-pytorch
# https://download.pytorch.org/whl/torch_stable.html


@app.route("/", methods=["GET", "POST"])
def index():
    form = SearchForm()
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
    # Download model
    url = "https://drive.google.com/uc?id=11OHi9wETRPAHUTIH4p6BqZY3gH6NJtve"
    model_path = "app/static/models/cord_smallbert_grounding.pt"
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=True)
    if form.query.data is not None:
        config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
        model = nn.DataParallel(
            SentenceMappingsProducer("google/bert_uncased_L-4_H-512_A-8", config, 3)
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.train(False)
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(form.query.data)).unsqueeze(0)
            attn_mask = torch.ones_like(input_ids)
            preds = model(input_ids, attn_mask).numpy()
        articles = [
            Article(article_json, preds)
            for article_json in json.load(open("app/static/articles_coords_10_4.json"))
        ]
        articles_sorted = sorted(articles, key=lambda x: x.distance_to_query)

        return render_template("index.html", documents=articles_sorted[:200], form=form)
    return render_template("index.html", form=form)
