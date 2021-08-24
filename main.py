# Import and load the spacy model
import spacy
from pathlib import Path
from flask import Flask, request, jsonify
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import random


app = Flask(__name__)
# Output directory
output_dir = Path('./model/')

nlp = spacy.load("en_core_web_sm")

# Getting the ner component
ner = nlp.get_pipe('ner')


def train_custom_entity_type(label, train_data, no_of_iterations, drop_percent):
    # Add the new label to ner
    ner.add_label(label)

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for n iterations
        for itn in range(no_of_iterations):
            # shuffle examples before training
            random.shuffle(train_data)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=sizes)
            # ictionary to store losses
            losses = {}
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, drop=drop_percent)
                #print("Losses", losses)

    # Saving the model to the output directory
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta['name'] = "custom_entity_type"  # rename model
    nlp.to_disk(output_dir)
    #print("Saved model to", output_dir)
    output_dict = {"NER model accuracy": losses, "Iterations": no_of_iterations,
                   "drop_out_percentage": drop_percent}
    return output_dict


@app.route("/NER_train_custom_entity", methods=['GET', 'POST'])
def train_SpaCyNER_custom_entity_type():
    if request.method == 'POST':
        train_data = request.form["train_data"]
        label = request.form["label"]
        no_of_iterations = int(request.form["no_of_iterations"])
        drop_percent = float(request.form["drop_percent"])
        if len(train_data):
            if type(train_data) == str:
                train_data = eval(train_data)
            return_obj = train_custom_entity_type(label, train_data, no_of_iterations, drop_percent)
        else:
            return_obj = "check training data"
    else:
        return_obj = "Request could not be processed"
    return return_obj


@app.route("/predict_NER_entity", methods=['GET', 'POST'])
def predict_custom_NER_entity():
    if request.method == 'POST':
        data_input = request.form["text"]
        # Loading the model from the directory
        nlp2 = spacy.load(output_dir)
        nlp2.get_pipe("ner")
        doc2 = nlp2(data_input)
        entity_dict = {}
        for ent in doc2.ents:
            entity_dict[ent.label_] = ent.text
            #print(ent.label_, ent.text)
    else:
        entity_dict = "Request could not be processed"
    return entity_dict


if __name__ == "__main__":
    app.run(debug=True)


