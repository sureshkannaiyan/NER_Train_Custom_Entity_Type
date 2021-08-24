Sample input for training custom entity type:

label= "FOOD"
train_data =[ ("Pizza is a common fast food.", {"entities": [(0, 5, "FOOD")]}), ("Pasta is an italian recipe", {"entities": [(0, 5, "FOOD")]})] 
no_of_iterations = 150 
drop_percent = 0.2 

Sample input to predict entity type:

text = "Dosa is an extremely famous south Indian dish"


NER Code Credits to : https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
