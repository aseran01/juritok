import  sentencepiece
import csv
from transformers import XLNetTokenizer
import matplotlib.pyplot as plt
import re
from collections import Counter



# Chemin vers votre fichier CSV
chemin_rad = r'C:\Users\seran\juritok\files\jorf_'


model = './spiece.model'
model = './mymodel.model'

def model_pretrained():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    tokenizer.save_pretrained('./')

def model_training(dataset,vocab_size):
    sentencepiece.SentencePieceTrainer.train(input=dataset, 
                                            model_prefix='mymodel', 
                                            vocab_size=vocab_size, 
                                            character_coverage=1.0,
                                            model_type='unigram')

def csv_in_txt():
    for annee in ['2018','2019','2020','2021','2022','2023']:
        with open(chemin_rad+annee+'.csv', 'r', newline='', encoding='utf-8') as fichier_csv:
            lecteur_csv = csv.reader(fichier_csv, delimiter='|')
            
            with open(chemin_rad+annee+'.txt', 'w', encoding='utf-8') as fichier_txt:
                i=0
                for ligne in lecteur_csv:
                    if ligne[3] == "1":
                        fichier_txt.write(ligne[5])  

def tokenization(annee, model):
    processor = sentencepiece.SentencePieceProcessor(model_file=model)

    # Lecture du fichier texte entier
    with open(chemin_rad+annee+'.txt', 'r', encoding='utf-8') as fichier:
        contenu = fichier.read()

    # Tokenisation de l'ensemble du contenu
    tokens = processor.encode_as_pieces(contenu)


    with open('tokens_list.txt', 'w', encoding='utf-8') as fichier_sortie:
        for token in tokens:
            fichier_sortie.write(token + '\n')
    
def test_tokens(annee_test):
    with open(r'./tokens_list.txt', 'r', encoding='utf-8') as file:
        tokens = file.read().lower()
    
    with open(chemin_rad+annee_test+'.txt', 'r', encoding='utf-8') as file:
        text = file.read().lower()
    
    escaped_tokens = [re.escape(token) for token in tokens]
    pattern = '|'.join(escaped_tokens)
    all_tokens = re.findall(pattern, text)
    token_counts = Counter(all_tokens)


    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    tokens, counts = zip(*sorted_tokens)

    plt.figure(figsize=(15, 5))  # Ajustez la taille selon vos besoins
    plt.bar(tokens[:30], counts[:30])
    plt.xticks(rotation=45)
    plt.xlabel('Tokens')
    plt.ylabel('Nombre d\'Occurrences')
    plt.title('Histogramme des Occurrences de Tokens Spécifiques')
    plt.show()
    print(tokens[-10:])




#csv_in_txt()
dataset_model = [chemin_rad+'2023.txt',chemin_rad+'2022.txt']
model_training(dataset_model,1000)
tokenization('2018',model)
test_tokens('2018')