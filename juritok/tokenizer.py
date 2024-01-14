import  sentencepiece
import csv
from transformers import XLNetTokenizer
import matplotlib.pyplot as plt
import re
from collections import Counter



# Paths
model = './spiece.model'
model = './mymodel.model' 

def model_pretrained(): #utiliser un modèle déjà pré-entrainé 
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    tokenizer.save_pretrained('./')

def model_training(dataset,vocab_size): # entrainement sur le dataset du jorf
    sentencepiece.SentencePieceTrainer.train(input=dataset, 
                                            model_prefix='mymodel', 
                                            vocab_size=vocab_size, 
                                            character_coverage=1.0,
                                            model_type='unigram')
    print('end training')

def csv_in_txt(): # transforme txt in cv
    for annee in ['2018','2019','2020','2021','2022','2023']:
        with open(chemin_rad+annee+'.csv', 'r', newline='', encoding='utf-8') as fichier_csv:
            lecteur_csv = csv.reader(fichier_csv, delimiter='|')
            
            with open(chemin_rad+annee+'.txt', 'w', encoding='utf-8') as fichier_txt:
                i=0
                for ligne in lecteur_csv:
                    if ligne[3] == "1":
                        fichier_txt.write(ligne[5]+'\n')  
    print('end csv in txt')

def tokenization(annee, model): # extrait les token d'un texte avec le modèle
    processor = sentencepiece.SentencePieceProcessor(model_file=model)

    # Lecture du fichier texte entier
    with open(chemin_rad+annee+'.txt', 'r', encoding='utf-8') as fichier:
        contenu = fichier.read()

    # Tokenisation de l'ensemble du contenu
    tokens = processor.encode_as_pieces(contenu)


    with open('tokens_list.txt', 'w', encoding='utf-8') as fichier_sortie:
        for token in tokens:
            fichier_sortie.write(token + '\n')
    print('end tokenisation')

def test_tokens(annee_test): # test les tokens les plus longs et regarde les occurrences
    with open(r'./tokens_list.txt', 'r', encoding='utf-8') as file:
        tokens_long = [line.strip().lower() for line in file if len(line.strip()) > 15]
    tokens_long = set(tokens_long)
    with open(chemin_rad+annee_test+'.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Distribution des tokens longs dans le texte
    escaped_tokens = [re.escape(token) for token in tokens_long]
    print('total token > 15 characters :',len(escaped_tokens))
    pattern = '|'.join(escaped_tokens)
    print('begin search of tokens in the txt, it can take some time')
    all_tokens = re.findall(pattern, text)
    token_counts = Counter(all_tokens)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    tokens, counts = zip(*sorted_tokens)

    # token les plus longs hors texte
    print('begin sort tokens')
    sorted_tokens = sorted(tokens_long, key=len, reverse=True)

    # figure
    plt.figure(figsize=(10, 5))
        # histogramme
    ax1 = plt.gca()  
    ax1.bar(tokens[:20], counts[:20])
    ax1.set_xlabel('Tokens > 15 caractères')
    ax1.set_ylabel('Nombre d\'Occurrences')
    ax1.tick_params(axis='y', labelcolor='b')  
    ax1.set_yscale('linear')  
    plt.xticks(rotation=45, ha='right')
        # histo en log
    ax2 = ax1.twinx()
    ax2.plot(tokens[:20], counts[:20], 'r-') 
    ax2.set_ylabel('',color='r') 
    ax2.tick_params(axis='y', labelcolor='r') 
    ax2.set_yscale('log') 
    ax1.set_title('Distribution des tokens longs (> 15 caractères) sur l\'année '+annee_test)
    ax1.text(0.35, 0.5, 'Analyse :\nFraction longs tokens effectivement présents dans le texte ={}%\nToken les plus longs : {}\n                                  {}\n                                  {}\n                                  {}\n                                  {}'.format(
        round(len(tokens)/len(tokens_long)*100,2),sorted_tokens[0][1:],sorted_tokens[1][1:],sorted_tokens[2][1:],sorted_tokens[3][1:],sorted_tokens[4][1:]),
        transform=ax1.transAxes)
    plt.tight_layout() 
    plt.show()




    print('end test')


# Paramètres :
chemin_rad = r'C:\Users\seran\juritok\files\jorf_' # modifier selon position des csv jorf
annee_test = '2018' #année sui sera tokenisée et testée
dataset_model = [chemin_rad+'2023.txt',chemin_rad+'2022.txt',chemin_rad+'2021.txt',chemin_rad+'2020.txt'] # ensemble des data d'entrainement
vocab_size = 40000



csv_in_txt()
model_training(dataset_model,vocab_size)
tokenization(annee_test,model)
test_tokens(annee_test)