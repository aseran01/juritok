import  sentencepiece
import csv
from transformers import XLNetTokenizer




# Chemin vers votre fichier CSV
chemin_csv = r'C:\Users\seran\juritok\files\jorf_2023.csv'

# Chemin vers le fichier texte de sortie
chemin_txt = r'C:\Users\seran\juritok\files\jorf_2023.txt'

model = './spiece.model'

def model_save():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    tokenizer.save_pretrained('./')

def csv_in_txt():
    with open(chemin_csv, 'r', newline='', encoding='utf-8') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv, delimiter='|')
        


        # Ouvrir le fichier de sortie
        with open(chemin_txt, 'w', encoding='utf-8') as fichier_txt:
            i=0
            for ligne in lecteur_csv:
                if ligne[3] == "1":
                    fichier_txt.write(ligne[5])  
def tokenization():
    processor = sentencepiece.SentencePieceProcessor(model_file=model)

    # Lecture du fichier texte entier
    with open(chemin_txt, 'r', encoding='utf-8') as fichier:
        contenu = fichier.read()

    # Tokenisation de l'ensemble du contenu
    tokens = processor.encode_as_pieces(contenu)


    with open('tokens_list.txt', 'w', encoding='utf-8') as fichier_sortie:
        for token in tokens:
            fichier_sortie.write(token + '\n')


tokenization()
