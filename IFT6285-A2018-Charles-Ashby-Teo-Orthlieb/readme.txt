Nos documents sont divisés de la manière suivantes:

- lstm: Le répertoire "lstm" contient tous les fichiers pour entraîner et évaluer nos modèles de type LSTM
- util: Le fichier "util" contient diverses fonctions utiles au traitement des données
- results: Le fichier "results" contient les données de tests entraînées sur nos modèles
    e.g. Le fichier results/lstmpc10k8-4n15_0c_40u.txt contient les résultats de notre modèle LSTM avec un dictionnaire
        de taille 10,000, entraînée avec 15 % de bruit avec comme hyper-paramètre prefix_len = 8 et suffix_len = 4
        pour le fichier avec 40% de "unk" et mots non contigues
	
	Le format des fichiers de résultat est le suivant:
	 results/lstmp{clean set (bool)}{dict_len}{prefix_len}-{suffix_len}_{mots contigues (bool)}c_{% unk}u.txt
- models: Implémentation des modèles du n-gram naïf et avec back-off
- evaluation: Fichier pour évaluer les modèles de n-gram naïf et avec back-off 

- main:
    Vous pouvez entraîner ou évaluer un modèle en entrant les différents hyper-paramètres dans la fonction "train()"
    ou "eval()" et en roulant le fichier main.py "python main.py"
