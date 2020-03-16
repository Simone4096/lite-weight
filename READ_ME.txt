Nomi: Matteo Colanero, Simone Bombari
Nickname: LiteWeight
30/5/2019

Oltre ad aver commentato il codice, abbiamo creato le funzioni d'esempio "start_monk" e "start_cup" che dimostrano
	sommariamente il funzionamento del programma.

start_monk(loc_train, loc_test, num) prende in input il percorso dei file train e test (in formato .txt) e il numero del monk;
	inizia un processo di cross validation con una griglia di iperparametri fissata e successivamente esegue
	il training e il testing su un set fissato di iperparametri, restituendo MSE e Accuracy per il TR e il TS.
	
start_cup(loc_train, loc_int_test) prende in input il percorso dei file train e test interno (presenti nella cartella consegnata);
	inizia un processo di cross validation con una griglia di iperparametri fissata e successivamente esegue
	il training e l'internal testing su un set fissato di iperparametri, restituendo MEE per il TR e il TS.