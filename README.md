# Helpdesk Copilot
### Smistamento automatico dei ticket e riuso della conoscenza con Machine Learning e RAG

**Capstone Project – Master in Python, AI & Machine Learning (EPICODE)**
**Autore:** Giovanni

**Repository del progetto:** https://github.com/GioForna/epicode_final_project

---

## 1. Problema
Un helpdesk aziendale riceve ogni giorno centinaia di ticket. Due attività consumano gran parte del tempo degli operatori:

1. **Smistamento (triage):** leggere ogni ticket e assegnarlo alla coda corretta (supporto tecnico, fatturazione, IT, resi, …). Un errore di smistamento allunga i tempi di risoluzione perché il ticket rimbalza tra i team.
2. **Risposta:** molte richieste sono simili a ticket già risolti, ma la conoscenza contenuta nelle risposte passate non viene riusata: l'operatore riscrive ogni volta da zero.

È un tipico problema di **knowledge management**: l'azienda possiede già la conoscenza (lo storico dei ticket risolti), ma non è organizzata né recuperabile nel momento in cui serve.

## 2. Obiettivo
Realizzare un prototipo, **Helpdesk Copilot**, che affianca l'operatore (*human-in-the-loop*, senza sostituirlo) e:

- **classifica automaticamente** un nuovo ticket nella coda corretta, indicando un livello di confidenza;
- **recupera i ticket storici più simili** e genera una **bozza di risposta** basata esclusivamente su di essi, citando le fonti.

## 3. Perimetro (cosa è incluso e cosa no)
| Incluso | Escluso |
|---|---|
| Ticket in lingua inglese | Altre lingue |
| Le 10 code principali dell'helpdesk | Integrazione con sistemi di ticketing reali (Jira, Zendesk, …) |
| Classificazione della coda + bozza di risposta | Invio automatico delle risposte al cliente |
| Prototipo dimostrativo (app web locale) | Messa in produzione, autenticazione, scalabilità |

La decisione finale resta **sempre all'operatore**.

## 4. Dataset
**Customer Support Tickets** (Hugging Face, *Tobi-Bueck/customer-support-tickets*, licenza CC BY-NC 4.0).
- 28.587 ticket, di cui **16.335 in inglese** dopo la pulizia (rimozione di valori mancanti e duplicati): è il sottoinsieme utilizzato;
- campi: oggetto, testo, **risposta dell'operatore**, **coda** (10 classi), tipo, priorità, tag;
- classi sbilanciate (da circa 4.700 ticket per *Technical Support* a circa 240 per *General Inquiry*);
- split stratificato per coda: 80% training (knowledge base e addestramento), 20% test (3.267 ticket).

Limite dichiarato: il dataset è di origine semi-sintetica. È adatto alla prototipazione, ma i risultati andrebbero validati su dati reali aziendali.

## 5. Architettura
Il sistema è composto da due rami che partono dallo stesso ticket, preceduti da un controllo sull'input:

1. **Guardrail di input:** validazione della lunghezza del testo e mascheramento dei dati personali.
2. **Ramo di classificazione:** TF-IDF → rete neurale (MLP) → coda + confidenza → soglia 0.70 → smistamento automatico oppure revisione umana.
3. **Ramo RAG:** embedding OpenAI (*text-embedding-3-small*) → ricerca per similarità coseno nella knowledge base (circa 13.000 ticket risolti) → soglia di similarità 0.45 → LLM (*gpt-4.1-mini*) → bozza di risposta con citazione delle fonti.
4. **Interfaccia:** app Streamlit che mostra all'operatore coda, confidenza, ticket simili e bozza.

Il codice è organizzato in moduli Python (`src/`): configurazione, download e pulizia dei dati, classificatore, guardrail, pipeline RAG, valutazione.

## 6. Metodologia
1. **Analisi esplorativa (EDA):** distribuzione delle classi, lunghezza dei testi, valori mancanti, parole più frequenti per coda.
2. **Preprocessing del testo:** unione di oggetto e corpo, pulizia, rimozione dei duplicati, vettorizzazione **TF-IDF**.
3. **Classificazione della coda (Machine Learning e Deep Learning):** confronto tra una baseline (classe più frequente), *Naive Bayes*, *Logistic Regression*, *Linear SVM* e una **rete neurale (MLP)**. Metriche: accuracy, F1 macro, matrice di confusione.
4. **Soglia di confidenza:** analisi del compromesso tra quota di ticket smistati in automatico e accuratezza, per scegliere la soglia.
5. **Knowledge base (RAG):** i ticket risolti del training set vengono convertiti in **embedding** e indicizzati; per un nuovo ticket si recuperano i *k* più simili.
6. **Generazione della bozza (GenAI):** l'LLM redige la risposta usando solo i ticket recuperati, con un prompt che impone vincoli di sicurezza.
7. **Valutazione:** *hit-rate@k* del retrieval (percentuale di casi in cui i ticket recuperati appartengono alla stessa coda del ticket di test) e test mirati dei guardrail (richieste fuori tema, *prompt injection*, dati personali).
8. **Demo:** applicazione **Streamlit** per l'operatore.

## 7. Guardrail
| Dove | Cosa |
|---|---|
| Input | Lunghezza minima e massima; mascheramento di email, telefono, IBAN, carta di credito e IP prima dell'invio all'API esterna |
| Classificazione | Sotto il 70% di confidenza il ticket va in revisione umana |
| Retrieval | Sotto similarità 0.45 nessuna bozza: la richiesta è considerata fuori dominio |
| LLM | Risponde solo sui ticket recuperati, cita le fonti, non chiede dati sensibili, ignora istruzioni contenute nel ticket; se il contesto non basta risponde *INSUFFICIENT_CONTEXT* |
| Processo | Nessuna azione automatica verso il cliente: decide sempre l'operatore |

## 8. Tecnologie
Python 3.12 · Pandas · NumPy · Matplotlib/Seaborn · scikit-learn · OpenAI API (embedding + LLM) · Streamlit · Jupyter · Hugging Face Hub

## 9. Moduli del Master applicati
| Modulo | Applicazione nel progetto |
|---|---|
| Python | Struttura modulare del codice, script riutilizzabili |
| Python for Data Science | Pulizia dei dati ed EDA con Pandas e Seaborn |
| Machine Learning | Classificazione supervisionata, confronto tra modelli, metriche, soglia di confidenza |
| Deep Learning | Rete neurale MLP confrontata con i modelli classici |
| NLP | Preprocessing del testo, TF-IDF, embedding semantici |
| Generative AI | Pipeline RAG con LLM, prompt con guardrail, app finale |

## 10. Stato del progetto e risultati
Il progetto è già stato sviluppato e i risultati sono disponibili nella repository indicata sopra.

| Modello | Accuracy | F1 macro |
|---|---|---|
| Baseline (classe più frequente) | 0.290 | 0.045 |
| Naive Bayes | 0.593 | 0.575 |
| Linear SVM | 0.665 | 0.688 |
| Logistic Regression | 0.670 | 0.690 |
| **Rete neurale (MLP)** | **0.688** | **0.696** |

- **Soglia di confidenza 0.70:** il 75% dei ticket viene smistato in automatico con accuracy del 79%; il resto va all'operatore.
- **Retrieval** (500 ticket di test): hit@1 79.4%, hit@5 87.2%.
- **Guardrail:** le richieste fuori tema vengono bloccate senza chiamare l'LLM, i tentativi di *prompt injection* vengono ignorati e i dati personali vengono mascherati.

## 11. Limiti e sviluppi futuri
- Dataset semi-sintetico con varianti parafrasate dello stesso ticket: le metriche di retrieval sono ottimistiche e serve una validazione su ticket reali;
- le code *Technical / Product / IT Support* e *Customer Service* si sovrappongono e concentrano gli errori di classificazione;
- le bozze sono valutate solo in modo qualitativo: in futuro *LLM-as-a-judge* e feedback degli operatori;
- modelli Transformer *fine-tuned* (es. BERT) per migliorare la classificazione;
- supporto multilingua (il dataset contiene anche ticket in tedesco);
- integrazione con un sistema di ticketing reale.
