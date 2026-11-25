# Arabic NLP QA Assistant

An Arabic question-answering assistant that answers a user's question, predicts its topic category, and translates the result into English, all in one Streamlit app. Built for the *Natural Language Processing* course at Al Hussein Technical University (2026).

## What the app does

Enter a question in Arabic, and the app:

1. **Generates an answer** with a fine-tuned **Qwen2.5-1.5B-Instruct** model (LoRA adapter)
2. **Predicts the question's category** with a **TF-IDF + Linear SVM** classifier
3. **Translates** the question, answer and category into English using **Qwen2.5-72B-Instruct** through the Hugging Face Inference API

## Data

- **AAFAQ dataset:** Arabic question–answer pairs, each labelled with a category
- The dataset was provided by the course, so it is not included in this repo.

## Notebooks

| # | File | What it does |
|---|------|--------------|
| 1 | `1_preprocessing_classification.ipynb` | Exploratory analysis and cleaning for classification: removes duplicates and questions with conflicting labels, emojis and Arabic stop words; translates stray English words; strips tashkeel and tatweel; normalises hamza; removes punctuation; applies ISRI stemming and label encoding |
| 2 | `2_preprocessing_qa_mt.ipynb` | Cleaning for question answering and machine translation: keeps the longest answer for duplicated questions, and normalises ta marbuta and alef maqsura. It keeps one copy *with* tashkeel for translation and one copy without it for question answering, then tokenises and stems |
| 3 | `3_qa_seq2seq_rnn.ipynb` | Baseline encoder–decoder models: SimpleRNN, LSTM and GRU, each with one-hot, FastText and AraBERT embeddings |
| 4 | `4_qa_transformers.ipynb` | Fine-tunes three transformer models (mT5, Qwen2.5-1.5B with LoRA, araGPT2) and compares them with BLEU |
| – | `app.py` | Streamlit interface that combines question answering, classification and translation |

## Results: question answering (test set of 929 questions)

| Model | BLEU (sacreBLEU, 0–100) |
|-------|-------------------------|
| **Qwen2.5-1.5B-Instruct + LoRA** | **9.17** |
| mT5 | 8.86 |
| araGPT2 | 7.64 |

The RNN, LSTM and GRU baselines in notebook 3 scored close to zero BLEU. They could not produce fluent Arabic answers from about 3,700 training pairs. That is why the final app uses the fine-tuned transformer model. BLEU is a strict word-overlap score, and low absolute values are common for open-ended Arabic answers.

## Run the app

```bash
pip install streamlit pandas nltk pyarabic scikit-learn torch transformers peft requests
```

1. Place `NLP_dataset.csv` (output of notebook 1) next to `app.py`.
2. Place the fine-tuned adapter folder `qwen_qa_final` (output of notebook 4) next to `app.py`.
3. Set your Hugging Face token as an environment variable. It is never stored in the code.
   ```bash
   # Windows
   set HF_TOKEN=your_token
   # macOS / Linux
   export HF_TOKEN=your_token
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Limitations and next steps

- The classifier is trained inside the app each time it starts, and it isn't evaluated on held-out data. A separate training and evaluation notebook would make it more robust.
- Answer quality is limited by the dataset's size. More data, or retrieval from a knowledge base (RAG), would make answers more accurate.
- Translation depends on an external API, so it needs an internet connection and a valid token.

## Tools

Python, NLTK, PyArabic, scikit-learn, TensorFlow/Keras, Gensim (FastText), Hugging Face Transformers, PEFT (LoRA), Streamlit
