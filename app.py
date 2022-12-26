import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
import langdetect
import sentencepiece
st.title("Multilingual News Summarizer")

@st.cache(suppress_st_warning = True)
def multilingual_summarizer(news):
    global res
    lang = langdetect.detect(news)
    model = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
    pipe = pipeline('summarization', model = model, device = 0 if torch.cuda.is_available() else -1)
    if lang == 'en':
        res = pipe(news)[0]['summary_text']
        print(res)
    if lang == 'hi':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    elif lang =='ar':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    elif lang == 'zh':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    elif lang == 'ja':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    elif lang == 'es':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    elif lang == 'ru':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    elif lang == 'fr':
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        batch = tokenizer([news], return_tensors = 'pt')
        ids = model.generate(**batch)
        text = tokenizer.batch_decode(ids, skip_special_tokens = True)[0]
        res = pipe(text)[0]['summary_text']
        print(res)
    return res, lang



news = st.text_area(label = 'Enter text to summarize: ', value="", height=300, max_chars=514, key=None,help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
#multilingual_summarizer(' ia m good', 'en')
#lang = st.selectbox(label = 'Select language:', options = ['en - English', 'hi - Hindi','zh - Chinese', 'ru - Russian', 'ja - Japanese', 'ar - Arabic', 'es - Spanish', 'fr - French'])


if st.button(label = 'Translate and Summarize'):
    result, lang = multilingual_summarizer(news)
    st.success(result)
    #st.success(lang)
