from transformers import pipeline
import streamlit as st


@st.cache_resource
def load_translator():
    return pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")


def load_text():
    return st.text_input('RU -> EN', 'Введите текст для перевода...')


def print_translation(translation):
    st.write(translation[0]['translation_text'])


st.title('Light Translator')

st.subheader('Русско-Английский переводчик.')

text = load_text()
translator = load_translator()
result = st.button('Перевести!')
if result:
    translation = translator(text)
    st.write('**Перевод:**')
    print_translation(translation)
