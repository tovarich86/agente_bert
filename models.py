# models.py
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

@st.cache_resource(show_spinner="Carregando modelo de embedding...")
def get_embedding_model():
    logger.info(f"Carregando modelo de embedding: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model

@st.cache_resource(show_spinner="Carregando modelo de re-ranking...")
def get_cross_encoder_model():
    logger.info(f"Carregando cross-encoder: {CROSS_ENCODER_MODEL_NAME}")
    model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return model
