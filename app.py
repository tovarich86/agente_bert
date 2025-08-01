# app.py (versão com Melhoria 1 e 2 e Indentação Corrigida)

import streamlit as st
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import requests
import re
import unicodedata
import logging
from pathlib import Path
import zipfile
import io
import shutil
import random
from pinecone import Pinecone
from datetime import datetime
from models import get_embedding_model, get_cross_encoder_model
from concurrent.futures import ThreadPoolExecutor # <<< MELHORIA 4 ADICIONADA
from tools import (
    find_companies_by_topic,
    get_final_unified_answer,
    suggest_alternative_query,
    analyze_topic_thematically,
    get_summary_for_topic_at_company,
    rerank_with_cross_encoder,
    rerank_by_recency,
    create_hierarchical_alias_map
    )
logger = logging.getLogger(__name__)

# --- Módulos do Projeto (devem estar na mesma pasta) ---
from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
from analytical_engine import AnalyticalEngine

# --- Configurações Gerais (Versão Refatorada para Pinecone) ---
st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# Importações necessárias para a nova lógica
from pinecone import Pinecone
import pandas as pd
import numpy as np

MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite" # Usando o modelo mais recente e eficiente
CVM_SEARCH_URL = "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"

# Define o nome do índice Pinecone e o arquivo de resumo
PINECONE_INDEX_NAME = "agente-rag-cvm" # Use o mesmo nome do seu índice
SUMMARY_FILENAME = "resumo_fatos_e_topicos_final_enriquecido.json"

# Define apenas os arquivos que AINDA são necessários para a aplicação
FILES_TO_DOWNLOAD = {
    SUMMARY_FILENAME: "https://github.com/tovarich86/agente_bert/releases/download/dados.v3/resumo_fatos_e_topicos_final_enriquecido.json"
}
CACHE_DIR = Path("data_cache")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- CARREGADOR DE DADOS (Versão Refatorada para Pinecone) ---
@st.cache_resource(show_spinner="Configurando o ambiente e conectando à base de conhecimento...")
def setup_and_load_data():
    """
    Nova versão: Baixa apenas o arquivo de resumo, extrai filtros dele e conecta-se ao Pinecone.
    Não carrega mais os pesados índices FAISS ou arquivos de chunks.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Lógica de download mantida para os arquivos que ainda são necessários
    for filename, url in FILES_TO_DOWNLOAD.items():
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            logger.info(f"Baixando arquivo '{filename}'...")
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"'{filename}' baixado com sucesso.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao baixar {filename} de {url}: {e}")
                st.stop()

    # --- Carregamento de Modelos (sem alterações) ---
    st.write("Carregando modelo de embedding...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    st.write("Carregando modelo de Re-ranking (Cross-Encoder)...")
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # --- LÓGICA DE CARREGAMENTO DE DADOS PRINCIPAIS ---

    # Carrega os dados de resumo para o AnalyticalEngine
    summary_file_path = CACHE_DIR / SUMMARY_FILENAME
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Erro crítico: '{SUMMARY_FILENAME}' não foi encontrado.")
        st.stop()

    # --- NOVA LÓGICA DE CONEXÃO AO PINECONE ---
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("Chave da API do Pinecone não configurada nos segredos do Streamlit.")
        st.stop()

    st.write("Conectando ao banco de dados vetorial (Pinecone)...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    st.write("Conexão estabelecida com sucesso!")
    # ---------------------------------------------

    # --- LÓGICA DE EXTRAÇÃO DE FILTROS (Preservada e Adaptada) ---
    # Agora, os filtros são extraídos do summary_data, que é leve.
    setores = set()
    controles = set()

    for company_data in summary_data.values():
        setor = company_data.get('setor')
        if isinstance(setor, str) and setor.strip():
            setores.add(setor.strip().capitalize())
        else:
            setores.add("Não identificado")

        controle = company_data.get('controle_acionario')
        if isinstance(controle, str) and controle.strip():
            controles.add(controle.strip().capitalize())
        else:
            controles.add("Não identificado")

    # Lógica de ordenação e formatação dos filtros (Preservada)
    sorted_setores = sorted([s for s in setores if s != "Não identificado"])
    if "Não identificado" in setores:
        sorted_setores.append("Não identificado")

    sorted_controles = sorted([c for c in controles if c != "Não identificado"])
    if "Não identificado" in controles:
        sorted_controles.append("Não identificado")

    all_setores = ["Todos"] + sorted_setores
    all_controles = ["Todos"] + sorted_controles

    logger.info(f"Filtros dinâmicos encontrados: {len(all_setores)-1} setores e {len(all_controles)-1} tipos de controle.")

    # A função agora retorna o índice Pinecone e os modelos, além dos outros dados.
    # O objeto 'artifacts' foi removido.
    return pinecone_index, embedding_model, cross_encoder_model, summary_data, all_setores, all_controles




# --- FUNÇÕES GLOBAIS E DE RAG ---

def _create_flat_alias_map(kb: dict) -> dict:
    alias_to_canonical = {}
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            canonical_name = topic_name_raw.replace('_', ' ')
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

AVAILABLE_TOPICS = list(set(_create_flat_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO).values()))

def expand_search_terms(base_term: str, kb: dict) -> list[str]:
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)

# Em app.py, substitua esta função
def search_by_tags(chunks_to_search: list[dict], target_tags: list[str]) -> list[dict]:
    """Busca chunks que contenham tags de tópicos específicos."""
    results = []
    target_tags_lower = {tag.lower() for tag in target_tags}

    for i, chunk_info in enumerate(chunks_to_search):
        chunk_text = chunk_info.get("text", "")
        found_topics_in_chunk = re.findall(r'\[topico:([^\]]+)\]', chunk_text)

        if found_topics_in_chunk:
            # O tópico pode ser uma lista, ex: [topico:Vesting,Aceleracao]
            topics_in_chunk_set = {t.strip().lower() for t in found_topics_in_chunk[0].split(',')}

            # Se houver qualquer sobreposição entre as tags procuradas e as encontradas
            if not target_tags_lower.isdisjoint(topics_in_chunk_set):
                results.append(chunk_info)
    return results

def get_final_unified_answer(query: str, context: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    has_complete_8_4 = "formulário de referência" in query.lower() and "8.4" in query.lower()
    has_tagged_chunks = "--- CONTEÚDO RELEVANTE" in context
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
    if has_complete_8_4:
        structure_instruction = "ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4: Use a estrutura oficial do item 8.4 do Formulário de Referência (a, b, c...)."
    elif has_tagged_chunks:
        structure_instruction = "PRIORIZE as informações dos chunks recuperados e organize a resposta de forma lógica."
    prompt = f"""Você é um consultor especialista em planos de incentivo de longo prazo (ILP).
    PERGUNTA ORIGINAL DO USUÁRIO: "{query}"
    CONTEXTO COLETADO DOS DOCUMENTOS:
    {context}
    {structure_instruction}
    INSTRUÇÕES PARA O RELATÓRIO FINAL:
    1. Responda diretamente à pergunta do usuário com base no contexto fornecido.
    2. Seja detalhado, preciso e profissional na sua linguagem. Use formatação Markdown.
    3. Se uma informação específica pedida não estiver no contexto, declare explicitamente: "Informação não encontrada nas fontes analisadas.". Não invente dados.
    RELATÓRIO ANALÍTICO FINAL:"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"Ocorreu um erro ao contatar o modelo de linguagem. Detalhes: {str(e)}"

# <<< MELHORIA 1 ADICIONADA >>>
def get_query_intent_with_llm(query: str) -> str:
    """
    Usa um LLM para classificar a intenção do usuário em 'quantitativa' ou 'qualitativa'.
    Retorna 'qualitativa' como padrão em caso de erro.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""
    Analise a pergunta do usuário e classifique a sua intenção principal. Responda APENAS com uma única palavra em JSON.

    As opções de classificação são:
    1. "quantitativa": Se a pergunta busca por números, listas diretas, contagens, médias, estatísticas ou agregações.
       Exemplos: "Quantas empresas têm TSR Relativo?", "Qual a média de vesting?", "Liste as empresas com desconto no strike.".
    2. "qualitativa": Se a pergunta busca por explicações, detalhes, comparações, descrições ou análises aprofundadas.
       Exemplos: "Como funciona o plano da Vale?", "Compare os planos da Hypera e Movida.", "Detalhe o tratamento de dividendos.".

    Pergunta do Usuário: "{query}"

    Responda apenas com o JSON da classificação. Exemplo de resposta: {{"intent": "qualitativa"}}
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50
        }
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()

        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        intent_json = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group())
        intent = intent_json.get("intent", "qualitativa").lower()

        logger.info(f"Intenção detectada pelo LLM: '{intent}' para a pergunta: '{query}'")

        if intent in ["quantitativa", "qualitativa"]:
            return intent
        else:
            logger.warning(f"Intenção não reconhecida '{intent}'. Usando 'qualitativa' como padrão.")
            return "qualitativa"

    except Exception as e:
        logger.error(f"ERRO ao determinar intenção com LLM: {e}. Usando 'qualitativa' como padrão.")
        return "qualitativa"

# <<< MELHORIA 2 APLICADA >>>
# Função modificada para lidar com buscas gerais (sem empresa)
# Em app.py, substitua esta função
def execute_dynamic_plan(
    query: str,
    plan: dict,
    pinecone_index: Pinecone.Index,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    # Os parâmetros extras da versão antiga já foram removidos na refatoração anterior
) -> tuple[str, list[dict]]:
    """
    Versão Final e Refatorada: Executa o plano de busca via Pinecone,
    re-ranqueia os resultados e constrói o contexto final.
    """
    logger.info(f"Executando plano de busca via Pinecone para query: '{query}'")

    # --- ETAPA 1: PREPARAR A CONSULTA ---
    topicos = plan.get("topicos", [])
    empresas = plan.get("empresas", [])
    filtros = plan.get("filtros", {})

    TOP_K_INITIAL_RETRIEVAL = 50
    TOP_K_FINAL = 10

    semantic_query = f"informações detalhadas sobre {', '.join(topicos)} para as empresas {', '.join(empresas)}"
    if not topicos and not empresas:
        semantic_query = query

    query_embedding = model.encode(semantic_query, normalize_embeddings=True).tolist()

    pinecone_filter = {}
    if filtros.get('setor'):
        pinecone_filter['setor'] = filtros['setor'].capitalize()
    if filtros.get('controle_acionario'):
        pinecone_filter['controle_acionario'] = filtros['controle_acionario'].capitalize()
    if empresas:
        pinecone_filter['company_name'] = {"$in": empresas}

    logger.info(f"Consultando Pinecone com o seguinte filtro de metadados: {pinecone_filter}")

    # --- ETAPA 2: EXECUTAR A BUSCA ---
    try:
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=TOP_K_INITIAL_RETRIEVAL,
            filter=pinecone_filter if pinecone_filter else None,
            include_metadata=True
        )
        # O resultado da busca agora é uma lista limpa chamada 'candidate_chunks'
        candidate_chunks = [match['metadata'] for match in results.get('matches', [])]
    except Exception as e:
        logger.error(f"Erro ao consultar o Pinecone: {e}")
        st.error(f"Ocorreu um erro ao buscar na base de conhecimento. Detalhes: {e}")
        return "", []

    # --- ETAPA 3: RE-RANKING (COM A VERIFICAÇÃO CORRIGIDA) ---

    # CORREÇÃO: A verificação agora é feita na lista 'candidate_chunks'
    if not candidate_chunks:
        logger.warning(f"Nenhum chunk candidato encontrado para a query: '{query}' com os filtros aplicados.")
        return "Não encontrei informações relevantes para esta combinação específica de consulta e filtros.", []

    logger.info(f"Total de {len(candidate_chunks)} chunks candidatos únicos encontrados. Re-ranqueando...")
    reranked_chunks = rerank_with_cross_encoder(query, candidate_chunks, cross_encoder_model, top_n=TOP_K_FINAL)

    # --- ETAPA 4: CONSTRUÇÃO DO CONTEXTO FINAL (Sem alterações) ---
    full_context = ""
    retrieved_sources = []
    seen_sources = set()

    for chunk in reranked_chunks:
        company_name = chunk.get('company_name', 'N/A')
        source_url = chunk.get('source_url', 'N/A')

        source_header = f"(Empresa: {company_name}, Setor: {chunk.get('setor', 'N/A')})"
        clean_text = chunk.get('text', '').strip()
        full_context += f"--- CONTEÚDO RELEVANTE {source_header} ---\n{clean_text}\n\n"

        source_tuple = (company_name, source_url)
        if source_tuple not in seen_sources:
            seen_sources.add(source_tuple)
            retrieved_sources.append(chunk)

    logger.info(f"Contexto final construído a partir de {len(reranked_chunks)} chunks re-ranqueados.")
    return full_context, retrieved_sources

def create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters: dict):
    """
    Versão 3.0 (Unificada) do planejador dinâmico.

    Esta versão combina o melhor de ambas as propostas:
    1.  EXTRAI filtros de metadados (setor, controle acionário).
    2.  EXTRAI tópicos hierárquicos completos.
    3.  RESTAURA a detecção de intenção de "Resumo Geral" para perguntas abertas.
    4.  MANTÉM a detecção da intenção especial "Item 8.4".
    """
    logger.info(f"Gerando plano dinâmico v3.0 para a pergunta: '{query}'")
    query_lower = query.lower().strip()

    plan = {
        "empresas": [],
        "topicos": [],
        "filtros": filters.copy(),
        "plan_type": "default" # O tipo de plano default aciona a busca RAG padrão.
    }



    # --- PASSO 2: Identificação Robusta de Empresas (Lógica Original Mantida) ---
    mentioned_companies = []
    if company_catalog_rich:
        companies_found_by_alias = {}
        for company_data in company_catalog_rich:
            canonical_name = company_data.get("canonical_name")
            if not canonical_name: continue

            all_aliases = company_data.get("aliases", []) + [canonical_name]
            for alias in all_aliases:
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            mentioned_companies = [c for c, s in sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)]

    if not mentioned_companies:
        for empresa_nome in summary_data.keys():
            if re.search(r'\b' + re.escape(empresa_nome.lower()) + r'\b', query_lower):
                mentioned_companies.append(empresa_nome)

    plan["empresas"] = mentioned_companies
    logger.info(f"Empresas identificadas: {plan['empresas']}")

    # --- PASSO 3: Detecção de Intenções Especiais (LÓGICA UNIFICADA) ---
    # Palavras-chave para as intenções especiais
    summary_keywords = ['resumo geral', 'plano completo', 'como funciona o plano', 'descreva o plano', 'resumo do plano', 'detalhes do plano']
    section_8_4_keywords = ['item 8.4', 'seção 8.4', '8.4 do fre']

    is_summary_request = any(keyword in query_lower for keyword in summary_keywords)
    is_section_8_4_request = any(keyword in query_lower for keyword in section_8_4_keywords)

    if plan["empresas"] and is_section_8_4_request:
        plan["plan_type"] = "section_8_4"
        # O tópico é o caminho hierárquico para a seção inteira
        plan["topicos"] = ["FormularioReferencia,Item_8_4"]
        logger.info("Plano especial 'section_8_4' detectado.")
        return {"status": "success", "plan": plan}

    # [LÓGICA RESTAURADA E ADAPTADA]
    # Se for uma pergunta de resumo para uma empresa, define um conjunto de tópicos essenciais.
    elif plan["empresas"] and is_summary_request:
        plan["plan_type"] = "summary" # Um tipo especial para indicar um resumo completo
        logger.info("Plano especial 'summary' detectado. Montando plano com tópicos essenciais.")
        # Define os CAMINHOS HIERÁRQUICOS essenciais para um bom resumo.
        plan["topicos"] = [
            "TiposDePlano",
            "ParticipantesCondicoes,Elegibilidade",
            "Mecanicas,Vesting",
            "Mecanicas,Lockup",
            "IndicadoresPerformance",
            "GovernancaRisco,MalusClawback",
            "EventosFinanceiros,DividendosProventos"
        ]
        return {"status": "success", "plan": plan}

    # --- PASSO 4: Extração de Tópicos Hierárquicos (Se Nenhuma Intenção Especial Foi Ativada) ---
    alias_map = create_hierarchical_alias_map(kb)
    with st.expander("🕵️ DEBUG: Conteúdo do Dicionário de Busca (Alias Map)"):
        st.json(alias_map)
    found_topics = set()

    # Ordena os aliases por comprimento para encontrar o mais específico primeiro
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        # Usamos uma regex mais estrita para evitar matches parciais (ex: 'TSR' em 'TSR Relativo')
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            found_topics.add(alias_map[alias])

    plan["topicos"] = sorted(list(found_topics))
    if plan["topicos"]:
        logger.info(f"Caminhos de tópicos identificados: {plan['topicos']}")
    if plan["empresas"] and not plan["topicos"]:
        logger.info("Nenhum tópico específico encontrado. Ativando modo de resumo/comparação geral.")
        plan["plan_type"] = "summary"
        # Define os CAMINHOS HIERÁRQUICOS essenciais para um bom resumo/comparação.
        plan["topicos"] = [
            "TiposDePlano",
            "ParticipantesCondicoes,Elegibilidade",
            "MecanicasCicloDeVida,Vesting",
            "MecanicasCicloDeVida,Lockup",
            "IndicadoresPerformance",
            "GovernancaRisco,MalusClawback",
            "EventosFinanceiros,DividendosProventos"
        ]
        logger.info(f"Tópicos de resumo geral adicionados ao plano: {plan['topicos']}")

    # --- PASSO 5: Validação Final ---
    if not plan["empresas"] and not plan["topicos"] and not plan["filtros"]:
        logger.warning("Planejador não conseguiu identificar empresa, tópico ou filtro na pergunta.")
        return {"status": "error", "message": "Não foi possível identificar uma intenção clara na sua pergunta. Tente ser mais específico."}

    return {"status": "success", "plan": plan}


def analyze_single_company(
    empresa: str,
    plan: dict,
    query: str,
    pinecone_index: Pinecone.Index, # << RECEBE pinecone_index
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable
) -> dict:
    """
    Executa o plano de análise para uma única empresa usando a nova arquitetura Pinecone.
    """
    # A lógica de criar um plano específico para a empresa é mantida
    single_plan = {
        'empresas': [empresa],
        'topicos': plan['topicos'],
        'filtros': plan.get('filtros', {})
    }

    # Chama a nova versão refatorada do executor de planos
    context, sources_list = execute_dynamic_plan_func(
        query,
        single_plan,
        pinecone_index, # << PASSA pinecone_index
        model,
        cross_encoder_model,
        kb
    )

    result_data = {
        "empresa": empresa,
        "resumos_por_topico": {topico: "Informação não encontrada" for topico in plan['topicos']},
        "sources": sources_list
    }

    # A lógica de resumir o contexto com o LLM é 100% preservada
    if context:
        summary_prompt = f"""
        Com base no CONTEXTO abaixo sobre a empresa {empresa}, crie um resumo para cada um dos TÓPICOS solicitados.
        Sua resposta deve ser APENAS um objeto JSON válido, sem nenhum texto adicional antes ou depois.

        TÓPICOS PARA RESUMIR: {json.dumps(plan['topicos'])}

        CONTEXTO:
        {context}

        FORMATO OBRIGATÓRIO DA RESPOSTA (APENAS JSON):
        {{
          "resumos_por_topico": {{
            "Tópico 1": "Resumo conciso sobre o Tópico 1...",
            "Tópico 2": "Resumo conciso sobre o Tópico 2...",
            "...": "..."
          }}
        }}
        """
        try:
            json_response_str = get_final_unified_answer_func(summary_prompt, context)
            json_match = re.search(r'\{.*\}', json_response_str, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                result_data["resumos_por_topico"] = parsed_json.get("resumos_por_topico", result_data["resumos_por_topico"])
            else:
                logger.warning(f"Não foi possível extrair JSON da resposta para a empresa {empresa}.")
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Erro ao processar o resumo JSON para {empresa}: {e}")

    return result_data


def handle_rag_query(
    query: str,
    pinecone_index: Pinecone.Index, # << RECEBE pinecone_index
    embedding_model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    summary_data: dict,
    filters: dict
) -> tuple[str, list[dict]]:
    """
    Orquestra o pipeline de RAG para perguntas qualitativas, incluindo a geração do plano,
    a execução da busca (com re-ranking) e a síntese da resposta final.
    """
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters)

        if plan_response['status'] != "success":
            status.update(label="⚠️ Falha na identificação", state="error", expanded=True)

            st.warning("Não consegui identificar uma empresa conhecida na sua pergunta para realizar uma análise profunda.")
            st.info("Para análises detalhadas, por favor, use o nome de uma das empresas listadas na barra lateral.")

            with st.spinner("Estou pensando em uma pergunta alternativa que eu possa responder..."):
                alternative_query = suggest_alternative_query(query, kb) # Passe o kb

            st.markdown("#### Que tal tentar uma pergunta mais geral?")
            st.markdown("Você pode copiar a sugestão abaixo ou reformular sua pergunta original.")
            st.code(alternative_query, language=None)

            # Retornamos uma string vazia para o texto e para as fontes, encerrando a análise de forma limpa.
            return "", []
        # --- FIM DO NOVO BLOCO ---

        plan = plan_response['plan']

        summary_keywords = ['resumo', 'geral', 'completo', 'visão geral', 'como funciona o plano', 'detalhes do plano']
        is_summary_request = any(keyword in query.lower() for keyword in summary_keywords)

        specific_topics_in_query = list({canonical for alias, canonical in _create_flat_alias_map(kb).items() if re.search(r'\b' + re.escape(alias) + r'\b', query.lower())})
        is_summary_plan = is_summary_request and not specific_topics_in_query

        if plan['empresas']:
            st.write(f"**🏢 Empresas identificadas:** {', '.join(plan['empresas'])}")
        else:
            st.write("**🏢 Nenhuma empresa específica identificada. Realizando busca geral.**")

        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        if is_summary_plan:
            st.info("💡 Modo de resumo geral ativado. A busca será otimizada para os tópicos encontrados.")

        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    final_answer, all_sources_structured = "", []
    seen_sources_tuples = set()

    # --- Lógica para Múltiplas Empresas (Comparação) ---
    if len(plan.get('empresas', [])) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas...")

        with st.spinner(f"Analisando {len(plan['empresas'])} empresas em paralelo..."):
            with ThreadPoolExecutor(max_workers=len(plan['empresas'])) as executor:
                futures = [
                    executor.submit(
                        analyze_single_company, # Chama a nova versão da função
                        empresa, plan, query,
                        pinecone_index, # << Passando o índice Pinecone
                        embedding_model, cross_encoder_model, kb,
                        company_catalog_rich, company_lookup_map,
                        execute_dynamic_plan, get_final_unified_answer
                    ) for empresa in plan['empresas']
                ]
                results = [future.result() for future in futures]

        for result in results:
            for src_dict in result.get('sources', []):
                company_name = src_dict.get('company_name')
                source_url = src_dict.get('source_url')

                if company_name and source_url:
                    src_tuple = (company_name, source_url)
                    if src_tuple not in seen_sources_tuples:
                        seen_sources_tuples.add(src_tuple)
                        all_sources_structured.append(src_dict)

        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            clean_results = []
            for company_result in results:
                # Remove a chave 'sources' temporariamente para limpeza
                sources = company_result.pop("sources", [])
                clean_sources = []
                for source_chunk in sources:
                    # Remove a chave 'relevance_score' de cada chunk
                    source_chunk.pop('relevance_score', None)
                    clean_sources.append(source_chunk)

                # Adiciona as fontes limpas de volta
                company_result["sources"] = clean_sources
                clean_results.append(company_result)
            structured_context = json.dumps(results, indent=2, ensure_ascii=False)
            comparison_prompt = f"""
            Sua tarefa é criar um relatório comparativo detalhado sobre "{query}".
            Use os dados estruturados fornecidos no CONTEXTO JSON abaixo.
            O relatório deve começar com uma breve análise textual e, em seguida, apresentar uma TABELA MARKDOWN clara e bem formatada.

            CONTEXTO (em formato JSON):
            {structured_context}
            """
            final_answer = get_final_unified_answer(comparison_prompt, structured_context)
            status.update(label="✅ Relatório comparativo gerado!", state="complete")

    # --- Lógica para Empresa Única ou Busca Geral ---
    else:
        with st.status("2️⃣ Recuperando contexto via Pinecone...", expanded=True) as status:
            context, all_sources_structured = execute_dynamic_plan(
                query, plan,
                pinecone_index, # << Passando o índice Pinecone
                embedding_model, cross_encoder_model, kb
            )

            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", []

            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto relevante selecionado!", state="complete")

        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, all_sources_structured

def main():
    st.title("🤖 Agente de Análise de Planos de Incentivo de Longo Prazo")
    st.markdown("---")

    # --- CARREGAMENTO DE DADOS E MODELOS (AGORA UNIFICADO) ---
    # A chamada para setup_and_load_data agora retorna o índice do Pinecone e os modelos.
    # O pesado objeto 'artifacts' não existe mais.
    (
        pinecone_index,
        embedding_model,
        cross_encoder_model,
        summary_data,
        setores_disponiveis,
        controles_disponiveis,
    ) = setup_and_load_data()

    # Validação para garantir que os dados essenciais foram carregados
    if not summary_data or not pinecone_index:
        st.error("❌ Falha crítica no carregamento dos dados ou na conexão com a base de conhecimento. O app não pode continuar.")
        st.stop()

    engine = AnalyticalEngine(summary_data, DICIONARIO_UNIFICADO_HIERARQUICO)

    try:
        from catalog_data import company_catalog_rich
    except ImportError:
        company_catalog_rich = []

    st.session_state.company_catalog_rich = company_catalog_rich


    from tools import _create_company_lookup_map
    st.session_state.company_lookup_map = _create_company_lookup_map(company_catalog_rich)


    with st.sidebar:
        st.header("📊 Informações do Sistema")

        # --- LÓGICA DA MÉTRICA ATUALIZADA ---
        # Tenta buscar as estatísticas do índice Pinecone para uma métrica mais relevante.
        try:
            index_stats = pinecone_index.describe_index_stats()
            # O total_vector_count é o número total de chunks que você indexou.
            st.metric("Documentos na Base de Conhecimento (RAG)", f"{index_stats.get('total_vector_count', 0):,}")
        except Exception as e:
            # Fallback caso a API de stats falhe, para não quebrar o app.
            logger.error(f"Não foi possível obter estatísticas do Pinecone: {e}")
            st.metric("Status da Base de Conhecimento", "Conectado")
        # --- FIM DA ATUALIZAÇÃO ---

        st.metric("Empresas no Resumo (Análise Rápida)", len(summary_data))
                # --- MODIFICAÇÃO 2: Usar as listas dinâmicas ---
        st.header("⚙️ Filtros da Análise")
        st.caption("Filtre a base de dados antes de fazer sua pergunta.")

        selected_setor = st.selectbox(
            label="Filtrar por Setor",
            options=setores_disponiveis, # Usa a lista dinâmica
            index=0
        )

        selected_controle = st.selectbox(
            label="Filtrar por Controle Acionário",
            options=controles_disponiveis, # Usa a lista dinâmica
            index=0
        )



        # Checkbox para ativar/desativar o re-ranking por recência
        prioritize_recency = st.checkbox(
        "Priorizar documentos mais recentes",
        value=True, # Ligado por padrão, pois é uma feature poderosa
        help="Dá um bônus de relevância para os documentos mais novos, fazendo com que apareçam primeiro nos resultados."
        )
        st.markdown("---")
        with st.expander("Empresas com dados no resumo"):
            st.dataframe(pd.DataFrame(sorted(list(summary_data.keys())), columns=["Empresa"]), use_container_width=True, hide_index=True)
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`")

    st.header("💬 Faça sua pergunta")

    # Em app.py, localize o bloco `with st.expander(...)` e substitua seu conteúdo por este:

    with st.expander("ℹ️ **Guia do Usuário: Como Extrair o Máximo do Agente**", expanded=False): # `expanded=False` é uma boa prática para não poluir a tela inicial
        st.markdown("""
        Este agente foi projetado para atuar como um consultor especialista em Planos de Incentivo de Longo Prazo (ILP), analisando uma base de dados de documentos públicos da CVM. Para obter os melhores resultados, formule perguntas que explorem suas principais capacidades.
        """)

        st.subheader("1. Perguntas de Listagem (Quem tem?) 🎯")
        st.info("""
        Use estas perguntas para identificar e listar empresas que adotam uma prática específica. Ideal para mapeamento de mercado.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Liste as empresas que pagam dividendos ou JCP durante o período de carência (vesting).
        - Quais companhias possuem cláusulas de Malus ou Clawback?
        - Gere uma lista de empresas com contrapartida do empregador (Matching/Coinvestimento).
        - Quais organizações mencionam explicitamente o Comitê de Remuneração como órgão aprovador dos planos?""")

        st.subheader("2. Análise Estatística (Qual a média?) 📈")
        st.info("""
        Pergunte por médias, medianas e outros dados estatísticos para entender os números por trás das práticas de mercado e fazer benchmarks.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Qual o período médio de vesting (em anos) entre as empresas analisadas?
        - Qual a diluição máxima média (% do capital social) que os planos costumam aprovar?
        - Apresente as estatísticas do desconto no preço de exercício (mínimo, média, máximo).
        - Qual o prazo de lock-up (restrição de venda) mais comum após o vesting das ações?""")

        st.subheader("3. Padrões de Mercado (Como é o normal?) 🗺️")
        st.info("""
        Faça perguntas abertas para que o agente analise diversos planos e descreva os padrões e as abordagens mais comuns para um determinado tópico.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Analise os modelos típicos de planos de Ações Restritas (RSU), o tipo mais comum no mercado.
        - Além do TSR, quais são as metas de performance (ESG, Financeiras) mais utilizadas pelas empresas?
        - Descreva os padrões de tratamento para condições de saída (Good Leaver vs. Bad Leaver) nos planos.
        - Quais as abordagens mais comuns para o tratamento de dividendos em ações ainda não investidas?""")

        st.subheader("4. Análise Profunda e Comparativa (Me explique em detalhes) 🧠")
        st.info("""
        Use o poder do RAG para pedir análises detalhadas sobre uma ou mais empresas, comparando regras e estruturas específicas.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Como o plano da Vale trata a aceleração de vesting em caso de mudança de controle?
        - Compare as cláusulas de Malus/Clawback da Vale com as do Itaú.
        - Descreva em detalhes o plano de Opções de Compra da Localiza, incluindo prazos, condições e forma de liquidação.
        - Descreva o Item 8.4 da M.dias Braco.
        - Quais as diferenças na elegibilidade de participantes entre os planos da Magazine Luiza e da Lojas Renner?""")


        st.subheader("❗ Conhecendo as Limitações")
        st.warning("""
        - **Fonte dos Dados:** Minha análise se baseia em documentos públicos da CVM com data de corte 31/07/2025. Não tenho acesso a informações em tempo real ou privadas.
        - **Identificação de Nomes:** Para análises profundas, preciso que o nome da empresa seja claro e reconhecível. Se o nome for ambíguo ou não estiver na minha base, posso não encontrar os detalhes.
        - **Escopo:** Sou altamente especializado em Incentivos de Longo Prazo. Perguntas fora deste domínio podem não ter respostas adequadas.
        """)

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quais são os modelos típicos de vesting? ou Como funciona o plano da Vale?")
    
    # ########################################################################## #
    # ## INÍCIO DA CORREÇÃO DE INDENTAÇÃO                                     ## #
    # ## O bloco de código abaixo foi movido para dentro da função `main()`.    ## #
    # ########################################################################## #
    
    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            st.stop()
        
        active_filters = {}
        if selected_setor != "Todos":
            active_filters['setor'] = selected_setor.lower()
        if selected_controle != "Todos":
            active_filters['controle_acionario'] = selected_controle.lower()
        
        if active_filters:
            filter_text_parts = []
            if 'setor' in active_filters:
                filter_text_parts.append(f"**Setor**: {active_filters['setor'].capitalize()}")
            if 'controle_acionario' in active_filters:
                filter_text_parts.append(f"**Controle**: {active_filters['controle_acionario'].capitalize()}")
            filter_text = " e ".join(filter_text_parts)
            st.info(f"🔎 Análise sendo executada com os seguintes filtros: {filter_text}")

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        # --- INÍCIO DA NOVA LÓGICA DE ROTEAMENTO HÍBRIDO ---
        intent = None
        query_lower = user_query.lower()
        final_answer = ""
        sources = []

        # 1. Camada de Regras
        quantitative_keywords = [
            'liste', 'quais empresas', 'quais companhias', 'quantas', 'média',
            'mediana', 'estatísticas', 'mais comuns', 'prevalência', 'contagem'
        ]
        if any(keyword in query_lower for keyword in quantitative_keywords):
            intent = "quantitativa"
            logger.info("Intenção 'quantitativa' detectada por regras de palavras-chave.")

        # 2. Camada de LLM
        if intent is None:
            with st.spinner("Analisando a intenção da sua pergunta..."):
                intent = get_query_intent_with_llm(user_query)
        
        # --- FIM DA NOVA LÓGICA DE ROTEAMENTO HÍBRIDO ---

        if intent == "quantitativa":
            listing_keywords = ["quais empresas", "liste as empresas", "quais companhias"]
            thematic_keywords = ["modelos típicos", "padrões comuns", "analise os planos", "formas mais comuns"]

            alias_map = create_hierarchical_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO)
            found_topics = set()
            for alias in sorted(alias_map.keys(), key=len, reverse=True):
                if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                    full_path = alias_map[alias]
                    topic_leaf = full_path.split(',')[-1].replace('_', ' ')
                    found_topics.add(topic_leaf)

            topics_to_search = list(found_topics)
            # Remove palavras-chave genéricas da lista de tópicos
            topics_to_search = [t for t in topics_to_search if t.lower() not in listing_keywords and t.lower() not in thematic_keywords]

            # Rota 1: Análise Temática
            if any(keyword in query_lower for keyword in thematic_keywords) and topics_to_search:
                with st.spinner(f"Iniciando análise temática... Este processo é detalhado e pode levar alguns minutos."):
                    st.write(f"**Tópico identificado para análise temática:** `{topics_to_search[0]}`")
                    final_report = analyze_topic_thematically(
                        topic=topics_to_search[0],
                        query=user_query,
                        summary_data=summary_data,
                        pinecone_index=pinecone_index,
                        embedding_model=embedding_model,
                        cross_encoder_model=cross_encoder_model,
                        kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                        execute_dynamic_plan_func=execute_dynamic_plan,
                        get_final_unified_answer_func=get_final_unified_answer,
                        filters=active_filters
                    )
                    st.markdown(final_report)

            # Rota 2: Listagem de Empresas por Tópico
            elif any(keyword in query_lower for keyword in listing_keywords) and topics_to_search:
                with st.spinner(f"Buscando empresas nos dados de resumo..."):
                    st.write(f"**Tópicos identificados para busca:** `{', '.join(topics_to_search)}`")
                    all_found_companies = set()
                    for topic_item in topics_to_search:
                        companies = find_companies_by_topic(
                            topic=topic_item,
                            summary_data=summary_data,
                            kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                            filters=active_filters
                        )
                        all_found_companies.update(companies)

                    if all_found_companies:
                        sorted_companies = sorted(list(all_found_companies))
                        final_answer = f"#### Foram encontradas {len(sorted_companies)} empresas para os tópicos relacionados:\n"
                        final_answer += "\n".join([f"- {company}" for company in sorted_companies])
                    else:
                        final_answer = "Nenhuma empresa encontrada nos documentos para os tópicos identificados."
                    st.markdown(final_answer)

            # Rota 2.5: Listagem de Empresas APENAS POR FILTRO
            elif any(keyword in query_lower for keyword in listing_keywords) and active_filters and not topics_to_search:
                with st.spinner("Listando empresas com base nos filtros selecionados..."):
                    st.write("Nenhum tópico técnico identificado. Listando todas as empresas que correspondem aos filtros.")
                    companies_from_filter = set()
                    for company_name, company_data in summary_data.items():
                        setor_metadata = company_data.get('setor', '')
                        setor_match = (not active_filters.get('setor') or
                                        (isinstance(setor_metadata, str) and setor_metadata.lower() == active_filters['setor']))
                        controle_metadata = company_data.get('controle_acionario', '')
                        controle_match = (not active_filters.get('controle_acionario') or
                                            (isinstance(controle_metadata, str) and controle_metadata.lower() == active_filters['controle_acionario']))
                        if setor_match and controle_match:
                            companies_from_filter.add(company_name)

                    if companies_from_filter:
                        sorted_companies = sorted(list(companies_from_filter))
                        final_answer = f"#### Foram encontradas {len(sorted_companies)} empresas para os filtros selecionados:\n"
                        final_answer += "\n".join([f"- {company}" for company in sorted_companies])
                    else:
                        final_answer = "Nenhuma empresa foi encontrada para a combinação de filtros selecionada."
                    st.markdown(final_answer)

            # Rota 3: Fallback para o AnalyticalEngine
            else:
                st.info("Intenção quantitativa detectada. Usando o motor de análise rápida...")
                with st.spinner("Executando análise quantitativa rápida..."):
                    report_text, data_result = engine.answer_query(user_query, filters=active_filters)
                    if report_text: st.markdown(report_text)
                    if data_result is not None:
                        if isinstance(data_result, pd.DataFrame):
                            if not data_result.empty: st.dataframe(data_result, use_container_width=True, hide_index=True)
                        elif isinstance(data_result, dict):
                            for df_name, df_content in data_result.items():
                                if df_content is not None and not df_content.empty:
                                    st.markdown(f"#### {df_name}")
                                    st.dataframe(df_content, use_container_width=True, hide_index=True)
                    else:
                        st.info("Nenhuma análise tabular foi gerada para a sua pergunta ou dados insuficientes.")
        else: # Se a intenção for 'qualitativa'
            final_answer, sources = handle_rag_query(
                user_query,
                pinecone_index,
                embedding_model,
                cross_encoder_model,
                DICIONARIO_UNIFICADO_HIERARQUICO,
                st.session_state.company_catalog_rich,
                st.session_state.company_lookup_map,
                summary_data,
                active_filters
            )
            st.markdown(final_answer)

        if sources:
            with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=True):
                st.caption("Nota: Links diretos para a CVM podem falhar. Use a busca no portal com o protocolo como plano B.")
                for src in sorted(sources, key=lambda x: x.get('company_name', '')):
                    company_name = src.get('company_name', 'N/A')
                    doc_date = src.get('document_date', 'N/A')
                    doc_type_raw = src.get('doc_type', '')
                    url = src.get('source_url', '')
                    if doc_type_raw == 'outros_documentos':
                        display_doc_type = 'Plano de Remuneração'
                    else:
                        display_doc_type = doc_type_raw.replace('_', ' ')
                    display_text = f"{company_name} - {display_doc_type} - (Data: **{doc_date}**)"
                    if "frmExibirArquivoIPEExterno" in url:
                        protocolo_match = re.search(r'NumeroProtocoloEntrega=(\d+)', url)
                        protocolo = protocolo_match.group(1) if protocolo_match else "N/A"
                        st.markdown(f"**{display_text}** (Protocolo: **{protocolo}**)")
                        st.markdown(f"↳ [Link Direto para Plano de ILP]({url}) ", unsafe_allow_html=True)
                    elif "frmExibirArquivoFRE" in url:
                        st.markdown(f"**{display_text}**")
                        st.markdown(f"↳ [Link Direto para Formulário de Referência]({url})", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{display_text}**: [Link]({url})")

if __name__ == "__main__":
    main()
