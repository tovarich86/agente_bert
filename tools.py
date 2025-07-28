# tools_v3.0.py (Versão Unificada e Completa)
#
# DESCRIÇÃO:
# Este módulo combina as melhores características de ambas as versões anteriores.
#
# FUNCIONALIDADES PRESERVADAS DO ORIGINAL:
# 1. Busca Híbrida: Combina busca vetorial e busca por metadados para máxima relevância.
# 2. Motor de Sugestão Inteligente: Usa CAPABILITY_MAP e QUESTION_TEMPLATES para
#    gerar sugestões de perguntas diversificadas e garantidas.
# 3. Normalização de Nomes de Empresas: Reintroduzida a função _create_company_lookup_map.
#
# MELHORIAS INTEGRADAS DA VERSÃO ATUALIZADA:
# 1. Dicionário Hierárquico: Totalmente compatível com a estrutura de KB aninhada (v6.0).
# 2. Filtros de Metadados: Suporte para filtrar buscas por setor, controle acionário, etc.
# 3. Código Robusto: Inclui logging e tratamento de exceções aprimorados.

import faiss
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import requests
import logging
import random
from datetime import datetime
import math

logger = logging.getLogger(__name__)

# --- ESTRUTURAS DE DADOS DA VERSÃO ORIGINAL (PRESERVADAS PELA SUA ROBUSTEZ) ---
# Habilidades: 'thematic', 'listing', 'statistic', 'comparison'
CAPABILITY_MAP = {
    # Mapeia o TÓPICO FINAL (folha da hierarquia) para suas capacidades
    "AcoesRestritas": ["listing", "thematic", "comparison", "example_deep_dive"],
    "OpcoesDeCompra": ["listing", "thematic", "comparison", "example_deep_dive"],
    "AcoesFantasmas": ["listing", "thematic", "comparison"],
    "Matching Coinvestimento": ["listing", "thematic", "comparison"],
    "Vesting": ["statistic", "thematic", "listing", "comparison", "example_deep_dive"],
    "Lockup": ["statistic", "thematic", "listing", "comparison"],
    "PrecoDesconto": ["statistic", "listing", "thematic"],
    "VestingAcelerado": ["listing", "thematic", "comparison"],
    "Outorga": ["thematic", "listing"],
    "MalusClawback": ["listing", "thematic", "comparison", "example_deep_dive"],
    "Diluicao": ["statistic", "listing", "thematic"],
    "OrgaoDeliberativo": ["listing", "thematic"],
    "Elegibilidade": ["listing", "thematic", "comparison"],
    "CondicaoSaida": ["thematic", "listing", "comparison", "example_deep_dive"],
    "TSR Relativo": ["listing", "thematic", "comparison", "example_deep_dive"],
    "TSR Absoluto": ["listing", "thematic", "comparison"],
    "ESG": ["listing", "thematic"],
    "GrupoDeComparacao": ["thematic", "listing"],
    "DividendosProventos": ["listing", "thematic", "comparison", "example_deep_dive"],
    "MudancaDeControle": ["listing", "thematic", "comparison"],
}

QUESTION_TEMPLATES = {
    "thematic": [
        "Analise os modelos típicos de **{topic}** encontrados nos planos das empresas.",
        "Quais são as abordagens mais comuns para **{topic}** no mercado brasileiro?",
        "Descreva os padrões de mercado relacionados a **{topic}**."
    ],
    "listing": [
        "Quais empresas na base de dados possuem planos com **{topic}**?",
        "Gere uma lista de companhias que mencionam **{topic}** em seus documentos.",
        "Liste as empresas que adotam práticas de **{topic}**."
    ],
    "statistic": [
        "Qual o valor médio ou mais comum para o tópico de **{topic}** entre as empresas?",
        "Apresente as estatísticas (média, mediana, máximo) para **{topic}**.",
        "Qual a prevalência de **{topic}** nos planos analisados?"
    ],
    "comparison": [
        "Compare como a Vale e a Petrobras abordam o tópico de **{topic}** em seus planos.",
        "Quais as principais diferenças entre os planos da Magazine Luiza e da Localiza sobre **{topic}**?",
    ],
    "example_deep_dive": [
        "Como o plano da Vale define e aplica o conceito de **{topic}**?",
        "Descreva em detalhes como a Hypera trata a questão de **{topic}** em seu plano de remuneração.",
    ]
}

# --- NOVAS FUNÇÕES DE MAPEAMENTO HIERÁRQUICO (DA v2.1) ---

def create_hierarchical_alias_map(kb: dict) -> dict:
    alias_map = {}
    def _recursive_builder(sub_dict, path_so_far):
        for topic_key, topic_data in sub_dict.items():
            current_path = path_so_far + [topic_key]
            path_str = ",".join(current_path)
            for alias in topic_data.get("aliases", []):
                alias_map[alias.lower()] = path_str
            canonical_alias = topic_key.replace('_', ' ').lower()
            if canonical_alias not in alias_map:
                alias_map[canonical_alias] = path_str
            if "subtopicos" in topic_data and isinstance(topic_data.get("subtopicos"), dict):
                _recursive_builder(topic_data["subtopicos"], current_path)

    for section_key, section_data in kb.items():
        path_str = section_key
        canonical_alias = section_key.replace('_', ' ').lower()
        if canonical_alias not in alias_map:
            alias_map[canonical_alias] = path_str
        _recursive_builder(section_data, [section_key])
    return alias_map

# --- FUNÇÃO DE BUSCA ADAPTADA PARA A NOVA ESTRUTURA DE DADOS E FAISS COMPATÍVEL ---


def _create_company_lookup_map(company_catalog_rich: list) -> dict:
    """
    (REINTRODUZIDA) Cria um dicionário de mapeamento reverso para normalizar nomes de empresas.
    """
    lookup_map = {}
    if not company_catalog_rich:
        return lookup_map
        
    for company_data in company_catalog_rich:
        canonical_name = company_data.get("canonical_name")
        if not canonical_name:
            continue
        
        all_names_to_map = [canonical_name] + company_data.get("aliases", [])
        
        for name in all_names_to_map:
            lookup_map[name.lower()] = canonical_name
            
    return lookup_map

def rerank_by_recency(search_results, chunk_map, decay_factor=0.95):
    """
    Re-ranqueia os resultados da busca, dando um bônus para documentos mais recentes.
    Um decay_factor de 0.95 significa que cada ano de idade do documento mantém 95% do seu "bônus".
    
    Args:
        search_results (list): Lista de tuplas (score, chunk_id) da busca FAISS.
        chunk_map (list): O mapa de chunks completo.
        decay_factor (float): Fator de decaimento anual.
        
    Returns:
        list: A mesma lista de entrada, mas reordenada.
    """
    if not search_results:
        return []

    current_year = datetime.now().year
    reranked_results = []

    for score, chunk_id in search_results:
        metadata = chunk_map[chunk_id]
        new_score = score
        
        try:
            doc_date_str = metadata.get("document_date")
            if doc_date_str and doc_date_str != "N/A":
                # Extrai o ano da data no formato 'AAAA-MM-DD'
                doc_year = int(doc_date_str.split('-')[0])
                age = max(0, current_year - doc_year)
                
                # A distância L2 do FAISS é "menor é melhor".
                # Multiplicamos por um fator < 1 para diminuir a distância (melhorar o score) de itens recentes.
                recency_boost = decay_factor ** age
                new_score = score * recency_boost
        except (ValueError, IndexError) as e:
            logger.warning(f"Não foi possível processar a data para o chunk {chunk_id}: {doc_date_str} - {e}")
            pass # Mantém o score original se a data for inválida

        reranked_results.append((new_score, chunk_id))

    # Reordena a lista final pelo novo score (menor é melhor)
    return sorted(reranked_results, key=lambda x: x[0])

def get_final_unified_answer(query: str, context: str) -> str:
    """Chama a API do LLM para gerar uma resposta final sintetizada."""
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-1.5-flash-latest" # Mantém o modelo mais recente
    if not GEMINI_API_KEY: return "Erro: A chave da API do Gemini não foi configurada."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
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
        candidates = response.json().get('candidates', [])
        if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
            return candidates[0]['content']['parts'][0]['text'].strip()
        else:
            logger.error(f"Resposta inesperada da API Gemini: {response.json()}")
            return "Ocorreu um erro ao processar a resposta do modelo de linguagem."
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"Ocorreu um erro ao contatar o modelo de linguagem. Detalhes: {str(e)}"

def rerank_with_cross_encoder(query: str, chunks: list[dict], cross_encoder_model: CrossEncoder, top_n: int = 10) -> list[dict]:
    """Usa um Cross-Encoder para reordenar chunks por relevância."""
    if not chunks or not query: return []
    pairs = [[query, chunk.get('text', '')] for chunk in chunks]
    try:
        scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
        for i, chunk in enumerate(chunks):
            chunk['relevance_score'] = scores[i]
        reranked_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        return reranked_chunks[:top_n]
    except Exception as e:
        logger.error(f"Erro durante o re-ranking com CrossEncoder: {e}")
        return chunks[:top_n]

# --- MOTOR DE SUGESTÃO ROBUSTO (LÓGICA ORIGINAL RESTAURADA) ---
def suggest_alternative_query(failed_query: str, kb: dict) -> str:
    """
    Motor de sugestão robusto que usa os mapas enriquecidos (CAPABILITY_MAP) para gerar
    sugestões variadas, relevantes e garantidas de funcionar.
    """
    logger.info("---EXECUTANDO MOTOR DE SUGESTÃO ROBUSTO (v3 Unificado)---")

    alias_map = create_hierarchical_alias_map(kb)
    
    # Identifica tópicos na consulta falha
    topics_found_paths = set()
    for alias, path in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', failed_query.lower()):
            topics_found_paths.add(path)

    safe_suggestions = []
    context_for_llm = ""

    if topics_found_paths:
        primary_path = list(topics_found_paths)[0]
        # Pega o tópico mais específico (última parte do caminho) para usar com os mapas
        primary_topic_key = primary_path.split(',')[-1]
        primary_topic_display = primary_topic_key.replace('_', ' ')
        
        context_for_llm = f"O usuário demonstrou interesse no tópico de '{primary_topic_display}'."
        logger.info(f"---TÓPICO IDENTIFICADO: '{primary_topic_key}'---")

        capabilities = CAPABILITY_MAP.get(primary_topic_key, ["listing", "thematic"])
        logger.info(f"---HABILIDADES ENCONTRADAS PARA O TÓPICO: {capabilities}---")
        
        for cap in capabilities:
            template_variations = QUESTION_TEMPLATES.get(cap)
            if template_variations:
                chosen_template = random.choice(template_variations)
                safe_suggestions.append(chosen_template.format(topic=primary_topic_display))
    else:
        context_for_llm = "O usuário fez uma pergunta genérica ou sobre um tópico não reconhecido."
        logger.warning("---NENHUM TÓPICO IDENTIFICADO. USANDO SUGESTÕES GERAIS.---")
        safe_suggestions = [
            "Liste as empresas que utilizam TSR Relativo como métrica de performance.",
            "Analise os modelos típicos de planos de Ações Restritas (RSU).",
            "Como funciona o plano de vesting da Vale?"
        ]

    safe_suggestions = safe_suggestions[:3]

    prompt = f"""
    Você é um assistente de IA prestativo. A pergunta de um usuário falhou.
    PERGUNTA ORIGINAL: "{failed_query}"
    CONTEXTO: {context_for_llm}
    Eu gerei uma lista de perguntas "seguras". Sua tarefa é apresentá-las de forma clara e convidativa. Você pode fazer pequenas melhorias para soar mais natural, mas mantenha a intenção original.

    PERGUNTAS SEGURAS PARA APRESENTAR:
    {json.dumps(safe_suggestions, indent=2, ensure_ascii=False)}

    Apresente o resultado como uma lista de marcadores em Markdown.
    """
    return get_final_unified_answer(prompt, "")


# --- FERRAMENTA DE BUSCA HÍBRIDA (LÓGICA ORIGINAL RESTAURADA E ADAPTADA) ---

# Em tools.py, substitua a função inteira por esta versão final e precisa:

def find_companies_by_topic(
    topic: str,
    summary_data: dict,  # Parâmetro principal agora
    kb: dict,
    filters: dict = None
) -> list[str]:
    """
    Ferramenta de Listagem PRECISA.
    Usa os dados de resumo pré-analisados (a mesma fonte do AnalyticalEngine)
    para garantir uma listagem 100% consistente com a análise geral.
    """
    alias_map = create_hierarchical_alias_map(kb)
    # Precisamos encontrar o caminho completo para o tópico (ex: 'IndicadoresPerformance,TSR')
    # para podermos procurá-lo na estrutura aninhada do resumo.
    topic_path_str = alias_map.get(topic.lower(), topic.lower())
    
    logger.info(f"Buscando empresas para o tópico '{topic}' (caminho: {topic_path_str}) usando dados de resumo.")

    # Primeiro, aplica os filtros de metadados (setor, controle)
    data_to_analyze = {
        comp: data for comp, data in summary_data.items()
        if (not filters.get('setor') or data.get('setor', '').lower() == filters['setor'].lower()) and \
           (not filters.get('controle_acionario') or data.get('controle_acionario', '').lower() == filters['controle_acionario'].lower())
    }

    companies_with_topic = set()
    for company, details in data_to_analyze.items():
        # Navega na estrutura de tópicos da empresa para ver se o tópico pesquisado existe
        current_level = details.get("topicos_encontrados", {})
        path_keys = topic_path_str.split(',')
        found = True
        for key in path_keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                found = False
                break
        
        if found:
            companies_with_topic.add(company)

    final_companies = sorted(list(companies_with_topic))
    logger.info(f"Encontradas {len(final_companies)} empresas com o tópico preciso '{topic}' nos dados de resumo.")
    return final_companies
def get_summary_for_topic_at_company(
    company: str,
    topic: str,
    query: str,
    artifacts: dict,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable,
    filters: dict = None  # <-- Movido para o final
) -> str:
    """Ferramenta de Extração: Busca, re-ranqueia e resume um tópico para uma empresa específica."""
    plan = {"empresas": [company], "topicos": [topic], "filtros": filters or {}}
    context, _ = execute_dynamic_plan_func(query, plan, artifacts, model, cross_encoder_model, kb, company_catalog_rich, company_lookup_map, search_by_tags, expand_search_terms)
    if not context:
        return "Não foi possível encontrar detalhes específicos sobre este tópico para esta empresa e filtros."
    
    # Prompt mais detalhado para melhor qualidade do resumo
    summary_prompt = f"""
    Com base no contexto fornecido sobre a empresa {company}, resuma em detalhes as regras e o funcionamento do plano relacionadas ao tópico: '{topic}'.
    Seja direto e foque apenas nas informações relevantes para o tópico.

    CONTEXTO:
    {context}
    """
    summary = get_final_unified_answer_func(summary_prompt, context)
    return summary


def analyze_topic_thematically(
    topic: str,
    query: str,
    artifacts: dict,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable,
    filters: dict = None  # <-- Movido para o final
) -> str:
    """Ferramenta de Orquestração: Realiza uma análise temática completa de um tópico usando a busca híbrida."""
    logger.info(f"Iniciando análise temática para '{topic}' com filtros: {filters}")
    
    # Utiliza a nova função de busca híbrida
    companies_to_analyze = find_companies_by_topic(topic, artifacts, model, kb, filters)
    
    if not companies_to_analyze:
        return f"Não foram encontradas empresas com informações suficientes sobre '{topic}' para os filtros selecionados."
    
    # Limita o número de empresas para análise para evitar sobrecarga e custos
    limit = 15
    if len(companies_to_analyze) > limit:
        logger.warning(f"Muitas empresas ({len(companies_to_analyze)}) encontradas. Analisando uma amostra de {limit}.")
        companies_to_analyze = random.sample(companies_to_analyze, limit)
        
    logger.info(f"Analisando '{topic}' para {len(companies_to_analyze)} empresas...")
    company_summaries = []
    
    # Usa ThreadPool para paralelizar a coleta de resumos
    with ThreadPoolExecutor(max_workers=min(len(companies_to_analyze), 10)) as executor:
        futures = {
            executor.submit(
                get_summary_for_topic_at_company,
                company, topic, query, artifacts, model, cross_encoder_model,
                kb, company_catalog_rich, company_lookup_map, execute_dynamic_plan_func, get_final_unified_answer_func, filters
            ): company for company in companies_to_analyze
        }
        for future in futures:
            company = futures[future]
            try:
                summary_text = future.result()
                company_summaries.append({"empresa": company, "resumo_do_plano": summary_text})
            except Exception as e:
                logger.error(f"Erro ao analisar a empresa {company}: {e}")
                company_summaries.append({"empresa": company, "resumo_do_plano": f"Erro ao processar a análise."})

    synthesis_context = json.dumps(company_summaries, indent=2, ensure_ascii=False)
    
    # Usa o prompt de síntese mais detalhado da versão original
    synthesis_prompt = f"""
    Você é um consultor especialista em remuneração e planos de incentivo.
    Sua tarefa é responder à pergunta original do usuário: "{query}"
    Para isso, analise o CONTEXTO JSON abaixo, que contém resumos dos planos de várias empresas sobre o tópico '{topic}'.

    CONTEXTO:
    {synthesis_context}

    INSTRUÇÕES PARA O RELATÓRIO TEMÁTICO:
    1.  **Introdução:** Comece com um parágrafo que resume suas principais descobertas.
    2.  **Identificação de Padrões:** Analise todos os resumos e identifique de 2 a 4 "modelos" ou "padrões" comuns de mercado.
    3.  **Descrição dos Padrões:** Para cada padrão, descreva-o detalhadamente e liste as empresas que o seguem.
    4.  **Exceções e Casos Únicos:** Destaque abordagens que fogem ao padrão ou são inovadoras.
    5.  **Conclusão:** Finalize com uma breve conclusão sobre as práticas de mercado para '{topic}'.

    Seja analítico, estruturado e use Markdown para formatar sua resposta de forma clara e profissional.
    """
    final_report = get_final_unified_answer_func(synthesis_prompt, synthesis_context)
    return final_report
