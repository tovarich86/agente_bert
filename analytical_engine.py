import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy import stats
import unicodedata
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AnalyticalEngine:
    """
    Motor de análise que opera sobre dados de resumo para responder perguntas
    quantitativas, com capacidade de aplicar filtros de metadados e analisar
    estruturas de tópicos hierárquicos.
    """
    def __init__(self, summary_data: dict, knowledge_base: dict):
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        
        # --- Listas de palavras-chave para extração de filtros ---
        self.FILTER_KEYWORDS = {
            "setor": [
                "bancos", "varejo", "energia", "saude", "metalurgia", "siderurgia",
                "educacao", "transporte", "logistica", "tecnologia", "alimentos",
                "farmaceutico e higiene", "construcao civil", "telecomunicacoes",
                "intermediacao financeira", "seguradoras e corretoras",
                "extracaomineral", "textil e vestuario", "embalagens", "brinquedos e lazer",
                "hospedagem e turismo", "saneamento", "servicos agua e gas",
                "maquinas, equipamentos, veiculos e pecas", "petroleo e gas", "papel e celulose",
                "securitizacao de recebiveis", "reflorestamento", "arrendamento mercantil"
            ],
            "controle_acionario": [
                "privado", "privada", "privados", "privadas",
                "estatal", "estatais", "publico", "publica", "estrangeiro"
            ]
        }
        self.CANONICAL_MAP = {
            "privada": "Privado", "privados": "Privado", "privadas": "Privado",
            "estatais": "Estatal", "publico": "Estatal", "publica": "Estatal",
            "bancos": "Bancos", "varejo": "Comércio (Atacado e Varejo)", 
            "energia": "Energia Elétrica", "saude": "Serviços médicos", 
            "metalurgia": "Metalurgia e Siderurgia", "siderurgia": "Metalurgia e Siderurgia",
            "educacao": "Educação", "transporte": "Serviços Transporte e Logística",
            "logistica": "Serviços Transporte e Logística", "tecnologia": "Comunicação e Informática",
            "alimentos": "Alimentos", "farmaceutico e higiene": "Farmacêutico e Higiene",
            "construcao civil": "Construção Civil, Mat. Constr. e Decoração", "telecomunicacoes": "Telecomunicações",
            "intermediacao financeira": "Intermediação Financeira", "seguradoras e corretoras": "Seguradoras e Corretoras",
            "extracaomineral": "Extração Mineral", "textil e vestuario": "Têxtil e Vestuário", 
            "embalagens": "Embalagens", "brinquedos e lazer": "Brinquedos e Lazer",
            "hospedagem e turismo": "Hospedagem e Turismo", "saneamento": "Saneamento, Serv. Água e Gás",
            "servicos agua e gas": "Saneamento, Serv. Água e Gás",
            "maquinas, equipamentos, veiculos e pecas": "Máquinas, Equipamentos, Veículos e Peças",
            "petroleo e gas": "Petróleo e Gás", "papel e celulose": "Papel e Celulose",
            "securitizacao de recebiveis": "Securitização de Recebíveis", "reflorestamento": "Reflorestamento",
            "arrendamento mercantil": "Arrendamento Mercantil"
        }

        # --- Mapeamento Canônico de Indicadores e Categorias ---
        # Isso centraliza a lógica de unificação e categorização para análise de indicadores
        self.INDICATOR_CANONICAL_MAP = {
            "TSR": "TSR (Retorno Total ao Acionista)",
            "Total Shareholder Return": "TSR (Retorno Total ao Acionista)",
            "Retorno Total ao Acionista": "TSR (Retorno Total ao Acionista)",
            "TSR Absoluto": "TSR (Retorno Total ao Acionista)",
            "TSR Relativo": "TSR (Retorno Total ao Acionista)",
            "TSR versus": "TSR (Retorno Total ao Acionista)",
            "TSR comparado a": "TSR (Retorno Total ao Acionista)",

            "Lucro": "Lucro (Geral)",
            "lucro líquido": "Lucro (Geral)",
            "lucro operacional": "Lucro (Geral)",
            "lucros por ação": "Lucro (Geral)",
            "Earnings per Share": "Lucro (Geral)",
            "EPS": "Lucro (Geral)",

            "ROIC": "ROIC (Retorno sobre Capital Investido)",
            "retorno sobre investimentos": "ROIC (Retorno sobre Capital Investido)",
            "retorno sobre capital": "ROIC (Retorno sobre Capital Investido)",
            "Return on Investment": "ROIC (Retorno sobre Capital Investido)",
            "ROCE": "ROIC (Retorno sobre Capital Investido)",

            "EBITDA": "EBITDA",
            "fluxo de caixa": "Fluxo de Caixa / FCF",
            "geração de caixa": "Fluxo de Caixa / FCF",
            "Free Cash Flow": "Fluxo de Caixa / FCF",
            "FCF": "Fluxo de Caixa / FCF",
            "Receita Líquida": "Receita Líquida",
            "vendas líquidas": "Receita Líquida",
            "margem bruta": "Margem Bruta",
            "margem operacional": "Margem Operacional",
            "redução de dívida": "Redução de Dívida",
            "Dívida Líquida / EBITDA": "Dívida Líquida / EBITDA",
            "capital de giro": "Capital de Giro",
            "valor econômico agregado": "Valor Econômico Agregado",
            "CAGR": "CAGR (Taxa de Crescimento Anual Composta)",

            "qualidade": "Qualidade (Operacional)",
            "produtividade": "Produtividade (Operacional)",
            "crescimento": "Crescimento (Operacional)",
            "eficiência operacional": "Eficiência Operacional",
            "desempenho de entrega": "Desempenho de Entrega",
            "desempenho de segurança": "Desempenho de Segurança",
            "satisfação do cliente": "Satisfação do Cliente",
            "NPS": "NPS (Net Promoter Score)",
            "conclusão de aquisições": "Conclusão de Aquisições (Operacional)",
            "expansão comercial": "Expansão Comercial (Operacional)",

            "IPCA": "IPCA (Inflação)",
            "CDI": "CDI (Taxa Interbancária)",
            "Selic": "Selic (Taxa Básica de Juros)",
            "preço da ação": "Preço da Ação (Mercado)",
            "cotação das ações": "Preço da Ação (Mercado)",
            "participação de mercado": "Participação de Mercado",
            "market share": "Participação de Mercado",

            "Sustentabilidade": "ESG (Sustentabilidade)",
            "inclusão": "ESG (Inclusão/Diversidade)",
            "diversidade": "ESG (Inclusão/Diversidade)",
            "Igualdade de Gênero": "ESG (Inclusão/Diversidade)",
            "Neutralização de Emissões": "ESG (Meio Ambiente)",
            "Redução de Emissões": "ESG (Meio Ambiente)",
            "IAGEE": "ESG (Meio Ambiente)", # Assumindo um contexto de emissões ou energia
            "ICMA": "ESG (Meio Ambiente)", # Assumindo um contexto de emissões ou energia
            "objetivos de desenvolvimento sustentável": "ESG (Objetivos de Desenvolvimento Sustentável)",

            # Termos que não são indicadores de performance e devem ser tratados separadamente ou ignorados em listagens diretas
            "metas": "Outros/Genéricos",
            "critérios de desempenho": "Outros/Genéricos",
            "Metas de Performance": "Outros/Genéricos",
            "Performance Shares": "Outros/Genéricos", # É um tipo de plano, não um indicador
            "PSU": "Outros/Genéricos", # É um tipo de plano, não um indicador
            "Peer Group": "Grupos de Comparação",
            "Empresas Comparáveis": "Grupos de Comparação",
            "Companhias Comparáveis": "Grupos de Comparação"
        }

        self.INDICATOR_CATEGORIES = {
            "Financeiro": [
                "Lucro (Geral)", "EBITDA", "Fluxo de Caixa / FCF", "ROIC (Retorno sobre Capital Investido)",
                "CAGR (Taxa de Crescimento Anual Composta)", "Receita Líquida", "Margem Bruta",
                "Margem Operacional", "Redução de Dívida", "Dívida Líquida / EBITDA",
                "Capital de Giro", "Valor Econômico Agregado"
            ],
            "Mercado": [
                "TSR (Retorno Total ao Acionista)", "IPCA (Inflação)", "CDI (Taxa Interbancária)",
                "Selic (Taxa Básica de Juros)", "Preço da Ação (Mercado)", "Participação de Mercado"
            ],
            "Operacional": [
                "Qualidade (Operacional)", "Produtividade (Operacional)", "Crescimento (Operacional)",
                "Eficiência Operacional", "Desempenho de Entrega", "Desempenho de Segurança",
                "Satisfação do Cliente", "NPS (Net Promoter Score)", "Conclusão de Aquisições (Operacional)",
                "Expansão Comercial (Operacional)"
            ],
            "ESG": [
                "ESG (Sustentabilidade)", "ESG (Inclusão/Diversidade)", "ESG (Meio Ambiente)",
                "ESG (Objetivos de Desenvolvimento Sustentável)"
            ],
            "Outros/Genéricos": ["Outros/Genéricos"], # Para agrupar termos que não são indicadores específicos
            "Grupos de Comparação": ["Grupos de Comparação"]
        }
        
        # --- Roteador Declarativo (Completo e com todas as funções implementadas) ---
        self.intent_rules = [
            # Vesting: Adicionado "carência", "tempo", "duração"
            (lambda q: 'vesting' in q and ('periodo' in q or 'prazo' in q or 'medio' in q or 'media' in q or 'carencia' in q or 'tempo' in q or 'duracao' in q), self._analyze_vesting_period),
            
            # Lock-up: Adicionado "restrição de venda"
            (lambda q: ('lockup' in q or 'lock-up' in q or 'restricao de venda' in q) and ('periodo' in q or 'prazo' in q or 'medio' in q or 'media' in q), self._analyze_lockup_period),

            # Diluição: Adicionado "percentual", "estatisticas"
            (lambda q: 'diluicao' in q and ('media' in q or 'percentual' in q or 'estatisticas' in q), self._analyze_dilution),

            # Desconto/Strike: A regra original já era boa.
            (lambda q: 'desconto' in q and ('preco de exercicio' in q or 'strike' in q), self._analyze_strike_discount),
            
            # TSR: A regra original já era boa.
            (lambda q: 'tsr' in q, self._analyze_tsr),
            
            # Malus/Clawback: Adicionado "lista", "quais" para forçar listagem.
            (lambda q: ('malus' in q or 'clawback' in q) and ('lista' in q or 'quais' in q), self._analyze_malus_clawback),
            
            # Dividendos: Adicionado "lista", "quais"
            (lambda q: 'dividendos' in q and 'carencia' in q and ('lista' in q or 'quais' in q), self._analyze_dividends_during_vesting),
            
            # Elegibilidade/Membros: A regra original já era boa.
            (lambda q: 'membros do plano' in q or 'elegiveis' in q or 'quem sao os membros' in q, self._analyze_plan_members),
            
            # Conselho: A regra original já era boa.
            (lambda q: 'conselho de administracao' in q and ('elegivel' in q or 'aprovador' in q), self._count_plans_for_board),
            
            # Metas/Indicadores: A regra original já era boa.
            (lambda q: 'metas mais comuns' in q or 'indicadores de desempenho' in q or 'metas de desempenho' in q or 'metas de performance' in q or 'indicadores de performance' in q or 'quais os indicadores mais comuns' in q, self._analyze_common_goals),
            
            # Regra para tipos de plano (agora separada e com sua própria vírgula)
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            
            # Fallback (sempre por último)
            (lambda q: True, self._find_companies_by_general_topic),
        ]

    def _collect_leaf_aliases_recursive(self, node: dict or list, collected_aliases: list):
        """
        Percorre qualquer estrutura baseada no modelo dado e coleta todos os aliases.
        Se 'node' for uma lista, ela itera sobre seus elementos.
        Se 'node' for um dicionário, ela verifica as chaves '_aliases' e 'subtopicos'
        para continuar a recursão ou adicionar aliases.
        """
        if isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    collected_aliases.append(item)
                elif isinstance(item, (dict, list)):
                    self._collect_leaf_aliases_recursive(item, collected_aliases)
        elif isinstance(node, dict):
            if "_aliases" in node and isinstance(node["_aliases"], list):
                collected_aliases.extend(node["_aliases"])
            for k, v in node.items():
                if k != "_aliases" and isinstance(v, (dict, list)):
                    self._collect_leaf_aliases_recursive(v, collected_aliases)
                elif isinstance(v, list) and k != "_aliases": # Handle lists that are not _aliases but contain values
                    for item in v:
                        if isinstance(item, str):
                            collected_aliases.append(item)


    def _normalize_text(self, text: str) -> str:
        """Normaliza o texto para minúsculas e remove acentos."""
        nfkd_form = unicodedata.normalize('NFKD', text.lower())
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _extract_filters(self, normalized_query: str) -> dict:
        """Extrai filtros da pergunta com base em palavras-chave."""
        filters = {}
        for filter_type, keywords in self.FILTER_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(self._normalize_text(keyword)) + r'\b', normalized_query):
                    canonical_term = self.CANONICAL_MAP.get(keyword, keyword.capitalize())
                    filters[filter_type] = canonical_term
                    break
        if filters:
            logging.info(f"Filtros extraídos da pergunta: {filters}")
        return filters

    def _apply_filters_to_data(self, filters: dict) -> dict:
        """Aplica um dicionário de filtros aos dados principais."""
        if not filters:
            return self.data
        filtered_data = {
            comp: data for comp, data in self.data.items()
            if ('setor' not in filters or self._normalize_text(data.get('setor', '')) == self._normalize_text(filters['setor'])) and \
               ('controle_acionario' not in filters or data.get('controle_acionario', '').lower() == filters['controle_acionario'].lower())
        }
        logging.info(f"{len(filtered_data)} empresas correspondem aos filtros aplicados.")
        return filtered_data

    def answer_query(self, query: str, filters: dict | None = None) -> tuple:
        """
        Responde a uma consulta quantitativa.
        
        Args:
            query (str): A pergunta do usuário.
            filters (dict | None, optional): Um dicionário de filtros pré-selecionados
                                            (ex: da interface). Se for None, os filtros
                                            serão extraídos do texto da query.
        
        Returns:
            tuple: Uma tupla contendo o texto do relatório e um DataFrame/dicionário.
        """
        normalized_query = self._normalize_text(query)
        
        # Prioriza os filtros passados como argumento (da UI).
        # Se nenhum for passado, usa a extração da query como fallback.
        final_filters = filters if filters is not None else self._extract_filters(normalized_query)
        
        for intent_checker_func, analysis_func in self.intent_rules:
            if intent_checker_func(normalized_query):
                logging.info(f"Intenção detectada. Executando: {analysis_func.__name__}")
                return analysis_func(normalized_query, final_filters)
                
        return "Não consegui identificar uma intenção clara na sua pergunta.", None
    
    # --- Funções de Análise Detalhadas e Completas ---

    def _analyze_vesting_period(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        periods = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_vesting' in facts and facts['periodo_vesting'].get('presente', False):
                valor = facts['periodo_vesting'].get('valor')
                if valor is not None and valor > 0:
                    periods.append((company, valor))
        if not periods:
            return "Nenhuma informação de vesting encontrada para os filtros selecionados.", None
        
        vesting_values = np.array([item[1] for item in periods])
        mode_result = stats.mode(vesting_values, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f} anos" for m in modes]) if len(modes) > 0 else "N/A"
        
        report_text = "### Análise de Período de Vesting\n"
        report_text += f"- **Total de Empresas com Dados:** {len(vesting_values)}\n"
        report_text += f"- **Vesting Médio:** {np.mean(vesting_values):.2f} anos\n"
        report_text += f"- **Desvio Padrão:** {np.std(vesting_values):.2f} anos\n"
        report_text += f"- **Mediana:** {np.median(vesting_values):.2f} anos\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(vesting_values):.2f} / {np.max(vesting_values):.2f} anos\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(periods, columns=["Empresa", "Período de Vesting (Anos)"])
        return report_text, df.sort_values(by="Período de Vesting (Anos)", ascending=False).reset_index(drop=True)

    def _analyze_lockup_period(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        periods = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_lockup' in facts and facts['periodo_lockup'].get('presente', False):
                valor = facts['periodo_lockup'].get('valor')
                if valor is not None and valor > 0:
                    periods.append((company, valor))
        if not periods:
            return "Nenhuma informação de lock-up encontrada para os filtros selecionados.", None

        lockup_values = np.array([item[1] for item in periods])
        mode_result = stats.mode(lockup_values, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f} anos" for m in modes]) if len(modes) > 0 else "N/A"

        report_text = "### Análise de Período de Lock-up\n"
        report_text += f"- **Total de Empresas com Dados:** {len(lockup_values)}\n"
        report_text += f"- **Lock-up Médio:** {np.mean(lockup_values):.2f} anos\n"
        report_text += f"- **Mediana:** {np.median(lockup_values):.2f} anos\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(lockup_values):.2f} / {np.max(lockup_values):.2f} anos\n"
        report_text += f"- **Moda(s):** {mode_str}\n"

        df = pd.DataFrame(periods, columns=["Empresa", "Período de Lock-up (Anos)"])
        return report_text, df.sort_values(by="Período de Lock-up (Anos)", ascending=False).reset_index(drop=True)

    def _analyze_dilution(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        diluicao_percentual = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'diluicao_maxima_percentual' in facts and facts['diluicao_maxima_percentual'].get('presente', False):
                valor = facts['diluicao_maxima_percentual'].get('valor')
                if valor is not None:
                    diluicao_percentual.append((company, valor * 100))
        if not diluicao_percentual:
            return "Nenhuma informação de diluição encontrada para os filtros selecionados.", None

        percents = np.array([item[1] for item in diluicao_percentual])
        mode_result = stats.mode(percents, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"
        
        report_text = "### Análise de Diluição Máxima Percentual\n"
        report_text += f"- **Total de Empresas com Dados:** {len(percents)}\n"
        report_text += f"- **Média:** {np.mean(percents):.2f}%\n"
        report_text += f"- **Mediana:** {np.median(percents):.2f}%\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(percents):.2f}% / {np.max(percents):.2f}%\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df_percent = pd.DataFrame(diluicao_percentual, columns=["Empresa", "Diluição Máxima (%)"])
        return report_text, df_percent.sort_values(by="Diluição Máxima (%)", ascending=False).reset_index(drop=True)

    def _analyze_strike_discount(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies_and_discounts = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico')
                if valor_numerico is not None:
                    companies_and_discounts.append((company, valor_numerico * 100))
        if not companies_and_discounts:
            return "Nenhuma empresa com desconto no preço de exercício foi encontrada para os filtros selecionados.", None
        
        discounts = np.array([item[1] for item in companies_and_discounts])
        mode_result = stats.mode(discounts, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"
        
        report_text = "### Análise de Desconto no Preço de Exercício\n"
        report_text += f"- **Total de Empresas com Desconto:** {len(discounts)}\n"
        report_text += f"- **Desconto Médio:** {np.mean(discounts):.2f}%\n"
        report_text += f"- **Desvio Padrão:** {np.std(discounts):.2f}%\n"
        report_text += f"- **Mediana:** {np.median(discounts):.2f}%\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(discounts):.2f}% / {np.max(discounts):.2f}%\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(companies_and_discounts, columns=["Empresa", "Desconto Aplicado (%)"])
        return report_text, df.sort_values(by="Desconto Aplicado (%)", ascending=False).reset_index(drop=True)

    def _analyze_tsr(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        results = defaultdict(list)
        tsr_type_filter = 'qualquer'
        if 'relativo' in normalized_query and 'absoluto' not in normalized_query:
            tsr_type_filter = 'relativo'
        elif 'absoluto' in normalized_query and 'relativo' not in normalized_query:
            tsr_type_filter = 'absoluto'

        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            tipos = facts.get('tsr', {}).get('tipos', [])
            has_tsr_absoluto = 'Absoluto' in tipos
            has_tsr_relativo = 'Relativo' in tipos
            if has_tsr_absoluto: results['absoluto'].append(company)
            if has_tsr_relativo: results['relativo'].append(company)
            if has_tsr_absoluto or has_tsr_relativo: results['qualquer'].append(company)

        target_companies = results.get(tsr_type_filter, [])
        if not target_companies:
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type_filter}' para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type_filter.upper()}** para os filtros aplicados."
        
        df = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type_filter.upper()})"])
        return report_text, df

    def _analyze_malus_clawback(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'malus_clawback_presente' in facts and facts['malus_clawback_presente'].get('presente', False):
                companies.append(company)
        if not companies:
            return "Nenhuma empresa com cláusulas de Malus ou Clawback foi encontrada para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(companies)}** empresas com cláusulas de **Malus ou Clawback** para os filtros aplicados."
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Malus/Clawback"])
        return report_text, df

    def _analyze_dividends_during_vesting(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'dividendos_durante_carencia' in facts and facts['dividendos_durante_carencia'].get('presente', False):
                companies.append(company)
        if not companies:
            return "Nenhuma empresa que paga dividendos durante a carência foi encontrada para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(companies)}** empresas que distribuem dividendos durante a **carência/vesting** para os filtros aplicados."
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Dividendos Durante Carência"])
        return report_text, df

    def _analyze_plan_members(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        member_role_counts = defaultdict(int)
        company_member_details = []
        for company, details in data_to_analyze.items():
            topics = details.get("topicos_encontrados", {})
            elegibility_section = topics.get("ParticipantesCondicoes", {}).get("Elegibilidade", [])
            
            if elegibility_section: # Check if elegibility_section exists and is not empty
                company_member_details.append({"Empresa": company, "Funções Elegíveis": ", ".join(elegibility_section)})
                for role in elegibility_section:
                    member_role_counts[role] += 1
        
        if not member_role_counts:
            return "Nenhuma informação sobre membros elegíveis foi encontrada para os filtros selecionados.", None
        
        report_text = "### Análise de Membros Elegíveis ao Plano\n**Contagem de Empresas por Tipo de Membro:**\n"
        df_counts_data = []
        for role, count in sorted(member_role_counts.items(), key=lambda item: item[1], reverse=True):
            report_text += f"- **{role}:** {count} empresas\n"
            df_counts_data.append({"Tipo de Membro Elegível": role, "Nº de Empresas": count})
        
        dfs_to_return = {
            'Contagem por Tipo de Membro': pd.DataFrame(df_counts_data),
            'Detalhes por Empresa': pd.DataFrame(company_member_details).sort_values(by="Empresa").reset_index(drop=True)
        }
        return report_text, dfs_to_return

    def _count_plans_for_board(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = []
        for company, details in data_to_analyze.items():
            topics = details.get("topicos_encontrados", {})
            governance_section = topics.get("GovernancaRisco", {})
            if "OrgaoDeliberativo" in governance_section:
                deliberative_organs = governance_section["OrgaoDeliberativo"]
                normalized_deliberative_organs = [self._normalize_text(org) for org in deliberative_organs]
                if "conselho de administracao" in normalized_deliberative_organs:
                    companies.append(company)
        if not companies:
            return "Nenhuma empresa com menção ao Conselho de Administração como elegível/aprovador foi encontrada para os filtros selecionados.", None
        
        report_text = f"**{len(companies)}** empresas com menção ao **Conselho de Administração** como elegível ou aprovador de planos foram encontradas para os filtros aplicados."
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Menção ao Conselho de Administração"])
        return report_text, df

    def _analyze_common_goals(self, normalized_query: str, filters: dict) -> tuple:
        """
        Analisa e contabiliza os indicadores de performance mais comuns de forma precisa,
        contando apenas os tópicos já extraídos para cada empresa e categorizando-os
        corretamente.
        """
        data_to_analyze = self._apply_filters_to_data(filters)
        
        # Dicionário para armazenar, para cada indicador canônico, o CONJUNTO de empresas que o mencionam.
        # Usar um set garante que cada empresa seja contada apenas uma vez por indicador.
        canonical_indicator_companies = defaultdict(set)
        
        # Itera sobre os dados já filtrados
        for company, details in data_to_analyze.items():
            # Navega até a seção de Indicadores de Performance já extraída para a empresa
            performance_section = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
            if not performance_section:
                continue

            # Coleta todos os aliases/tópicos folha DENTRO da seção de performance desta empresa
            company_leaf_aliases = []
            self._collect_leaf_aliases_recursive(performance_section, company_leaf_aliases)

            # Para cada alias encontrado na empresa, normaliza-o para seu nome canônico
            # e adiciona a empresa ao conjunto daquele indicador.
            for alias in set(company_leaf_aliases): # Usa set() para evitar reprocessamento do mesmo alias
                canonical_name = self.INDICATOR_CANONICAL_MAP.get(alias, alias)
                
                # IGNORA termos que não são indicadores reais para a contagem
                if canonical_name not in ["Outros/Genéricos", "Grupos de Comparação"]:
                    canonical_indicator_companies[canonical_name].add(company)

        if not canonical_indicator_companies:
            return "Nenhum indicador de performance específico encontrado para os filtros selecionados.", None

        # Agora, a contagem é simplesmente o tamanho do conjunto de empresas para cada indicador
        canonical_counts = {
            indicator: len(companies)
            for indicator, companies in canonical_indicator_companies.items()
        }

        # Categoriza os indicadores para o relatório final
        categorized_indicators = defaultdict(list)
        for indicator, count in canonical_counts.items():
            found_category = "Outros (Não Categorizados)" # Categoria padrão
            for category, indicators_in_category in self.INDICATOR_CATEGORIES.items():
                if indicator in indicators_in_category:
                    found_category = category
                    break
            categorized_indicators[found_category].append((indicator, count))

        # Monta o relatório e o DataFrame
        report_text = "### Principais Indicadores de Performance\n\n"
        df_data = []
        ordered_categories = ["Financeiro", "Mercado", "Operacional", "ESG", "Outros (Não Categorizados)"]
        
        for category in ordered_categories:
            if category in categorized_indicators:
                report_text += f"#### **{category}**\n"
                sorted_indicators = sorted(categorized_indicators[category], key=lambda item: item[1], reverse=True)
                for indicator, count in sorted_indicators:
                    report_text += f"- **{indicator}:** {count} empresas\n"
                    df_data.append({"Indicador": indicator, "Categoria": category, "Nº de Empresas": count})
                report_text += "\n"

        df = pd.DataFrame(df_data).sort_values(by=["Categoria", "Nº de Empresas"], ascending=[True, False]).reset_index(drop=True)
        return report_text, df
        
    def _analyze_common_plan_types(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        plan_type_counts = defaultdict(int)
        for details in data_to_analyze.values():
            plan_topics = details.get("topicos_encontrados", {}).get("TiposDePlano", {})
            
            # This part needs to correctly extract the *keys* from TiposDePlano,
            # which represent the plan types (e.g., AcoesRestritas, OpcoesDeCompra)
            # and count them.
            for plan_type_raw in plan_topics.keys():
                if plan_type_raw not in ['_aliases', 'subtopicos']: # Exclude metadata keys
                    plan_type_counts[plan_type_raw.replace('_', ' ')] += 1

        if not plan_type_counts:
            return "Nenhum tipo de plano encontrado para os filtros selecionados.", None
            
        report_text = "### Tipos de Planos Mais Comuns\n"
        df_data = [{"Tipo de Plano": k, "Nº de Empresas": v} for k, v in sorted(plan_type_counts.items(), key=lambda item: item[1], reverse=True)]
        for item in df_data:
            report_text += f"- **{item['Tipo de Plano'].capitalize()}:** {item['Nº de Empresas']} empresas\n"
        return report_text, pd.DataFrame(df_data)

    # --- Funções de Busca Hierárquica (Fallback) ---

    def _recursive_flat_map_builder(self, sub_dict: dict, section: str, flat_map: dict):
        """Função auxiliar recursiva para construir o mapa plano de aliases."""
        for topic_name_raw, data in sub_dict.items():
            if not isinstance(data, dict):
                continue
            
            topic_name_formatted = topic_name_raw.replace('_', ' ')
            details = (section, topic_name_formatted, topic_name_raw)

            flat_map[self._normalize_text(topic_name_formatted)] = details
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details
            
            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)
    
    def _kb_flat_map(self) -> dict:
        """Cria um mapa plano de alias -> (seção, nome_formatado, nome_bruto)."""
        if hasattr(self, '_kb_flat_map_cache'):
            return self._kb_flat_map_cache
        
        flat_map = {}
        for section, data in self.kb.items():
            if not isinstance(data, dict):
                continue

            section_name_formatted = section.replace('_', ' ')
            details = (section, section_name_formatted, section)
            
            flat_map[self._normalize_text(section_name_formatted)] = details
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details

            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)
        
        self._kb_flat_map_cache = flat_map
        return flat_map

    def _find_companies_by_general_topic(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        flat_map = self._kb_flat_map()
        found_topic_details = None
        
        # Busca pelo alias mais longo que corresponde à pergunta
        for alias in sorted(flat_map.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', normalized_query):
                found_topic_details = flat_map[alias]
                break
        
        if not found_topic_details:
            return "Não foi possível identificar um tópico técnico conhecido na sua pergunta para realizar a busca.", None

        section, topic_name_formatted, topic_name_raw = found_topic_details
        
        # Lógica para encontrar empresas que mencionam o tópico (incluindo sub-tópicos)
        companies = []
        for name, details in data_to_analyze.items():
            if section in details.get("topicos_encontrados", {}):
                # Using deque for a BFS-like traversal
                queue = deque([details["topicos_encontrados"][section]])
                found_in_company = False
                while queue:
                    current_node = queue.popleft()
                    if isinstance(current_node, dict):
                        for k, v in current_node.items():
                            if k == topic_name_raw:
                                found_in_company = True
                                break
                            if isinstance(v, dict) and 'subtopicos' in v and v['subtopicos']:
                                queue.append(v['subtopicos'])
                            elif isinstance(v, list):
                                if topic_name_raw in current_node: # Check if the raw topic name is directly in a list
                                    found_in_company = True
                                    break
                    elif isinstance(current_node, list): # Added this check for lists not under a specific key
                        if topic_name_raw in current_node:
                            found_in_company = True
                            break
                    if found_in_company:
                        break # Exit inner loop once found in this company
                if found_in_company:
                    companies.append(name)

        if not companies:
            return f"Nenhuma empresa encontrada com o tópico '{topic_name_formatted}' para os filtros aplicados.", None

        report_text = f"Encontradas **{len(companies)}** empresas com o tópico: **{topic_name_formatted.capitalize()}**"
        df = pd.DataFrame(sorted(companies), columns=[f"Empresas com {topic_name_formatted.capitalize()}"])
        return report_text, df
