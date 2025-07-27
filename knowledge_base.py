# =================================================================================
# DICIONÁRIO HIERÁRQUICO UNIFICADO (v6.1 - Estrutura de Aliases Completa)
#
# DESCRIÇÃO:
# Versão com a estrutura de aliases aplicada a todas as categorias principais,
# permitindo buscas genéricas e de alto nível em todo o dicionário.
# =================================================================================
DICIONARIO_UNIFICADO_HIERARQUICO = {
    "FormularioReferencia_Item_8_4": {
        "a_TermosGerais": {"aliases": ["termos e condições gerais", "objetivos do plano", "elegíveis", "principais regras"], "subtopicos": {}},
        "b_Aprovacao": {"aliases": ["data de aprovação", "órgão responsável", "assembleia geral"], "subtopicos": {}},
        "c_MaximoAcoes": {"aliases": ["número máximo de ações abrangidas", "diluição máxima"], "subtopicos": {}},
        "d_MaximoOpcoes": {"aliases": ["número máximo de opções a serem outorgadas", "limite de opções"], "subtopicos": {}},
        "e_CondicoesAquisicao": {"aliases": ["condições de aquisição de ações", "metas de desempenho", "tempo de serviço"], "subtopicos": {}},
        "f_CriteriosPreco": {"aliases": ["critérios para fixação do preço de aquisição", "preço de exercício", "preço fixo previamente estabelecido"], "subtopicos": {}},
        "g_CriteriosPrazo": {"aliases": ["critérios para fixação do prazo de aquisição", "prazo de exercício"], "subtopicos": {}},
        "h_FormaLiquidacao": {"aliases": ["forma de liquidação", "pagamento em dinheiro", "entrega física das ações", "entrega de ações"], "subtopicos": {}},
        "i_RestricoesTransferencia": {"aliases": ["restrições à transferência", "períodos de bloqueio", "lockup", "bloqueio"], "subtopicos": {}},
        "j_SuspensaoExtincao": {"aliases": ["suspensão, alteração ou extinção do plano", "mudanças nas políticas"], "subtopicos": {}},
        "k_EfeitosSaida": {"aliases": ["efeitos da saída do administrador", "regras de desligamento", "aposentadoria", "demissão"], "subtopicos": {}},
    },
    "TiposDePlano": {
        "AcoesRestritas": {
            "aliases": ["Ações Restritas", "Restricted Shares", "RSU"],
            "subtopicos": {
                "PerformanceShares": {
                    "aliases": ["Performance Shares", "PSU", "Ações de Performance"],
                    "subtopicos": {}
                }
            }
        },
        "OpcoesDeCompra": {
            "aliases": ["Opções de Compra", "Stock Options", "ESOP", "SOP"],
            "subtopicos": {}
        },
        "PlanoCompraAcoes_ESPP": {
            "aliases": ["Plano de Compra de Ações", "Employee Stock Purchase Plan", "ESPP"],
            "subtopicos": {
                "Matching_Coinvestimento": {
                    "aliases": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching"],
                    "subtopicos": {}
                }
            }
        },
        "AcoesFantasmas": {
            "aliases": ["Ações Fantasmas", "Phantom Shares", "Ações Virtuais"],
            "subtopicos": {}
        },
        "OpcoesFantasmas_SAR": {
            "aliases": ["Opções Fantasmas", "Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
            "subtopicos": {}
        },
        "BonusRetencaoDiferido": {
            "aliases": ["Bônus de Retenção", "Bônus de Permanência", "Staying Bonus", "Retention Bonus", "Deferred Bonus"],
            "subtopicos": {}
        }
    },
    "MecanicasCicloDeVida": {
        "Outorga": {"aliases": ["Outorga", "Concessão", "Grant", "Grant Date"], "subtopicos": {}},
        "Vesting": {"aliases": ["Vesting", "Período de Carência", "Aquisição de Direitos", "cronograma de vesting", "Vesting Gradual"], "subtopicos": {}},
        "VestingCliff": {"aliases": ["Cliff", "Cliff Period", "Período de Cliff", "Carência Inicial"], "subtopicos": {}},
        "VestingAcelerado": {"aliases": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação do vesting"], "subtopicos": {}},
        "VestingTranche": {"aliases": ["Tranche", "Lote", "Parcela do Vesting", 'Parcela', "Aniversário"], "subtopicos": {}},
        "PrecoExercicio": {"aliases": ["Preço de Exercício", "Strike", "Strike Price"], "subtopicos": {}},
        "PrecoDesconto": {"aliases": ["Desconto de", "preço com desconto", "desconto sobre o preço"], "subtopicos": {}},
        "CicloExercicio": {"aliases": ["Exercício", "Período de Exercício", "pagamento", "liquidação", "vencimento", "expiração"], "subtopicos": {}},
        "Lockup": {"aliases": ["Lockup", "Período de Lockup", "Restrição de Venda"], "subtopicos": {}},
    },
    "GovernancaRisco": {
        "DocumentosPlano": {"aliases": ["Regulamento", "Regulamento do Plano", "Contrato de Adesão", "Termo de Outorga"], "subtopicos": {}},
        "OrgaoDeliberativo": {"aliases": ["Comitê de Remuneração", "Comitê de Pessoas", "Deliberação do Conselho", "Conselho de Administração"], "subtopicos": {}},
        "MalusClawback": {"aliases": ["Malus", "Clawback", "Cláusula de Recuperação", "Forfeiture"], "subtopicos": {}},
        "Diluicao": {"aliases": ["Diluição", "Dilution", "Capital Social", "Fator de Diluição"], "subtopicos": {}},
        "NaoConcorrencia": {
            "aliases": ["Non-Compete", "Não-Competição", "Garden-leave", "obrigação de não concorrer", "proibição de competição"],
            "subtopicos": {}
        }
    },
    "ParticipantesCondicoes": {
        "Elegibilidade": {"aliases": ["Participantes", "Beneficiários", "Elegíveis", "Empregados", "Administradores", "Colaboradores", "Executivos", "Diretores", "Gerentes", "Conselheiros"], "subtopicos": {}},
        "CondicaoSaida": {"aliases": ["Desligamento", "Saída", "Término do Contrato", "Rescisão", "Demissão", "Good Leaver", "Bad Leaver"], "subtopicos": {}},
        "CasosEspeciais": {"aliases": ["Aposentadoria", "Morte", "Invalidez", "Afastamento"], "subtopicos": {}},
    },
    "IndicadoresPerformance": {
        "ConceitoGeral_Performance": {
            "aliases": ["Plano de Desempenho", "Metas de Performance", "critérios de desempenho", "metas"],
            "subtopicos": {
                "Financeiro": {
                    "aliases": [
                        "ROIC", "EBITDA", "LAIR", "Lucro", "CAGR", "Receita Líquida",
                        "fluxo de caixa", "geração de caixa", "Free Cash Flow", "FCF",
                        "lucros por ação", "Earnings per Share", "EPS", "redução de dívida",
                        "Dívida Líquida / EBITDA", "capital de giro", "retorno sobre investimentos",
                        "retorno sobre capital", "Return on Investment", "ROCE",
                        "margem bruta", "margem operacional", "lucro líquido",
                        "lucro operacional", "receita operacional", "vendas líquidas",
                        "valor econômico agregado", "custo de capital", "WACC",
                        "Weighted Average Capital Cost", "retorno sobre ativo",
                        "retorno sobre ativo líquido", "rotatividade de ativos líquidos",
                        "rotatividade do estoque", "despesas de capital", "dívida financeira bruta",
                        "receita operacional líquida", "lucros por ação diluídos",
                        "lucros por ação básicos", "rentabilidade", "Enterprise Value", "EV",
                        "Valor Teórico da Companhia", "Valor Teórico Unitário da Ação",
                        "Economic Value Added", "EVA", "NOPAT", "Net Operating Profit After Tax",
                        "Capital Total Investido", "CAGR EBITDA per Share", "Equity Value"
                    ],
                    "subtopicos": {}
                },
                "Mercado": {
                    "aliases": ["CDI", "IPCA", "Selic", "preço da ação", "cotação das ações", "participação de mercado", "market share"],
                    "subtopicos": {}
                },
                "TSR": {
                    "aliases": ["TSR", "Total Shareholder Return", "Retorno Total ao Acionista"],
                    "subtopicos": {
                        "TSR_Absoluto": {"aliases": ["TSR Absoluto"], "subtopicos": {}},
                        "TSR_Relativo": {"aliases": ["TSR Relativo", "Relative TSR", "TSR versus", "TSR comparado a"], "subtopicos": {}}
                    }
                },
                "ESG": {
                    "aliases": ["Metas ESG", "ESG", "Sustentabilidade", "Neutralização de Emissões", "Redução de Emissões", "Igualdade de Gênero", "diversidade", "inclusão", "objetivos de desenvolvimento sustentável", "IAGEE", "ICMA"],
                    "subtopicos": {}
                },
                "Operacional": {
                    "aliases": ["produtividade", "eficiência operacional", "desempenho de entrega", "desempenho de segurança", "qualidade", "satisfação do cliente", "NPS", "conclusão de aquisições", "expansão comercial", "crescimento"],
                    "subtopicos": {}
                }
            }
        },
        "GrupoDeComparacao": {"aliases": ["Peer Group", "Empresas Comparáveis", "Companhias Comparáveis"], "subtopicos": {}}
    },
    "EventosFinanceiros": {
        "EventosCorporativos": {"aliases": ["grupamento", "desdobramento", "cisão", "fusão", "incorporação", "bonificação"], "subtopicos": {}},
        "MudancaDeControle": {"aliases": ["Mudança de Controle", "Change of Control", "Transferência de Controle"], "subtopicos": {}},
        "DividendosProventos": {"aliases": ["Dividendos", "JCP", "Juros sobre capital próprio", "dividend equivalent", "proventos"], "subtopicos": {}},
        "EventosDeLiquidez": {
            "aliases": ["Evento de Liquidez", "liquidação antecipada", "saída da companhia", "transação de controle", "reorganização societária", "desinvestimento", "deslistagem", "Operação Relevante"],
            "subtopicos": {
                "IPO_OPI": {"aliases": ["IPO", "Oferta Pública Inicial", "Oferta Publica Inicial", "abertura do capital"], "subtopicos": {}},
                "AlienacaoControle": {"aliases": ["Alienação de Controle", "alienação de mais de 50% de ações ordinárias", "venda de controle", "transferência de controle acionário", "venda ou permuta de Ações"], "subtopicos": {}},
                "FusaoAquisicaoVenda": {"aliases": ["Fusão", "Aquisição", "Incorporação", "Venda da Companhia", "venda, locação, arrendamento, cessão, licenciamento, transferência ou qualquer outra forma de disposição da totalidade ou de parte substancial dos ativos"], "subtopicos": {}},
                "InvestimentoRelevante": {"aliases": ["investimento primário de terceiros", "aumento de capital", "Capitalização da Companhia"], "subtopicos": {}}
            }
        }
    },
    "AspectosFiscaisContabeis": {
        "TributacaoEncargos": {"aliases": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"], "subtopicos": {}},
        "NormasContabeis": {"aliases": ["IFRS 2", "CPC 10", "Valor Justo", "Fair Value", "Black-Scholes", "Despesa Contábil", "Volatilidade"], "subtopicos": {}},
    }
}
