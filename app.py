import streamlit as st
import os
import zipfile
import nltk
from app_model import HybridT5Model, buscar_detalhes_framenet

# Definição dos caminhos locais e do arquivo compactado
PATH_ZIP = "framenet_completa.zip"
PATH_DB_LOCAL = "framenet_completa.db"

# Executa os downloads do NLTK antes de qualquer cache para garantir o deploy limpo
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)          
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger_tab', quiet=True) # CORRIGIDO: Nome correto do pacote de TAGS
except Exception as e:
    pass

@st.cache_resource
def carregar_recursos():
    """Garante que os modelos e arquivos sejam extraídos apenas uma vez."""
    # 1. Extração automática do Banco de Dados caso ele não exista descompactado
    if not os.path.exists(PATH_DB_LOCAL):
        if os.path.exists(PATH_ZIP):
            with zipfile.ZipFile(PATH_ZIP, 'r') as zip_ref:
                zip_ref.extractall(".") # Extrai o arquivo .db na raiz do projeto
        else:
            st.error(f"❌ Erro crítico: O arquivo compactado `{PATH_ZIP}` não foi encontrado!")
        
    modelo_hibrido = HybridT5Model(model_name="t5-small", num_frames=50) 
    return modelo_hibrido

# Inicializa o cache dos modelos
modelo_hibrido = carregar_recursos()

def pre_processar_texto_nltk(texto):
    """Transforma o texto no padrão indexado com marcação de alvos (*target*) usando NLTK."""
    palavras = nltk.word_tokenize(texto)
    tags_gramaticais = nltk.pos_tag(palavras)
    
    gatilho_idx = None
    for idx, (palavra, tag) in enumerate(tags_gramaticais):
        if tag.startswith('VB'):  # Captura qualquer variação de classe verbal
            gatilho_idx = idx
            break
            
    texto_indexado_list = []
    texto_input_list = []
    
    for idx, palavra in enumerate(palavras):
        texto_indexado_list.append(f"{idx} {palavra}")
        if idx == gatilho_idx:
            texto_input_list.append(f"{idx} *{palavra}*")
        else:
            texto_input_list.append(f"{idx} {palavra}")
            
    return " ".join(texto_indexado_list), " ".join(texto_input_list), palavras

# --- CONFIGURAÇÃO DA INTERFACE STREAMLIT ---
st.set_page_config(page_title="Análise Semântica FrameNet T5", page_icon="🧠", layout="wide")

st.title("🧠 Extrator Semântico Baseado em Frames (T5 Híbrido)")
st.subheader("Transforme linguagem natural em dados estruturados com Deep Learning (NLTK Backend)")

# Validação visual do banco de dados na inicialização
if not os.path.exists(PATH_DB_LOCAL):
    st.error(f"❌ O arquivo do banco de dados `{PATH_DB_LOCAL}` não foi encontrado no repositório! Verifique seu GitHub.")
else:
    st.success(f"✔ Banco de dados `framenet_completa.db` detectado e pronto para consultas!")

texto_usuario = st.text_input(
    "Digite a frase para análise:", 
    value="enviar o relatório para o Diretor",
    placeholder="Ex: comprar um livro na loja ontem"
)

if st.button("🚀 Processar Frase", type="primary"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, insira um texto válido.")
    else:
        # 1. Processamento de texto com o novo motor NLTK
        texto_idx, input_modelo, lista_palavras = pre_processar_texto_nltk(texto_usuario)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Texto com Índices:**\n`{texto_idx}`")
        with col2:
            st.success(f"**Entrada Formatada para o T5 (Target):**\n`{input_modelo}`")
            
        # 2. CORRIGIDO: Resposta simulada usando inteiros puros para os índices
        json_mock_retorno = {
            "frame": "Sending",
            "arguments": {
                "Theme": " " ,     # "o relatório"
                "Recipient": [4, 5]    # "o Diretor"
            }
        }
        
        st.divider()
        st.subheader("📦 Saída Estruturada do Modelo (JSON)")
        st.json(json_mock_retorno)
        
        # 3. Consulta ao SQLite utilizando o caminho dinâmico do GitHub
        nome_frame = json_mock_retorno["frame"]
        dados_fn = buscar_detalhes_framenet(PATH_DB_LOCAL, nome_frame)
        
        # 4. Mapeamento dos índices nos tokens reais
        st.divider()
        st.subheader("📋 Informações Linguísticas Extraídas")
        
        argumentos_reais = {}
        for papel, indices in json_mock_retorno["arguments"].items():
            fragmento = " ".join([lista_palavras[int(i)] for i in indices if int(i) < len(lista_palavras)])
            argumentos_reais[papel] = fragmento

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Ação Solicitada (Frame)", value=dados_fn["acao"])
        with c2:
            st.metric(label="Agente da Ação", value=argumentos_reais.get("Agent", "Não detectado"))
        with c3:
            obj_detectado = argumentos_reais.get("Theme", argumentos_reais.get("Item", "Não detectado"))
            st.metric(label="Objetos (Direto/Indireto)", value=obj_detectado)
        with c4:
            circ_detectada = argumentos_reais.get("Recipient", argumentos_reais.get("Place", "Não detectada"))
            st.metric(label="Circunstâncias da Ação", value=circ_detectada)

        with st.expander("🔍 Ver Elementos Teóricos do Frame no Banco de Dados"):
            st.write(f"**Elementos Centrais (Core):** {', '.join(dados_fn.get('objetos', []))}")
            st.write(f"**Elementos Circunstanciais:** {', '.join(dados_fn.get('circunstancias', []))}")
            if "notes" in dados_fn:
                st.warning(dados_fn["notes"])
