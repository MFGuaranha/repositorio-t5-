import streamlit as st
import os
import zipfile
import nltk
from deep_translator import GoogleTranslator  # Tradutor leve e rápido
from app_model import HybridT5Model, buscar_detalhes_framenet

PATH_ZIP = "framenet_completa.zip"
PATH_DB_LOCAL = "framenet_completa.db"

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    pass

@st.cache_resource
def carregar_recursos():
    if not os.path.exists(PATH_DB_LOCAL):
        if os.path.exists(PATH_ZIP):
            with zipfile.ZipFile(PATH_ZIP, 'r') as zip_ref:
                zip_ref.extractall(".")
        else:
            st.error(f"❌ Erro crítico: O arquivo compactado `{PATH_ZIP}` não foi encontrado!")
    
    return HybridT5Model(model_name="t5-small", num_frames=50)

modelo_hibrido = carregar_recursos()

def pre_processar_texto_nltk(texto_ingles):
    """Processa o texto já traduzido para o inglês no padrão exigido pelo T5."""
    palavras = nltk.word_tokenize(texto_ingles)
    tags_gramaticais = nltk.pos_tag(palavras)
    gatilho_idx = None
    
    for idx, (palavra, tag) in enumerate(tags_gramaticais):
        if tag.startswith('VB'):  # Identifica o verbo em inglês (ex: 'send', 'buy')
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

# --- INTERFACE ---
st.set_page_config(page_title="Análise Semântica FrameNet T5", page_icon="🧠", layout="wide")
st.title("🧠 Extrator Semântico Baseado em Frames (T5 Híbrido)")
st.subheader("Tradução Automática e Integração com Banco de Dados FrameNet")

if not os.path.exists(PATH_DB_LOCAL):
    st.error(f"❌ O arquivo do banco de dados `{PATH_DB_LOCAL}` não foi encontrado!")
else:
    st.success(f"✔ Banco de dados FrameNet detectado e pronto!")

frases_sugeridas = [
    "enviar o relatório para o Diretor",
    "comprar um livro na loja ontem",
    "entregar os documentos para a secretária",
    "vender o carro antigo para o vizinho"
]

frase_selecionada = st.selectbox("Escolha uma frase de exemplo ou digite abaixo (em Português):", frases_sugeridas)
texto_usuario = st.text_input("Digite ou modifique a frase para análise:", value=frase_selecionada)

if st.button("🚀 Processar Frase", type="primary"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, insira um texto válido.")
    else:
        # 1. FLUXO DE TRADUÇÃO AUTOMÁTICA (PT -> EN)
        with st.spinner("Traduzindo frase para o inglês padrão FrameNet..."):
            try:
                frase_ingles = GoogleTranslator(source='pt', target='en').translate(texto_usuario)
                st.info(f"🇺🇸 **Tradução gerada para processamento:** `{frase_ingles}`")
            except Exception as e:
                st.error(f"Erro na tradução: {e}")
                frase_ingles = texto_usuario # Fallback seguro
        
        # 2. Processamento do texto traduzido com o motor NLTK
        texto_idx, input_modelo, lista_palavras = pre_processar_texto_nltk(frase_ingles)
        print("texto_idx, input_modelo, lista_palavras ",texto_idx, input_modelo, lista_palavras )
        col1, col2 = st.columns(2)
        with col1:
            st.caption("**Texto em Inglês Indexado:**")
            st.code(texto_idx)
        with col2:
            st.caption("**Entrada Formatada para o T5 (Target):**")
            st.code(input_modelo)
            
        # 3. Execução da Inteligência Artificial
        with st.spinner("Modelos preditivos processando os frames estruturados..."):
            json_real_retorno = modelo_hibrido.predict(input_modelo)
            
        st.divider()
        st.subheader("📦 Saída Estruturada do Modelo (JSON Real)")
        st.json(json_real_retorno)
        
        # 4. Consulta ao banco de dados usando o frame predito em inglês
        nome_frame = json_real_retorno["frame"]
        dados_fn = buscar_detalhes_framenet(PATH_DB_LOCAL, nome_frame)
        print("dados_fn ", dados_fn )
        st.divider()
        st.subheader("📋 Informações Linguísticas Extraídas (Valores Reais)")
        
        argumentos_reais = {}
        for papel, indices in json_real_retorno.get("arguments", {}).items():
            if not isinstance(indices, list):
                indices = [indices]
                
            tokens_fragmento = []
            for i in indices:
                try:
                    if i is not None:
                        idx_limpo = int(str(i).strip())
                        if idx_limpo < len(lista_palavras):
                            tokens_fragmento.append(lista_palavras[idx_limpo])
                except ValueError:
                    continue 
            # Salva o termo real extraído da frase em inglês
            argumentos_reais[papel] = " ".join(tokens_fragmento) if tokens_fragmento else "Não detectado"
            
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Ação Solicitada (Frame)", value=dados_fn["acao"])
        with c2:
            st.metric(label="Agente da Ação (Agent)", value=argumentos_reais.get("Agent", "Não detectado"))
        with c3:
            obj_detectado = argumentos_reais.get("Theme", argumentos_reais.get("Item", "Não detectado"))
            st.metric(label="Objeto Extraído (Theme)", value=obj_detectado)
        with c4:
            circ_detectada = argumentos_reais.get("Recipient", argumentos_reais.get("Place", "Não detectada"))
            st.metric(label="Circunstâncias (Recipient)", value=circ_detectada)
            
        with st.expander("🔍 Ver Elementos Teóricos do Frame no Banco de Dados"):
            # Converte a tupla de definição do banco para string pura de exibição
            def_limpa = dados_fn['definicao'][0] if isinstance(dados_fn['definicao'], tuple) else dados_fn['definicao']
            st.markdown(f"**Definição do Frame no Banco:** *{def_limpa}*")
            st.write(f"**Elementos Centrais (Core):** {', '.join(dados_fn.get('objetos', []))}")
            st.write(f"**Elementos Circunstanciais:** {', '.join(dados_fn.get('circunstancias', []))}")
            
        if "notes" in dados_fn:
            st.warning(dados_fn["notes"])
