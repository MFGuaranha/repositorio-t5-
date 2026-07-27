import sqlite3
import torch
import torch.nn as nn
import json
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

class HybridT5Model(nn.Module):
    def __init__(self, model_name="t5-small", num_frames=50):
        super(HybridT5Model, self).__init__()
        print("[LOG app_model] 🔄 Inicializando a classe HybridT5Model e carregando pesos do T5...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.hidden_size = self.t5.config.d_model
        self.frame_classifier = nn.Linear(self.hidden_size, num_frames)
        self.id_para_frame = {i: "Sending" if i == 0 else f"Frame_{i}" for i in range(num_frames)}
        print("[LOG app_model] 🎯 Classe HybridT5Model pronta para uso.")

    def forward(self, input_ids, attention_mask, labels=None, frame_labels=None):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        frame_logits = self.frame_classifier(pooled_output)
        
        if labels is not None and frame_labels is not None:
            outputs = self.t5(encoder_outputs=encoder_outputs, attention_mask=attention_mask, labels=labels)
            loss_generation = outputs.loss
            loss_fn = nn.CrossEntropyLoss()
            loss_classification = loss_fn(frame_logits, frame_labels)
            loss = loss_generation + loss_classification
            return loss, frame_logits, outputs
            
        return frame_logits, encoder_outputs

    def predict(self, texto_input):
        print(f"\n[LOG app_model] 🧠 ENTRADA EM 'predict': Recebendo string formatada: '{texto_input}'")
        self.eval()
        inputs = self.tokenizer(texto_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            print("[LOG app_model] ⚙ Rodando a inferência no PyTorch (Classificação + Geração)...")
            frame_logits, encoder_outputs = self.forward(input_ids, attention_mask)
            frame_id = torch.argmax(frame_logits, dim=-1).item()
            frame_predito = self.id_para_frame.get(frame_id, "Unknown_Frame")
            
            outputs_gerados = self.t5.generate(
                encoder_outputs=encoder_outputs, 
                attention_mask=attention_mask,
                max_length=128,
                num_beams=2,
                early_stopping=True
            )
            
            textos_decodificados = self.tokenizer.batch_decode(outputs_gerados, skip_special_tokens=True)
            texto_gerado = textos_decodificados[0] if textos_decodificados else "{}"
            
        texto_gerado = str(texto_gerado).strip()
        print(f"[LOG app_model] 📝 Texto gerado decodificado: '{texto_gerado}'")
            
        try:
            dados_argumentos = json.loads(texto_gerado)
        except (json.JSONDecodeError, TypeError):
            print("[LOG app_model] ⚠️ Falha ao ler JSON puro. Ativando extração Regex Fallback...")
            dados_argumentos = {"Theme": [], "Recipient": []}
            theme_match = re.search(r"Theme\":\s*\[(.*?)\]", texto_gerado)
            recipient_match = re.search(r"Recipient\":\s*\[(.*?)\]", texto_gerado)
            
            if theme_match:
                dados_argumentos["Theme"] = [int(x) for x in theme_match.group(1).split(",") if x.strip().isdigit()]
            if recipient_match:
                dados_argumentos["Recipient"] = [int(x) for x in recipient_match.group(1).split(",") if x.strip().isdigit()]
                
        print(f"[LOG app_model] 🚀 SAÍDA DE 'predict': Retornando Frame '{frame_predito}' e dicionário de argumentos.")
        return {
            "frame": frame_predito,
            "arguments": dados_argumentos if isinstance(dados_argumentos, dict) else {}
        }

def buscar_detalhes_framenet(db_path, frame_name):
    print(f"\n[LOG app_model] 📂 ENTRADA EM 'buscar_detalhes_framenet': Buscando frame '{frame_name}' no SQLite...")
    dados_linguisticos = {
        "acao": frame_name,
        "definicao": "Sem definição cadastrada.",
        "agente": "Não identificado",
        "objetos": [],
        "circunstancias": []
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT definicao FROM frames WHERE nome = ?", (frame_name,))
        print (" SQL ", "SELECT definicao FROM frames WHERE nome = ?", (frame_name,))
        row_frame = cursor.fetchone()
        print ("row_frame ", row_frame)
        if row_frame:
            dados_linguisticos["definicao"] = row_frame[0]
            
        cursor.execute("""
            SELECT fe.nome_fe, fe.tipo_coreness 
            FROM frame_elements fe
            JOIN frames f ON fe.frame_id = f.id_frame
            WHERE f.nome = ?
        """, (frame_name,))
        print (" SQL 2 ", """"
            SELECT fe.nome_fe, fe.tipo_coreness 
            FROM frame_elements fe
            JOIN frames f ON fe.frame_id = f.id_frame
            WHERE f.nome = ?
        """, (frame_name,))
        rows_fe = cursor.fetchall()
        print(f"[LOG app_model] 🗄 Encontrados {len(rows_fe)} elementos vinculados ao frame no banco.")
        for nome_fe, tipo_coreness in rows_fe:
            nome_lower = nome_fe.lower()
            
            if "agent" in nome_lower or "protagonist" in nome_lower:
                dados_linguisticos["agente"] = nome_fe
            elif tipo_coreness == "Core" or "theme" in nome_lower or "item" in nome_lower:
                dados_linguisticos["objetos"].append(nome_fe)
            else:
                dados_linguisticos["circunstancias"].append(nome_fe)
                
        conn.close()
    except Exception as e:
        print(f"[LOG app_model] ❌ Erro ao acessar o banco de dados: {e}")
        dados_linguisticos["notes"] = f"Erro ao acessar a estrutura do banco: {str(e)}"
        
    print("[LOG app_model] 📤 SAÍDA DE 'buscar_detalhes_framenet': Retornando dados estruturados para a interface.")
    print("dados_linguisticos  ", dados_linguisticos )
    return dados_linguisticos
