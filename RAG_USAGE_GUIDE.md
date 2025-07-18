# 🎯 **MedMax RAG System - Guia Completo de Uso**

## 📋 **Overview do Sistema**

O sistema RAG tem **3 tipos de engines** diferentes, cada um com características específicas:

### **1. Simple Engine (Padrão)**
- **Mais rápido** e **simples**
- Usa configurações padrão do LlamaIndex
- **Ideal para:** Testes rápidos, demos, uso geral

### **2. Standard Engine**
- Usa **retriever customizado** (QdrantRetriever)
- **Controle total** sobre busca semântica
- Configurações específicas de `top_k` e `score_threshold`
- **Ideal para:** Produção, performance otimizada

### **3. Enhanced Engine**
- **Máxima flexibilidade**
- Modo debug avançado
- Configurações customizáveis
- **Ideal para:** Pesquisa, experimentação, A/B testing

---

## 🚀 **Comandos Disponíveis**

### **Modo Interactive (Sessão de Perguntas)**
```bash
# Basic interactive session
python -m src.rag.main interactive

# Com configurações customizadas
python -m src.rag.main interactive --engine-type enhanced --top-k 10

# Com collection específica
python -m src.rag.main interactive --collection-name "minha_collection"
```

### **Modo Query (Pergunta Única)**
```bash
# Pergunta simples
python -m src.rag.main query "What is diabetes?"

# Com detalhes das fontes (verbose)
python -m src.rag.main query "Treatment for hypertension" --verbose

# Com engine específico
python -m src.rag.main query "COVID-19 treatments" --engine-type standard --top-k 8

# Com modelo específico
python -m src.rag.main query "Cancer immunotherapy" --model gpt-4 --verbose
```

---

## ⚙️ **Parâmetros Detalhados**

| Parâmetro | Opções | Padrão | Descrição |
|-----------|--------|--------|-----------|
| **mode** | `interactive`, `query` | - | Modo de operação |
| **question** | String | - | Pergunta (obrigatório para query) |
| **--engine-type** | `simple`, `standard`, `enhanced` | `simple` | Tipo de engine |
| **--top-k** | Número inteiro | `5` | Documentos recuperados |
| **--model** | Modelo OpenAI | `gpt-4o-mini` | LLM a usar |
| **--collection-name** | String | `medmax_pubmed` | Collection do Qdrant |
| **--verbose** | Flag | `False` | Detalhes das fontes |

---

## 🎮 **Exemplos Práticos**

### **Teste Rápido (Simple)**
```bash
python -m src.rag.main query "Is vitamin D good for bones?" --verbose
```
**Output esperado:**
```
📚 SOURCES (5 found):
1. Score: 0.892 | ID: 1234 | Decision: yes
   Question: Does vitamin D supplementation improve bone health in elderly?

2. Score: 0.847 | ID: 5678 | Decision: yes  
   Question: Vitamin D deficiency and fracture risk in postmenopausal women?
```

### **Sessão Interativa (Enhanced)**
```bash
python -m src.rag.main interactive --engine-type enhanced --top-k 8 --verbose
```
**Permite:**
- Fazer múltiplas perguntas
- Ver detalhes avançados
- Configurações otimizadas

### **Produção (Standard)**
```bash
python -m src.rag.main query "Efficacy of metformin in diabetes" --engine-type standard --top-k 10
```
**Características:**
- Retriever customizado
- Performance otimizada
- Controle fino da busca

---

## 🔧 **Diferenças Técnicas dos Engines**

### **Simple Engine**
```python
query_engine = create_simple_query_engine(index, similarity_top_k=top_k)
```
- **Prós:** Rápido, simples, funciona bem
- **Contras:** Menos controle, configurações limitadas

### **Standard Engine**
```python
custom_retriever = create_custom_retriever(client, collection, top_k, threshold)
query_engine = create_query_engine(index, retriever=custom_retriever)
```
- **Prós:** Retriever customizado, controle total, performance
- **Contras:** Mais complexo, setup mais demorado

### **Enhanced Engine**
```python
query_engine = enhanced_query_engine(index, custom_prompt=prompt, verbose=True)
```
- **Prós:** Máxima flexibilidade, debug avançado, experimentação
- **Contras:** Mais lento, complexo para uso básico

---

## 📊 **Interpretando as Saídas**

### **Campos dos Metadados:**
```json
{
  "question": "Pergunta original do estudo",
  "final_decision": "yes/no/unclear", 
  "record_id": "ID único do registro",
  "contexts": ["Contexto 1", "Contexto 2"],
  "long_answer": "Resposta detalhada do estudo"
}
```

### **Score de Similaridade:**
- **0.9-1.0:** Altamente relevante
- **0.8-0.9:** Muito relevante  
- **0.7-0.8:** Relevante
- **< 0.7:** Pouco relevante (filtrado por padrão)

### **Interpretação das Decisions:**
- **"yes":** Intervenção é benéfica/eficaz
- **"no":** Intervenção é prejudicial/ineficaz
- **"unclear":** Evidência insuficiente

---

## 🛠️ **Scripts Disponíveis**

### **1. População do Qdrant**
```bash
scripts\01_populate_qdrant.bat
```
- Popula vector store com dados PubMed

### **2. Teste RAG**
```bash
scripts\02_rag_query.bat
```
- Teste rápido do sistema RAG

### **3. Avaliação**
```bash
scripts\03_test_evaluation.bat    # Teste com 10 perguntas
scripts\03_run_evaluation.bat     # Avaliação completa
```

---

## 🎯 **Casos de Uso Recomendados**

### **Para Desenvolvimento/Teste:**
```bash
python -m src.rag.main interactive --engine-type simple
```

### **Para Pesquisa/Análise:**
```bash
python -m src.rag.main query "pergunta específica" --engine-type enhanced --verbose
```

### **Para Produção/Performance:**
```bash
python -m src.rag.main query "pergunta" --engine-type standard --top-k 8
```

### **Para Experimentos:**
```bash
python -m src.rag.main interactive --engine-type enhanced --top-k 15 --model gpt-4
```

---

## ❓ **Troubleshooting**

### **Se aparecer "N/A" nas fontes:**
- Verificar se Qdrant está rodando: `docker ps`
- Verificar se collection existe
- Verificar se dados foram indexados corretamente

### **Se der erro de conexão:**
- Verificar `OPENAI_API_KEY` no `.env`
- Verificar se Qdrant está acessível em `localhost:6333`

### **Para debug avançado:**
```bash
python -m src.rag.main query "pergunta" --engine-type enhanced --verbose
```

---

## 🚀 **Próximos Passos**

1. **Teste basic:** Comece com `simple` engine
2. **Explore interactive:** Use modo interativo para experimentar
3. **Otimize performance:** Mude para `standard` engine  
4. **Experimente:** Use `enhanced` para pesquisa avançada

O sistema está **completamente funcional** e pronto para uso! 🎉
