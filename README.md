# edge_segmentation
Evaluation and optimization of lightweight semantic segmentation models applied to risk analysis in road traffic, considering their implementation in embedded systems operating exclusively on CPUs

# Experimentos para desenvolvimento - Versão em PT
Investigação estruturada em quatro eixos experimentais complementares.
1 - Buscar estabelecer um referencial de desempenho espacial por meio de treinamento em bases públicas. 
2 - Ivestigar a capacidade de generalização e adaptação ao contexto brasileiro. 
3 - Analisar a consistência temporal das predições em sequências de vídeo. 
4 - Tecnicas de otimização computacional voltadas à execução eficiente em hardware restrito.

Os experimentos propostos investigam:
(i) desempenho espacial 
(ii) generalização entre domínios
(iii) consistência temporal
(iv) eficiência computacional após otimização

A condução desses experimentos visa fornecer subsídios técnicos para a seleção de modelos viáveis para aplicação embarcada, contribuindo para o desenvolvimento de soluções de baixo custo
e alta relevância para a segurança viária.

## EXPERIMENTOS E OBJETIVOS:
Objetivo de estabelecer um referencial de desempenho (baseline), no qual diferentes arquiteturas são comparadas em condições controladas. A ideia é identificar qual modelo apresenta melhor desempenho em termos de precisão espacial, utilizando métricas como mIoU e acurácia por pixel. A análise não se restringe a qualidade da segmentação: são também mensurados o tempo de inferência, o consumo de recursos computacionais e a complexidade teórica do modelo, expressa em GFLOPs (Giga Floating Point Operations). Essa abordagem permite caracterizar o trade-off entre precisão e custo computacional, fundamental para a seleção de modelos em cenários com restrição de hardware.

NOTAS: 
- Reprodutibilidade científica: resultados obtidos possam ser reproduzidos sob as mesmas condições experimentais. É necessário controlar fatores como inicialização aleatória por meio de random seeds, registrar detalhadamente hiperparâmetros, arquiteturas, versões de bibliotecas e especificações de hardware, além de assegurar rastreabilidade completa dos experimentos.
- Organização dos dados: conjuntos devem ser devidamente particionados em treinamento, validação e teste, evitando vazamento de dados. Sempre que possível, adotar divisão estratificada, garantindo representatividade das classes. A resolução das imagens deve ser tratada como um hiperparâmetro crítico, especialmente em cenários embarcados. Dessa forma, serão conduzidos experimentos avaliando o impacto do downsampling (por exemplo, de 1080p para resoluções como 512x256 ou 360p) na relação entre precisão e velocidade de processamento.
- Casos de maior variabilidade ou menor volume de dados: utilização de técnicas como validação cruzada (k-fold cross-validation), permitindo estimar o desempenho médio e sua variabilidade estatística. Os resultados devem ser analisados considerando média e desvio padrão, garantindo que diferentes execuções produzam resultados consistentes dentro de um intervalo confiável.
- Coleta de métricas de qualidade e desempenho computacional: mIoU, acurácia por pixel, F1-score, tempo de treinamento, tempo de inferência por imagem, taxa de processamento (FPS), latência, consumo de CPU e memória, além da complexidade teórica em GFLOPs.
- Desempenho computacional serão aferidas em hardware representativo do cenário alvo, comparando a execução em processadores de arquitetura x86 (tipicamente utilizados em ambientes de desenvolvimento e servidores de borda) e arquitetura ARM (característica de sistemas embarcados veiculares de baixo custo), de modo a validar a portabilidade dos modelos e a efetividade das otimizações propostas em diferentes plataformas.
- Resultados: devem ser estruturados e padronizados, incluindo logs de treinamento, métricas por execução, configurações experimentais e quando necessário, as próprias predições dos modelos. Importante para: reprodutibilidade, auditabilidade e reutilização dos dados, permitindo a geração de análises adicionais sem a necessidade de reexecução completa dos experimentos.

### EXPERIMENTO 1:
O objetivo é estabelecer um referencial de desempenho (baseline) para modelos leves de segmentação semântica em condições controladas.
Implementação: pipeline reprodutível, capaz de analisar, preparar, treinar, avaliar e armazenar resultados de forma padronizada.
Base metodológica para os demais conjuntos de dados da pesquisa. O pipeline será desenvolvido de forma genérica, permitindo sua reutilização posterior em bases como CaRINA e nos dados privados da Coamo.


#### Etapa 1: Análise Exploratória do Dataset
-> Script genérico de análise exploratória do dataset. Composição dos dados, classes, distribuição e limitações.
Extrair pelo script: 
- Quantidade total de imagens;
- Resolução original das imagens;
- Lista de classes presentes;
- Frequência absoluta e relativa de pixels por classe;
- Frequência de aparição de cada classe por imagem;
- Grau de desbalanceamento entre classes;
- Quantidade de pixels ignorados ou não rotulados;
- Exemplos visuais de imagem e máscara sobreposta.
Também deverão ser gerados gráficos e relatórios, tais como:
- Gráfico de barras com frequência de pixels por classe;
- Gráfico de aparição das classes por imagem;
- Histograma de resoluções;
- Relatório estatístico em formato .csv ou .json;
- Amostras visuais para conferência qualitativa das máscaras.

Objetivo:dentificar problemas como desbalanceamento de classes, ausência de classes críticas, inconsistências nas máscaras e diferenças de resolução. Os resultados dessa análise deverão orientar decisões posteriores, como escolha da função de perda, estratégia de amostragem, resolução de entrada e agrupamento de classes.


#### Etapa 2: Definição e Mapeamento de Classes

Função de cada arquivo:

dataset_statistics.json:
Contém estatísticas globais do dataset preparado, como número de imagens por split, resoluções, distribuição de classes, etc. É um resumo quantitativo do dataset já organizado.

split_report.csv:
Relatório tabular detalhando a divisão dos dados (train/val/test), normalmente com colunas como: nome do arquivo, split, resolução, presença de classes, etc. Serve para auditoria e rastreabilidade dos arquivos em cada split.