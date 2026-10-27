(function () {
  'use strict';
  const key = 'neural-earth-locale-v1';
  const languages = ['en', 'fr', 'zh', 'es', 'pt'];
  const messages = {
    en: { language:'Language', map:'Map', globe:'Globe', zoomOut:'Zoom out', zoomIn:'Zoom in', fit:'World view', initial:'Initial region', native:'30 m / pixel', measure:'Measure distance', random:'New seed', generate:'Generate', help:'Scroll: zoom · Drag: pan · Arrow keys: move', seed:'World seed', loading:'Loading', tools:'Tools', rendering:'Rendering', close:'Close panel' },
    fr: { language:'Langue', map:'Carte', globe:'Globe', zoomOut:'Dézoomer', zoomIn:'Zoomer', fit:'Vue mondiale', initial:'Région initiale', native:'30 m / pixel', measure:'Mesurer une distance', random:'Nouvelle graine', generate:'Générer', help:'Molette : zoom · Glisser : déplacer · Flèches : déplacer', seed:'Graine du monde', loading:'Chargement', tools:'Outils', rendering:'Rendu', close:'Fermer le panneau' },
    zh: { language:'语言', map:'地图', globe:'地球仪', zoomOut:'缩小', zoomIn:'放大', fit:'世界视图', initial:'初始区域', native:'30 米 / 像素', measure:'测量距离', random:'新种子', generate:'生成', help:'滚轮：缩放 · 拖动：平移 · 方向键：移动', seed:'世界种子', loading:'加载中', tools:'工具', rendering:'渲染', close:'关闭面板' },
    es: { language:'Idioma', map:'Mapa', globe:'Globo', zoomOut:'Alejar', zoomIn:'Acercar', fit:'Vista mundial', initial:'Región inicial', native:'30 m / píxel', measure:'Medir distancia', random:'Nueva semilla', generate:'Generar', help:'Rueda: zoom · Arrastrar: desplazar · Flechas: mover', seed:'Semilla del mundo', loading:'Cargando', tools:'Herramientas', rendering:'Renderizado', close:'Cerrar panel' },
    pt: { language:'Idioma', map:'Mapa', globe:'Globo', zoomOut:'Afastar', zoomIn:'Aproximar', fit:'Visão mundial', initial:'Região inicial', native:'30 m / pixel', measure:'Medir distância', random:'Nova semente', generate:'Gerar', help:'Roda: zoom · Arrastar: mover · Setas: mover', seed:'Semente do mundo', loading:'Carregando', tools:'Ferramentas', rendering:'Renderização', close:'Fechar painel' },
  };
  // Five translations in the same order as `languages`.
  const panelText = {
    worldPreparation:['World preparation','Préparation du monde','世界准备','Preparación del mundo','Preparação do mundo'],
    prepareEntire:['Prepare entire world','Préparer le monde entier','准备整个世界','Preparar el mundo entero','Preparar o mundo inteiro'],
    prepare:['Prepare neural world','Préparer le monde neuronal','准备神经世界','Preparar mundo neuronal','Preparar mundo neural'],
    pause:['Pause preparation','Suspendre la préparation','暂停准备','Pausar preparación','Pausar preparação'],
    preparationHint:['Disabled by default: computes the visible area, then its neighbors once ready. Whole-world preparation is optional.','Désactivé par défaut : calcule la zone visible puis ses voisines. La préparation du monde entier est facultative.','默认关闭：先计算可见区域，完成后计算相邻区域。准备整个世界是可选的。','Desactivado por defecto: calcula el área visible y después sus vecinas. La preparación del mundo entero es opcional.','Desativado por padrão: calcula a área visível e depois suas vizinhas. A preparação do mundo inteiro é opcional.'],
    loadRefine:['Loading & refinement','Chargement et raffinement','加载与细化','Carga y refinamiento','Carregamento e refinamento'],
    prefetch:['Prefetch neighboring tiles','Précharger les tuiles voisines','预加载相邻图块','Precargar teselas vecinas','Pré-carregar blocos vizinhos'],
    depth:['Refinement · LOD depth','Raffinement · profondeur LOD','细化 · LOD 深度','Refinamiento · profundidad LOD','Refinamento · profundidade LOD'],
    renderSea:['Render sea','Afficher la mer','渲染海洋','Mostrar mar','Exibir mar'],
    seaMax:['Sea max LOD','LOD maximal de la mer','海洋最大 LOD','LOD máximo del mar','LOD máximo do mar'],
    cacheBudget:['Cache budget (GiB)','Budget du cache (Gio)','缓存预算 (GiB)','Presupuesto de caché (GiB)','Limite de cache (GiB)'],
    cacheGap:['Cache · LOD gap','Cache · écart LOD','缓存 · LOD 差值','Caché · diferencia LOD','Cache · diferença de LOD'],
    gpuRender:['GPU rendering','Rendu GPU','GPU 渲染','Renderizado GPU','Renderização GPU'],
    coarseGpu:['Coarse on GPU','Coarse sur GPU','GPU 粗层','Coarse en GPU','Coarse na GPU'],
    mapDetails:['Map details','Détails de la carte','地图详情','Detalles del mapa','Detalhes do mapa'],
  };
  const layerSections = {
    'Main maps':['Main maps','Cartes principales','主要地图','Mapas principales','Mapas principais'],
    'Map styles':['Map styles','Styles de carte','地图样式','Estilos de mapa','Estilos de mapa'],
    'Seasonal climate':['Seasonal climate','Climat saisonnier','季节气候','Clima estacional','Clima sazonal'],
    'Atmosphere':['Atmosphere','Atmosphère','大气','Atmósfera','Atmosfera'],
    'Oceans':['Oceans','Océans','海洋','Océanos','Oceanos'],
    'Orogen plates':['Orogen plates','Plaques Orogen','Orogen 板块','Placas de Orogen','Placas Orogen'],
    'Relief formation':['Relief formation','Formation du relief','地形形成','Formación del relieve','Formação do relevo'],
    'Other layers':['Other layers','Autres couches','其他图层','Otras capas','Outras camadas'],
  };
  const layerNames = {
    relief:['Relief','Relief','地形','Relieve','Relevo'],
    soil:['Soils','Sols','土壤','Suelos','Solos'],
    pedology:['Pedology','Pédologie','土壤学','Pedología','Pedologia'],
    'orogen-biomes':['Biomes','Biomes','生物群系','Biomas','Biomas'],
    temperature:['Annual temperature','Température annuelle','年均温度','Temperatura anual','Temperatura anual'],
    precipitation:['Annual precipitation','Précipitations annuelles','年降水量','Precipitación anual','Precipitação anual'],
    'fantasy-ink':['Fantasy · Ink','Fantasy · Encre','奇幻 · 墨绘','Fantasía · Tinta','Fantasia · Tinta'],
    'fantasy-color':['Fantasy · Watercolor','Fantasy · Aquarelle','奇幻 · 水彩','Fantasía · Acuarela','Fantasia · Aquarela'],
    'orogen-plates':['Orogen · plates','Orogen · plaques','Orogen · 板块','Orogen · placas','Orogen · placas'],
    'orogen-crust':['Orogen · continental / oceanic crust','Orogen · croûte continentale / océanique','Orogen · 大陆 / 海洋地壳','Orogen · corteza continental / oceánica','Orogen · crosta continental / oceânica'],
    'orogen-boundaries':['Orogen · boundary types','Orogen · types de limites','Orogen · 板块边界类型','Orogen · tipos de límite','Orogen · tipos de limite'],
    'orogen-convergence':['Orogen · convergence / divergence','Orogen · convergence / divergence','Orogen · 汇聚 / 离散','Orogen · convergencia / divergencia','Orogen · convergência / divergência'],
    'orogen-uplift':['Orogen · mountain uplift','Orogen · soulèvement montagneux','Orogen · 山地隆升','Orogen · levantamiento montañoso','Orogen · soerguimento montanhoso'],
    'orogen-height':['Orogen · initial relief','Orogen · relief initial','Orogen · 初始地形','Orogen · relieve inicial','Orogen · relevo inicial'],
    'orogen-superplates':['Superplate groups','Groupes de superplaques','超级板块组','Grupos de superplacas','Grupos de superplacas'],
    'orogen-hotspots':['Original hotspot contribution','Contribution des points chauds initiaux','原始热点贡献','Contribución de los puntos calientes originales','Contribuição dos pontos quentes originais'],
    'orogen-orogeny':['Original orogenic power','Puissance orogénique initiale','原始造山强度','Potencia orogénica original','Potência orogênica original'],
    'orogen-back-arc':['Back arc contribution','Contribution d’arrière-arc','弧后贡献','Contribución de trasarco','Contribuição de retroarco'],
    'orogen-fold-ridges':['Folded mountain ridge contribution','Contribution des crêtes plissées','褶皱山脊贡献','Contribución de crestas plegadas','Contribuição das cristas dobradas'],
    'orogen-erosion':['Relief change from erosion','Modification du relief par l’érosion','侵蚀引起的地形变化','Cambio del relieve por erosión','Alteração do relevo pela erosão'],
    'orogen-pressure-summer':['Summer atmospheric pressure','Pression atmosphérique estivale','夏季气压','Presión atmosférica estival','Pressão atmosférica de verão'],
    'orogen-pressure-winter':['Winter atmospheric pressure','Pression atmosphérique hivernale','冬季气压','Presión atmosférica invernal','Pressão atmosférica de inverno'],
    'orogen-rain-shadow':['Summer rain shadow','Ombre pluviométrique estivale','夏季雨影','Sombra pluviométrica estival','Sombra de chuva no verão'],
    'orogen-continentality':['Distance from ocean influence','Distance à l’influence océanique','距海洋影响的距离','Distancia a la influencia oceánica','Distância da influência oceânica'],
    'orogen-temperature-summer':['Northern summer temperature','Température de l’été boréal','北半球夏季温度','Temperatura del verano boreal','Temperatura do verão boreal'],
    'orogen-temperature-winter':['Northern winter temperature','Température de l’hiver boréal','北半球冬季温度','Temperatura del invierno boreal','Temperatura do inverno boreal'],
    'orogen-precip-summer':['Northern summer precipitation','Précipitations du semestre d’été boréal','北半球夏半年降水量','Precipitación del semestre de verano boreal','Precipitação do semestre de verão boreal'],
    'orogen-precip-winter':['Northern winter precipitation','Précipitations du semestre d’hiver boréal','北半球冬半年降水量','Precipitación del semestre de invierno boreal','Precipitação do semestre de inverno boreal'],
    'orogen-wind-summer':['Northern summer winds','Vents de l’été boréal','北半球夏季风','Vientos del verano boreal','Ventos do verão boreal'],
    'orogen-wind-winter':['Northern winter winds','Vents de l’hiver boréal','北半球冬季风','Vientos del invierno boreal','Ventos do inverno boreal'],
    'orogen-currents-summer':['Northern summer currents','Courants de l’été boréal','北半球夏季洋流','Corrientes del verano boreal','Correntes do verão boreal'],
    'orogen-currents-winter':['Northern winter currents','Courants de l’hiver boréal','北半球冬季洋流','Corrientes del invierno boreal','Correntes do inverno boreal'],
  };
  const layerLabel = ['Layer visible','Couche affichée','显示的图层','Capa visible','Camada visível'];
  const generationText = {
    generation:['Generation','Génération','生成','Generación','Geração'],
    generationHeading:['Generation · initial relief','Génération · relief initial','生成 · 初始地形','Generación · relieve inicial','Geração · relevo inicial'],
    climate:['Climate','Climat','气候','Clima','Clima'],
    noise:['Relief noise · SNR','Bruit du relief · SNR','地形噪声 · SNR','Ruido del relieve · SNR','Ruído do relevo · SNR'],
    worldSource:['1 · World source','1 · Source du monde','1 · 世界来源','1 · Fuente del mundo','1 · Origem do mundo'],
    worldCreation:['World creation','Création du monde','创建世界','Creación del mundo','Criação do mundo'],
    generator:['Generator','Générateur','生成器','Generador','Gerador'],
    customSource:['Custom relief','Relief personnalisé','自定义地形','Relieve personalizado','Relevo personalizado'],
    geometry:['Geometry','Géométrie','几何形状','Geometría','Geometria'],
    diameter:['Diameter (km)','Diamètre (km)','直径（千米）','Diámetro (km)','Diâmetro (km)'],
    continentPreset:['Continents · preset','Continents · préréglage','大陆 · 预设','Continentes · preajuste','Continentes · predefinição'],
    continentStrength:['Continental strength','Intensité continentale','大陆强度','Intensidad continental','Intensidade continental'],
    smoothing:['Smoothing radius (km)','Rayon de lissage (km)','平滑半径（千米）','Radio de suavizado (km)','Raio de suavização (km)'],
    landSea:['Land / sea bias','Biais terre / mer','陆地 / 海洋偏向','Sesgo tierra / mar','Viés terra / mar'],
    erosion:['Orogen erosion','Érosion Orogen','Orogen 侵蚀','Erosión Orogen','Erosão Orogen'],
    sourceNoise:['Source fields · procedural noise','Champs source · bruit procédural','源字段 · 程序噪声','Campos de origen · ruido procedural','Campos de origem · ruído procedural'],
    channel:['Channel','Canal','通道','Canal','Canal'],
    apply:['Apply','Appliquer','应用','Aplicar','Aplicar'],
    reset:['Reset defaults','Rétablir les valeurs par défaut','恢复默认值','Restablecer valores','Restaurar padrões'],
    sphere:['Spherical planet','Planète sphérique','球形行星','Planeta esférico','Planeta esférico'],
    plane:['Plane · unprojected map','Plan · carte non projetée','平面 · 未投影地图','Plano · mapa sin proyección','Plano · mapa sem projeção'],
    reliefOptions:['Relief · options','Relief · options','地形 · 选项','Relieve · opciones','Relevo · opções'],
    erosionHeading:['Erosion','Érosion','侵蚀','Erosión','Erosão'],
    erosionHelp:['Erosion runs before climate and neural refinement. Choose None, Orogen CPU or GPU, or City GPU.','L’érosion précède le climat et le raffinement neuronal. Choisissez Aucune, Orogen CPU ou GPU, ou City GPU.','侵蚀在气候和神经网络细化之前运行。请选择无、Orogen CPU 或 GPU，或 City GPU。','La erosión precede al clima y al refinamiento neuronal. Elige Ninguna, Orogen CPU o GPU, o City GPU.','A erosão ocorre antes do clima e do refinamento neural. Escolha Nenhuma, Orogen CPU ou GPU, ou City GPU.'],
    cityGpu:['City · GPU settings','City · réglages GPU','City · GPU 设置','City · ajustes GPU','City · configurações GPU'],
    cityHelp:['Hydraulic incision, thermal relaxation and diffusion. Strength 0 preserves relief; the engine preserves sea level and oceans.','Incision hydraulique, relaxation thermique et diffusion. Une intensité de 0 préserve le relief ; le moteur préserve le niveau marin et les océans.','包括水力切割、热松弛和扩散。强度为 0 时保留地形；引擎保留海平面和海洋。','Incisión hidráulica, relajación térmica y difusión. Una intensidad de 0 conserva el relieve; el motor conserva el nivel del mar y los océanos.','Incisão hidráulica, relaxamento térmico e difusão. Intensidade 0 preserva o relevo; o mecanismo preserva o nível do mar e os oceanos.'],
    gpuHeading:['Orogen · optional GPU acceleration','Orogen · accélération GPU facultative','Orogen · 可选 GPU 加速','Orogen · aceleración GPU opcional','Orogen · aceleração GPU opcional'],
    gpuHelp:['GPU options follow your saved local settings. NVIDIA CUDA; unported stages run on CPU. GPU propagation and erosion change terrain shape. City is a separate erosion engine. Apply, then Save A / Show A to compare.','Les options GPU suivent vos réglages locaux enregistrés. NVIDIA CUDA ; les étapes non portées utilisent le CPU. La propagation et l’érosion GPU modifient le relief. City est un moteur d’érosion distinct. Appliquez, puis utilisez Enregistrer A / Afficher A pour comparer.','GPU 选项遵循本地保存的设置。使用 NVIDIA CUDA；未移植的阶段在 CPU 上运行。GPU 传播和侵蚀会改变地形。City 是独立的侵蚀引擎。应用后使用保存 A / 显示 A 进行比较。','Las opciones GPU siguen los ajustes locales guardados. NVIDIA CUDA; las etapas no portadas usan CPU. La propagación y la erosión GPU cambian el relieve. City es un motor de erosión independiente. Aplica y usa Guardar A / Mostrar A para comparar.','As opções GPU seguem as configurações locais salvas. NVIDIA CUDA; etapas não portadas usam CPU. Propagação e erosão GPU alteram o relevo. City é um mecanismo de erosão separado. Aplique e use Salvar A / Mostrar A para comparar.'],
    presetHelp:['Presets initialize plates (Orogen), noise (Noise), or the atlas (Custom). Custom retains the choice of atlas or procedural continents.','Les préréglages initialisent les plaques (Orogen), le bruit (Bruit) ou l’atlas (Personnalisé). Personnalisé conserve le choix entre atlas et continents procéduraux.','预设初始化板块（Orogen）、噪声或图集（自定义）。自定义保留图集与程序生成大陆的选择。','Los preajustes inicializan placas (Orogen), ruido o el atlas (Personalizado). Personalizado conserva la elección entre atlas y continentes procedurales.','As predefinições inicializam placas (Orogen), ruído ou o atlas (Personalizado). Personalizado mantém a escolha entre atlas e continentes procedurais.'],
    climateIntro:['Orogen supplies the shared climate for Noise, Orogen and Custom: seasons, winds, rain, classification and biomes.','Orogen fournit le climat commun à Bruit, Orogen et Personnalisé : saisons, vents, pluie, classification et biomes.','Orogen 为噪声、Orogen 和自定义提供共同气候：季节、风、降雨、分类和生物群系。','Orogen aporta el clima común para Ruido, Orogen y Personalizado: estaciones, vientos, lluvia, clasificación y biomas.','Orogen fornece o clima comum para Ruído, Orogen e Personalizado: estações, ventos, chuva, classificação e biomas.'],
    snrHelp:['SNR here means noise / signal amplitude. Lower values constrain relief more closely to its inputs. Climate remains shared.','Ici, SNR désigne le rapport des amplitudes bruit / signal. Des valeurs plus basses rapprochent le relief de ses entrées. Le climat reste commun.','此处 SNR 表示噪声与信号振幅之比。较低的值使地形更接近输入；气候仍然共享。','Aquí SNR es la amplitud de ruido / señal. Valores menores ajustan más el relieve a sus entradas. El clima sigue siendo común.','Aqui SNR é a amplitude de ruído / sinal. Valores menores aproximam mais o relevo das entradas. O clima continua compartilhado.'],
    adaptiveSnr:['Adaptive SNR · relief / temperature / latitude','SNR adaptatif · relief / température / latitude','自适应 SNR · 地形 / 温度 / 纬度','SNR adaptativo · relieve / temperatura / latitud','SNR adaptativo · relevo / temperatura / latitude'],
    lodSnr:['Per-LOD SNR · relief detail','SNR par LOD · détail du relief','各 LOD 的 SNR · 地形细节','SNR por LOD · detalle del relieve','SNR por LOD · detalhe do relevo'],
    lodHelp:['0 inherits global SNR. Per-LOD values scale reconstructed relief noise (the local residual after the NN). They do not change climate channels or shared network computation. Experimental settings.','0 utilise le SNR global. Les valeurs par LOD modulent le bruit de relief reconstruit (résidu local après le réseau). Elles ne modifient ni les canaux climatiques ni le calcul partagé du réseau. Réglages expérimentaux.','0 继承全局 SNR。各 LOD 的值缩放重建的地形噪声（神经网络之后的局部残差），不改变气候通道或共享网络计算。实验性设置。','0 hereda el SNR global. Los valores por LOD escalan el ruido de relieve reconstruido (residuo local tras la red). No cambian los canales climáticos ni el cálculo compartido. Ajustes experimentales.','0 herda o SNR global. Os valores por LOD ajustam o ruído de relevo reconstruído (resíduo local após a rede). Não alteram os canais climáticos nem o cálculo compartilhado. Configurações experimentais.'],
    proceduralHelp:['Frequency and octaves build the NN input fields; they do not control NN refinement.','La fréquence et les octaves construisent les champs d’entrée du réseau ; elles ne contrôlent pas son raffinement.','频率和倍频程构建神经网络输入场；它们不控制网络细化。','La frecuencia y las octavas construyen los campos de entrada de la red; no controlan su refinamiento.','Frequência e oitavas constroem os campos de entrada da rede; não controlam seu refinamento.'],
    geometryHelp:['Planet: equirectangular map with width pi times diameter. Plane: a square with side equal to diameter, Cartesian distances and finite edges. Globe view is disabled. The selected generator supplies elevations. Applying these settings creates a new world and its caches.','Planète : carte équirectangulaire de largeur π fois le diamètre. Plan : carré de côté égal au diamètre, distances cartésiennes et bords finis ; la vue globe est désactivée. Le générateur choisi fournit les altitudes. Appliquer ces réglages crée un nouveau monde et ses caches.','行星：等距柱状地图的宽度为直径的 π 倍。平面：边长等于直径的正方形，使用笛卡尔距离和有限边界；地球仪视图不可用。所选生成器提供高程。应用设置会创建新世界及其缓存。','Planeta: mapa equirectangular de anchura π veces el diámetro. Plano: cuadrado de lado igual al diámetro, distancias cartesianas y bordes finitos; la vista de globo queda desactivada. El generador elegido proporciona las alturas. Aplicar estos ajustes crea un mundo nuevo y sus cachés.','Planeta: mapa equiretangular com largura π vezes o diâmetro. Plano: quadrado de lado igual ao diâmetro, distâncias cartesianas e bordas finitas; a visão de globo fica desativada. O gerador escolhido fornece as altitudes. Aplicar estas configurações cria um novo mundo e seus caches.'],
  };
  const generationOptions = {
    generatorType: {
      natural:['Noise','Bruit','噪声','Ruido','Ruído'], custom:['Custom','Personnalisé','自定义','Personalizado','Personalizado'],
    },
    customSource: {
      native:['Continental atlas','Atlas continental','大陆图集','Atlas continental','Atlas continental'],
      'natural-continental':['Procedural continents','Continents procéduraux','程序生成大陆','Continentes procedurales','Continentes procedurais'],
    },
    continentalStyle: {
      gondwana:['Gondwana','Gondwana','冈瓦纳','Gondwana','Gondwana'],
      continents:['3 continents','3 continents','3 个大陆','3 continentes','3 continentes'],
      earthlike:['Terrestrial','Terrestre','类地','Terrestre','Terrestre'],
      archipelago:['Archipelago','Archipel','群岛','Archipiélago','Arquipélago'],
    },
    reliefPipeline: {
      original:['None','Aucune','无','Ninguna','Nenhuma'],
      orogen:['Orogen · CPU','Orogen · CPU','Orogen · CPU','Orogen · CPU','Orogen · CPU'],
      'orogen-gpu':['Orogen · GPU','Orogen · GPU','Orogen · GPU','Orogen · GPU','Orogen · GPU'],
      'city-gpu':['City · GPU','City · GPU','City · GPU','City · GPU','City · GPU'],
    },
  };
  const materialText = {
    title:['Render · natural materials','Render · matériaux naturels','Render · 自然材质','Render · materiales naturales','Render · materiais naturais'],
    hint:['Materials derived from continuous climate, slope and terrain position (ridges, valleys). Seasons run from northern winter to summer and back; the south follows its own climate. Changes apply immediately.','Matériaux issus du climat continu, de la pente et de la position du terrain (crêtes, vallées). Les saisons vont de l’hiver à l’été boréal puis reviennent ; le sud suit son propre climat. Les changements sont immédiats.','材质由连续气候、坡度和地形位置（山脊、山谷）决定。季节从北半球冬季到夏季再返回；南半球遵循自身气候。更改立即生效。','Materiales derivados del clima continuo, la pendiente y la posición del terreno (crestas y valles). Las estaciones van del invierno al verano boreal y de regreso; el sur sigue su propio clima. Los cambios se aplican de inmediato.','Materiais derivados do clima contínuo, da inclinação e da posição no terreno (cristas e vales). As estações vão do inverno ao verão boreal e retornam; o sul segue seu próprio clima. As alterações são imediatas.'],
    reset:['Reset Render','Réinitialiser Render','重置 Render','Restablecer Render','Redefinir Render'],
    season:['Annual cycle (January → December)','Cycle annuel (janvier → décembre)','年度周期（1 月 → 12 月）','Ciclo anual (enero → diciembre)','Ciclo anual (janeiro → dezembro)'],
    variation:['Material variation','Variation des matériaux','材质变化','Variación de materiales','Variação dos materiais'],
    forest:['Forest cover','Couvert forestier','森林覆盖','Cobertura forestal','Cobertura florestal'],
    moisture:['Moisture (arid ↔ lush)','Humidité (aride ↔ luxuriant)','湿度（干旱 ↔ 茂盛）','Humedad (árido ↔ exuberante)','Umidade (árido ↔ exuberante)'],
    rock_slope:['Rock slope (°)','Pente rocheuse (°)','岩石坡度（°）','Pendiente rocosa (°)','Inclinação rochosa (°)'],
    snow:['Snow cover','Couvert neigeux','积雪覆盖','Cobertura de nieve','Cobertura de neve'],
    vegetation_tint:['Vegetation tint','Teinte de la végétation','植被色调','Tinte de vegetación','Tonalidade da vegetação'],
    rock_tint:['Rock tint','Teinte de la roche','岩石色调','Tinte de roca','Tonalidade da rocha'],
    snow_color:['Snow color','Couleur de la neige','雪色','Color de la nieve','Cor da neve'],
  };
  const fantasyText = {
    title:['Fantasy brushes · oblique view','Pinceaux fantasy · vue oblique','奇幻画笔 · 斜视图','Pinceles de fantasía · vista oblicua','Pincéis de fantasia · vista oblíqua'],
    density:['Density','Densité','密度','Densidad','Densidade'],
    size:['Size','Taille','大小','Tamaño','Tamanho'],
    spacing:['Spacing','Espacement','间距','Espaciado','Espaçamento'],
    export:['Export view · PNG','Exporter la vue · PNG','导出视图 · PNG','Exportar vista · PNG','Exportar visualização · PNG'],
  };
  const forestText = {
    title:['Render3D · ground and forest','Render3D · sol et forêt','Render3D · 地面与森林','Render3D · suelo y bosque','Render3D · solo e floresta'],
    enabled:['Detailed canopy','Canopée détaillée','详细树冠','Dosel detallado','Dossel detalhado'],
    density:['Density (%)','Densité (%)','密度（%）','Densidad (%)','Densidade (%)'],
    clearings:['Clearings (%)','Clairières (%)','林间空地（%）','Claros (%)','Clareiras (%)'],
    hint:['Display only. Render stays the reference beyond about 32 m per pixel; ground grains, stands and crowns resolve progressively down to 1/128 m per pixel. Render3D Brush paints overhead crowns and shrubs.','Affichage uniquement. Render reste la référence au-delà d’environ 32 m par pixel ; grains du sol, peuplements et couronnes apparaissent progressivement jusqu’à 1/128 m par pixel. Render3D Brush dessine les couronnes et buissons vus du dessus.','仅影响显示。超过约 32 米/像素时以 Render 为参考；地面颗粒、林分和树冠逐渐解析至 1/128 米/像素。Render3D Brush 绘制俯视树冠和灌木。','Solo visualización. Render sigue siendo la referencia por encima de unos 32 m por píxel; granos del suelo, masas y copas se resuelven progresivamente hasta 1/128 m por píxel. Render3D Brush dibuja copas y arbustos desde arriba.','Apenas exibição. Render continua sendo a referência acima de cerca de 32 m por pixel; grãos do solo, povoamentos e copas são resolvidos progressivamente até 1/128 m por pixel. Render3D Brush desenha copas e arbustos vistos de cima.'],
  };
  const shareText = {
    share:['Share','Partager','分享','Compartir','Compartilhar'],
    title:['Share this view','Partager cette vue','分享此视图','Compartir esta vista','Compartilhar esta visualização'],
    hint:['One portable code restores generation, rendering options and the camera on another installation. Keep the recipe field when sharing between servers. Zoom is metres per pixel; position is world metres.','Un code portable restaure la génération, le rendu et la caméra sur une autre installation. Conservez le champ recette entre serveurs. Le zoom est en mètres par pixel ; la position en mètres monde.','一个可移植代码可在另一安装中恢复生成、渲染选项和相机。跨服务器分享时请保留配方字段。缩放单位为米/像素；位置为世界坐标米。','Un código portátil restaura la generación, el renderizado y la cámara en otra instalación. Conserva el campo de receta entre servidores. El zoom se expresa en metros por píxel y la posición en metros del mundo.','Um código portátil restaura geração, renderização e câmera em outra instalação. Mantenha o campo da receita entre servidores. O zoom é em metros por pixel; a posição, em metros do mundo.'],
    uid:['Generation UID','Identifiant de génération','生成 UID','UID de generación','UID de geração'],
    code:['Share code','Code de partage','分享代码','Código para compartir','Código de compartilhamento'],
    copyCode:['Copy code','Copier le code','复制代码','Copiar código','Copiar código'],
    copyLink:['Copy link','Copier le lien','复制链接','Copiar enlace','Copiar link'],
    openLabel:['Open a code or link','Ouvrir un code ou un lien','打开代码或链接','Abrir un código o enlace','Abrir um código ou link'],
    open:['Open view','Ouvrir la vue','打开视图','Abrir vista','Abrir visualização'],
    fullWindow:['Use full window','Utiliser toute la fenêtre','使用完整窗口','Usar ventana completa','Usar janela inteira'],
    copiedCode:['Code copied','Code copié','代码已复制','Código copiado','Código copiado'],
    copiedLink:['Link copied','Lien copié','链接已复制','Enlace copiado','Link copiado'],
    copyManually:['Select and copy the code above','Sélectionnez et copiez le code ci-dessus','请选择并复制上面的代码','Selecciona y copia el código de arriba','Selecione e copie o código acima'],
    saved:['View saved','Vue enregistrée','视图已保存','Vista guardada','Visualização salva'],
    wait:['Wait for generation to finish before sharing','Attendez la fin de la génération avant de partager','请等待生成完成后再分享','Espera a que termine la generación antes de compartir','Aguarde a geração terminar antes de compartilhar'],
    removePin:['Remove pin at','Supprimer le repère à','删除此位置的标记：','Quitar marcador en','Remover marcador em'],
  };
  const shareErrors = {
    'Invalid camera number':['Invalid camera number','Nombre de caméra invalide','相机数值无效','Número de cámara no válido','Número de câmera inválido'],
    'Paste an NE1 share code or a share link':['Paste an NE1 share code or a share link','Collez un code de partage NE1 ou un lien','请粘贴 NE1 分享代码或链接','Pega un código NE1 o un enlace','Cole um código NE1 ou um link'],
    'Invalid view field':['Invalid view field','Champ de vue invalide','视图字段无效','Campo de vista no válido','Campo de visualização inválido'],
    'Duplicate view field':['Duplicate view field','Champ de vue en double','视图字段重复','Campo de vista duplicado','Campo de visualização duplicado'],
    'Invalid share UID':['Invalid share UID','Identifiant de partage invalide','分享 UID 无效','UID de compartir no válido','UID de compartilhamento inválido'],
    'Invalid shared view':['Invalid shared view','Vue partagée invalide','分享视图无效','Vista compartida no válida','Visualização compartilhada inválida'],
    'Invalid globe camera':['Invalid globe camera','Caméra du globe invalide','地球仪相机无效','Cámara del globo no válida','Câmera do globo inválida'],
    'Invalid portable recipe':['Invalid portable recipe','Recette portable invalide','可移植配方无效','Receta portátil no válida','Receita portátil inválida'],
    'Invalid pin coordinates':['Invalid pin coordinates','Coordonnées de repère invalides','标记坐标无效','Coordenadas de marcador no válidas','Coordenadas de marcador inválidas'],
    'Invalid latent geometry':['Invalid latent geometry','Géométrie latente invalide','潜在几何无效','Geometría latente no válida','Geometria latente inválida'],
    'Sharing unavailable':['Sharing unavailable','Partage indisponible','分享不可用','Compartir no disponible','Compartilhamento indisponível'],
  };
  const amplificationText = {
    title:['Terrain amplification (experimental)','Amplification du terrain (expérimental)','地形增强（实验性）','Amplificación del terreno (experimental)','Amplificação do terreno (experimental)'],
    mode:['Mode','Mode','模式','Modo','Modo'], intensity:['Intensity','Intensité','强度','Intensidad','Intensidade'],
    off:['Classic (current refinement)','Classique (raffinement actuel)','经典（当前细化）','Clásico (refinamiento actual)','Clássico (refinamento atual)'],
    hybrid:['Hybrid (recommended)','Hybride (recommandé)','混合（推荐）','Híbrido (recomendado)','Híbrido (recomendado)'],
    phasor:['Phasor (Grenier 2024)','Phasor (Grenier 2024)','Phasor（Grenier 2024）','Phasor (Grenier 2024)','Phasor (Grenier 2024)'],
    mse:['MSE (experimental, Schott 2024)','MSE (expérimental, Schott 2024)','MSE（实验性，Schott 2024）','MSE (experimental, Schott 2024)','MSE (experimental, Schott 2024)'],
    before:['Before','Avant','之前','Antes','Antes'], after:['After','Après','之后','Después','Depois'],
    compare:['compare (B)','comparer (B)','比较 (B)','comparar (B)','comparar (B)'],
    cycle:['Before → Hybrid → Phasor → MSE (Shift+B)','Avant → Hybride → Phasor → MSE (Maj+B)','之前 → 混合 → Phasor → MSE（Shift+B）','Antes → Híbrido → Phasor → MSE (Mayús+B)','Antes → Híbrido → Phasor → MSE (Shift+B)'],
    hint:['Plausible procedural detail, not measured terrain.','Détail procédural plausible, pas un terrain mesuré.','合理的程序细节，并非实测地形。','Detalle procedural plausible, no terreno medido.','Detalhe procedural plausível, não terreno medido.'],
    unavailable:['unavailable','indisponible','不可用','no disponible','indisponível'],
    noModule:['module unavailable on this server','module non disponible sur ce serveur','此服务器上没有该模块','módulo no disponible en este servidor','módulo indisponível neste servidor'],
    disabled:['Amplification disabled.','Amplification désactivée.','增强已禁用。','Amplificación desactivada.','Amplificação desativada.'],
    experimental:['experimental','expérimental','实验性','experimental','experimental'],
  };
  const v2Text = {
    title:['Render v2 · satellite landscape','Render v2 · paysage satellite','Render v2 · 卫星地貌','Render v2 · paisaje satelital','Render v2 · paisagem de satélite'],
    season:['Season','Saison','季节','Estación','Estação'],
    diagnostic:['Diagnostic','Diagnostic','诊断','Diagnóstico','Diagnóstico'],
    none:['None · satellite landscape','Aucun · paysage satellite','无 · 卫星地貌','Ninguno · paisaje satelital','Nenhum · paisagem de satélite'],
    substrate:['Substrate','Substrat','基底','Sustrato','Substrato'],
    moisture:['Local moisture','Humidité locale','局部湿度','Humedad local','Umidade local'],
    vegetation:['Vegetation','Végétation','植被','Vegetación','Vegetação'],
    snow:['Snow and ice','Neige et glace','积雪与冰','Nieve y hielo','Neve e gelo'],
    hint:['One annual calendar for foliage, ground, snow and ice. Appearance model; no terrain regeneration. ','Un calendrier annuel pour la végétation, le sol, la neige et la glace. Modèle d’apparence, sans régénération du terrain. ','植被、地面、积雪和冰共用一个年度日历。外观模型，不重新生成地形。','Un calendario anual para el follaje, el suelo, la nieve y el hielo. Modelo visual; no regenera el terreno. ','Um calendário anual para a vegetação, o solo, a neve e o gelo. Modelo visual; não regenera o terreno. '],
    method:['Method and limitations','Méthode et limites','方法与局限','Método y limitaciones','Método e limitações'],
  };
  const mapToolsText = {
    cancel:['Cancel measurement','Annuler la mesure','取消测量','Cancelar medición','Cancelar medição'],
    measure:['Measure distance','Mesurer une distance','测量距离','Medir distancia','Medir distância'],
    area:['Visible area','Zone visible','可见区域','Área visible','Área visível'],
    dimensions:['width × height','largeur × hauteur','宽 × 高','ancho × alto','largura × altura'],
    distance:['Distance on map','Distance sur la carte','地图距离','Distancia en el mapa','Distância no mapa'],
    distanceEmpty:['Distance','Distance','距离','Distancia','Distância'],
    instruction:['Distance: click and drag · Esc to cancel','Distance : cliquer-glisser · Échap pour annuler','距离：点击并拖动 · Esc 取消','Distancia: haz clic y arrastra · Esc para cancelar','Distância: clique e arraste · Esc para cancelar'],
    elevation:['Cursor elevation','Altitude du curseur','光标海拔','Altitud del cursor','Altitude do cursor'],
    outside:['outside world','hors du monde','世界范围之外','fuera del mundo','fora do mundo'],
    waiting:['waiting for terrain…','en attente du terrain…','等待地形…','esperando el terreno…','aguardando o terreno…'],
    preview:['input preview','aperçu de l’entrée','输入预览','vista previa de entrada','prévia da entrada'],
  };
  const hydrologyText = {
    water:['Water','Eau','水系','Agua','Água'], rivers:['Rivers','Rivières','河流','Ríos','Rios'], close:['Close water settings','Fermer les réglages de l’eau','关闭水系设置','Cerrar ajustes de agua','Fechar configurações de água'],
    enabled:['Show water','Afficher l’eau','显示水系','Mostrar agua','Mostrar água'], month:['Flow month','Mois du débit','流量月份','Mes del caudal','Mês da vazão'], annual:['Annual mean','Moyenne annuelle','年均值','Media anual','Média anual'],
    diagnostic:['Display','Affichage','显示','Visualización','Exibição'], basins:['River basins','Bassins versants','流域','Cuencas fluviales','Bacias hidrográficas'], discharge:['Discharge','Débit','流量','Caudal','Vazão'], lowflow:['Climatological low flow','Étiage climatique','气候低流量','Caudal mínimo climático','Vazão mínima climática'], navigation:['Navigability estimate','Estimation de navigabilité','通航估计','Estimación de navegabilidad','Estimativa de navegabilidade'],
    density:['Stream density','Densité des cours d’eau','河流密度','Densidad de cursos de agua','Densidade de cursos d’água'], minArea:['Min catchment (km², 0 auto)','Surface minimale du bassin (km², 0 auto)','最小集水面积（平方千米，0 为自动）','Cuenca mínima (km², 0 automático)','Bacia mínima (km², 0 automático)'],
    widthScale:['Line width','Épaisseur des traits','线宽','Grosor de línea','Espessura da linha'], intermittent:['Seasonal and dry streams','Cours saisonniers et secs','季节性和干涸河流','Cursos estacionales y secos','Cursos sazonais e secos'],
    lakesHeading:['Lakes','Lacs','湖泊','Lagos','Lagos'], lakes:['Show lakes','Afficher les lacs','显示湖泊','Mostrar lagos','Mostrar lagos'], minLakeArea:['Min lake area (km²)','Surface minimale du lac (km²)','最小湖泊面积（平方千米）','Área mínima de lago (km²)','Área mínima do lago (km²)'], minLakeDepth:['Min lake depth (m)','Profondeur minimale du lac (m)','最小湖泊深度（米）','Profundidad mínima de lago (m)','Profundidade mínima do lago (m)'], smoothShores:['Smooth shorelines','Lisser les rives','平滑岸线','Suavizar orillas','Suavizar margens'],
    serverHeading:['Server work (slower)','Calcul serveur (plus lent)','服务器计算（较慢）','Cálculo en servidor (más lento)','Cálculo no servidor (mais lento)'], terrainDetail:['Detailed channel carving (slow)','Creusement détaillé des chenaux (lent)','详细河道刻蚀（慢）','Excavación detallada de cauces (lenta)','Escavação detalhada de canais (lenta)'], valleyFit:['Fit course to valley floor (slower)','Adapter le cours au fond de vallée (plus lent)','河道贴合谷底（较慢）','Ajustar el curso al fondo del valle (más lento)','Ajustar o curso ao fundo do vale (mais lento)'], smoothBanks:['Limit bend radius to 1.5 widths (slower)','Limiter le rayon des méandres à 1,5 largeur (plus lent)','将弯曲半径限制为 1.5 倍宽度（较慢）','Limitar el radio de curva a 1,5 anchuras (más lento)','Limitar o raio da curva a 1,5 larguras (mais lento)'],
    refine:['Local rivers refine as terrain loads.','Les rivières locales s’affinent avec le chargement du terrain.','随着地形加载，局部河流会逐渐细化。','Los ríos locales se refinan al cargar el terreno.','Os rios locais são refinados à medida que o terreno carrega.'], inspect:['Click a river or lake to inspect it.','Cliquez sur une rivière ou un lac pour l’inspecter.','点击河流或湖泊以查看详情。','Haz clic en un río o lago para inspeccionarlo.','Clique em um rio ou lago para inspecioná-lo.'], method:['Hydrology methods','Méthodes hydrologiques','水文方法','Métodos hidrológicos','Métodos de hidrologia'],
    hidden:['Hidden','Masqué','已隐藏','Oculto','Oculto'], preparing:['Preparing river network…','Préparation du réseau hydrographique…','正在准备河网…','Preparando la red fluvial…','Preparando a rede fluvial…'], loading:['River network loads with the map','Le réseau hydrographique se charge avec la carte','河网随地图加载','La red fluvial se carga con el mapa','A rede fluvial carrega com o mapa'],
    shown:['river reaches shown','tronçons de rivière affichés','个河段已显示','tramos de río visibles','trechos de rio exibidos'], of:['of','sur','/','de','de'], lakeCount:['lakes','lacs','个湖泊','lagos','lagos'], unavailable:['some fine detail unavailable','certains détails fins sont indisponibles','部分精细细节不可用','algunos detalles finos no están disponibles','alguns detalhes finos indisponíveis'],
    densityHelp:['Scales the automatic catchment threshold as 1/density^1.5; 0 keeps main rivers only. Display only.','Ajuste le seuil automatique de bassin selon 1/densité^1,5 ; 0 conserve les rivières principales. Affichage uniquement.','按 1/密度^1.5 调整自动集水阈值；0 仅保留主要河流。仅影响显示。','Ajusta el umbral automático de cuenca como 1/densidad^1,5; 0 conserva solo ríos principales. Solo visualización.','Ajusta o limite automático da bacia por 1/densidade^1,5; 0 mantém apenas rios principais. Apenas exibição.'],
    areaHelp:['Fixed display threshold instead of the zoom-dependent one; the mean flow must also match this catchment at 300 mm/yr of runoff.','Seuil d’affichage fixe plutôt que dépendant du zoom ; le débit moyen doit aussi correspondre à ce bassin avec un ruissellement de 300 mm/an.','使用固定显示阈值代替随缩放变化的阈值；平均流量也须符合该流域每年 300 毫米的径流量。','Umbral de visualización fijo en vez de dependiente del zoom; el caudal medio también debe corresponder a la cuenca con 300 mm/año de escorrentía.','Limite de exibição fixo em vez de dependente do zoom; a vazão média também deve corresponder à bacia com 300 mm/ano de escoamento.'],
    widthHelp:['Scales the cartographic stroke; resolved channels keep their physical width.','Ajuste le trait cartographique ; les chenaux résolus conservent leur largeur physique.','调整地图线条；已解析的河道保留物理宽度。','Ajusta el trazo cartográfico; los cauces resueltos conservan su anchura física.','Ajusta o traço cartográfico; os canais resolvidos mantêm sua largura física.'],
    intermittentHelp:['Dashed blue: seasonal; dashed brown: dry in the selected month or most of the year.','Bleu pointillé : saisonnier ; brun pointillé : sec au mois choisi ou presque toute l’année.','蓝色虚线：季节性；棕色虚线：所选月份或一年大部分时间干涸。','Azul discontinuo: estacional; marrón discontinuo: seco en el mes elegido o la mayor parte del año.','Azul tracejado: sazonal; marrom tracejado: seco no mês selecionado ou na maior parte do ano.'],
    lakeAreaHelp:['Display filter; routing and water budgets are unchanged.','Filtre d’affichage ; le routage et les bilans d’eau ne changent pas.','显示过滤；水流路径和水量平衡不变。','Filtro de visualización; el enrutamiento y los balances de agua no cambian.','Filtro de exibição; o roteamento e os balanços de água não mudam.'],
    lakeDepthHelp:['Display filter on the lake maximum depth.','Filtre d’affichage sur la profondeur maximale du lac.','按湖泊最大深度过滤显示。','Filtro de visualización según la profundidad máxima del lago.','Filtro de exibição pela profundidade máxima do lago.'],
    shoreHelp:['Rounds drawn shore corners; display only.','Arrondit les angles des rives dessinées ; affichage uniquement.','圆滑绘制的岸线转角；仅影响显示。','Redondea las esquinas de las orillas dibujadas; solo visualización.','Arredonda os cantos das margens desenhadas; apenas exibição.'],
    detailHelp:['Generates extra regional neural terrain and carves physical river channels into it; much slower','Génère du terrain neuronal régional supplémentaire et y creuse des chenaux physiques ; beaucoup plus lent','生成额外区域神经地形并刻蚀物理河道；慢得多','Genera terreno neuronal regional adicional y excava cauces físicos; mucho más lento','Gera terreno neural regional adicional e escava canais físicos; muito mais lento'],
    mainRiver:['Main river','Fleuve principal','主要河流','Río principal','Rio principal'], river:['River','Rivière','河流','Río','Rio'], stream:['Stream','Ruisseau','溪流','Arroyo','Riacho'], rill:['Rill','Ruisselet','细流','Reguero','Pequeno curso d’água'],
    connector:['through a lake','à travers un lac','穿过湖泊','a través de un lago','através de um lago'], mean:['mean','moyenne','平均','media','média'], lowest:['lowest month','mois le plus bas','最低流量月份','mes de caudal mínimo','mês de menor vazão'], upstream:['upstream','en amont','上游','aguas arriba','a montante'], basin:['basin','bassin','流域','cuenca','bacia'], navEstimate:['navigation estimate','estimation de navigabilité','通航估计','estimación de navegabilidad','estimativa de navegabilidade'],
    unresolved:['regime unresolved (no monthly variation','régime indéterminé (aucune variation mensuelle','流态未确定（无月度变化','régimen indeterminado (sin variación mensual','regime indeterminado (sem variação mensal'], reports:['backend reports','le serveur indique','后端报告','el servidor indica','o servidor informa'], lowMonths:['low-flow months','mois de faible débit','个低流量月份','meses de caudal bajo','meses de baixa vazão'], perennial:['perennial','permanent','常年','permanente','perene'], seasonal:['seasonal','saisonnier','季节性','estacional','sazonal'], ephemeral:['ephemeral','intermittent','短暂','efímero','efêmero'], below:['below','sous','低于','por debajo de','abaixo de'], inMonths:['in','en','在','en','em'], months:['months','mois','个月','meses','meses'],
    parentCatchment:['parent river watershed','bassin de la rivière principale','主河流流域','cuenca del río principal','bacia do rio principal'], catchment:['catchment','bassin versant','集水区','cuenca','bacia'], drainsTo:['drains to','s’écoule vers','流向','desemboca en','deságua em'],
    lake:['Lake','Lac','湖泊','Lago','Lago'], approximate:['approximate shore','rive approximative','近似岸线','orilla aproximada','margem aproximada'], surface:['surface','surface','水面','superficie','superfície'], maxDepth:['max depth','profondeur maximale','最大深度','profundidad máxima','profundidade máxima'], closedBasin:['closed basin, no outlet','bassin fermé, sans exutoire','闭合流域，无出口','cuenca cerrada, sin salida','bacia fechada, sem saída'], hasOutlet:['drains through an outlet','s’écoule par un exutoire','通过出口排水','desagua por una salida','escoa por uma saída'],
    canoe:['canoe','canoë','独木舟','canoa','canoa'], riverboat:['riverboat','bateau fluvial','内河船','embarcación fluvial','embarcação fluvial'], ship:['ship','navire','船舶','buque','navio'], noRoute:['no continuous route','aucun trajet continu','无连续航道','sin ruta continua','sem rota contínua'],
    outletSea:['sea','mer','海洋','mar','mar'], outletClosed:['closed basin','bassin fermé','闭合流域','cuenca cerrada','bacia fechada'], outletEdge:['map edge','bord de carte','地图边缘','borde del mapa','borda do mapa'], outletUnresolved:['unresolved','indéterminé','未确定','indeterminado','indeterminado'],
  };
  const modelText = {
    menu:['Neural models','Modèles neuronaux','神经模型','Modelos neuronales','Modelos neurais'],
    title:['Neural model comparison','Comparaison des NN','神经模型比较','Comparación de modelos neuronales','Comparação de modelos neurais'],
    hint:['Continuous map: choose each model separately.','Carte continue : choisissez chaque modèle séparément.','连续地图：分别选择各个模型。','Mapa continuo: elige cada modelo por separado.','Mapa contínuo: escolha cada modelo separadamente.'],
    runtime:['Inference engine','Moteur d’inférence','推理引擎','Motor de inferencia','Motor de inferência'],
    changeRuntime:['Switch engine','Changer de moteur','切换引擎','Cambiar de motor','Trocar de motor'],
    coarse:['Coarse model','Modèle coarse','粗层模型','Modelo coarse','Modelo coarse'], base:['Base model','Modèle base','基础模型','Modelo base','Modelo base'], decoder:['Decoder model','Modèle decoder','解码器模型','Modelo decoder','Modelo decoder'],
    apply:['Apply models','Appliquer les NN','应用模型','Aplicar modelos','Aplicar modelos'], reset:['Reset after Orogen','Reset après Orogen','在 Orogen 后重置','Reiniciar después de Orogen','Redefinir após Orogen'],
    resetHint:['Reset recomputes coarse, base and decoder. Orogen, seed and camera are retained.','Le reset recalcule coarse, base et decoder. Orogen, graine et caméra sont conservés.','重置会重新计算粗层、基础层和解码器。保留 Orogen、种子和相机。','El reinicio recalcula coarse, base y decoder. Se conservan Orogen, la semilla y la cámara.','A redefinição recalcula coarse, base e decoder. Orogen, semente e câmera são mantidos.'],
    tile:['40 × 40 km tile · 8k residual base · browser CPU / GPU','Tuile 40 × 40 km · base résiduel 8k · CPU / GPU navigateur','40 × 40 千米图块 · 8k 残差基础层 · 浏览器 CPU / GPU','Tesela de 40 × 40 km · base residual 8k · CPU / GPU del navegador','Bloco de 40 × 40 km · base residual 8k · CPU / GPU do navegador'],
    browserRuntime:['Active: Rust compiled to WASM, computing on the browser GPU. Base NF4 B75, 30 m minimum. Switching engines can change the terrain.','Actif : Rust compilé en WASM, calcul sur le GPU du navigateur. Base NF4 B75, 30 m minimum. Le changement de moteur peut modifier le terrain.','当前：Rust 编译为 WASM，在浏览器 GPU 上计算。基础层 NF4 B75，最小 30 米。切换引擎可能改变地形。','Activo: Rust compilado a WASM, cálculo en la GPU del navegador. Base NF4 B75, mínimo 30 m. Cambiar de motor puede modificar el terreno.','Ativo: Rust compilado em WASM, processamento na GPU do navegador. Base NF4 B75, mínimo de 30 m. Trocar o motor pode alterar o terreno.'],
    pythonRuntime:['Active: Python / PyTorch on CUDA. Default engine. Rendering WebGPU is independent of inference.','Actif : Python / PyTorch sur CUDA. Moteur par défaut. Le WebGPU de rendu est indépendant de l’inférence.','当前：Python / PyTorch 在 CUDA 上运行。默认引擎。渲染 WebGPU 与推理独立。','Activo: Python / PyTorch en CUDA. Motor predeterminado. El WebGPU de renderizado es independiente de la inferencia.','Ativo: Python / PyTorch em CUDA. Motor padrão. O WebGPU de renderização é independente da inferência.'],
    unbuilt:['to build','à construire','待构建','pendiente de compilar','a compilar'], wait:['Wait for generation to finish.','Attendez la fin de la génération.','请等待生成完成。','Espera a que termine la generación.','Aguarde a geração terminar.'],
    waitModels:['Wait for the current generation to finish.','Attendez la fin de la génération en cours.','请等待当前生成完成。','Espera a que termine la generación actual.','Aguarde a geração atual terminar.'], active:['Active','Actif','当前','Activo','Ativo'], applying:['Applying…','Application…','正在应用…','Aplicando…','Aplicando…'], applied:['Models applied','NN appliqués','模型已应用','Modelos aplicados','Modelos aplicados'], cacheReset:['Neural cache reset','Cache NN réinitialisé','神经缓存已重置','Caché neuronal reiniciada','Cache neural redefinido'],
    unavailable:['Models unavailable','Modèles indisponibles','模型不可用','Modelos no disponibles','Modelos indisponíveis'], refused:['Change refused','Changement refusé','更改被拒绝','Cambio rechazado','Alteração recusada'], transfer:['Unable to transfer the world','Transfert du monde impossible','无法传输世界','No se puede transferir el mundo','Não foi possível transferir o mundo'],
    runtimeConstraint:['The browser engine supports Orogen planets with Orogen erosion or no erosion.','Le moteur navigateur prend en charge les planètes Orogen avec érosion Orogen ou sans érosion.','浏览器引擎支持使用 Orogen 侵蚀或无侵蚀的 Orogen 行星。','El motor del navegador admite planetas Orogen con erosión Orogen o sin erosión.','O motor do navegador aceita planetas Orogen com erosão Orogen ou sem erosão.'],
    layerConstraint:['Choose a layer supported by WASM: Render, Relief, Biomes or Fantasy.','Choisissez une couche prise en charge par WASM : Render, Relief, Biomes ou Fantasy.','请选择 WASM 支持的图层：Render、地形、生物群系或奇幻。','Elige una capa compatible con WASM: Render, Relieve, Biomas o Fantasía.','Escolha uma camada compatível com WASM: Render, Relevo, Biomas ou Fantasia.'],
    lodConstraint:['WASM reaches 30 m: choose automatic or non-negative LOD.','WASM descend à 30 m : choisissez une LOD automatique ou positive.','WASM 最低为 30 米：请选择自动或非负 LOD。','WASM llega a 30 m: elige LOD automático o no negativo.','WASM chega a 30 m: escolha LOD automático ou não negativo.'],
    timeout:['The server is not responding yet. Reload the page to retry.','Le serveur ne répond pas encore. Rechargez la page pour réessayer.','服务器尚未响应。请重新加载页面重试。','El servidor aún no responde. Recarga la página para reintentar.','O servidor ainda não responde. Recarregue a página para tentar novamente.'],
    viewFailed:['Models applied, but the view could not be recomputed.','Modèles appliqués, mais la vue n’a pas pu être recalculée.','模型已应用，但无法重新计算视图。','Modelos aplicados, pero no se pudo recalcular la vista.','Modelos aplicados, mas não foi possível recalcular a visualização.'],
    wasmLabel:['Rust / WASM · browser WebGPU','Rust / WASM · WebGPU navigateur','Rust / WASM · 浏览器 WebGPU','Rust / WASM · WebGPU del navegador','Rust / WASM · WebGPU do navegador'],
  };
  const welcomeText = {
    title:['Enable GPU acceleration?','Activer l’accélération GPU ?','启用 GPU 加速？','¿Activar la aceleración GPU?','Ativar a aceleração por GPU?'],
    prompt:['Enable all GPU options for Orogen relief, propagation, erosion, climate and projection, plus map rendering?','Activer toutes les options GPU pour le relief Orogen, la propagation, l’érosion, le climat et la projection, ainsi que le rendu de la carte ?','为 Orogen 地形、传播、侵蚀、气候、投影及地图渲染启用所有 GPU 选项？','¿Activar todas las opciones GPU para el relieve Orogen, propagación, erosión, clima y proyección, además del renderizado del mapa?','Ativar todas as opções de GPU para relevo Orogen, propagação, erosão, clima e projeção, além da renderização do mapa?'],
    details:['Orogen acceleration uses NVIDIA CUDA. Map rendering uses WebGPU. You can adjust each option later in Settings and Rendering. Your choices are saved on this browser.','L’accélération Orogen utilise NVIDIA CUDA. Le rendu de la carte utilise WebGPU. Vous pouvez modifier chaque option plus tard dans Réglages et Rendu. Vos choix sont enregistrés dans ce navigateur.','Orogen 加速使用 NVIDIA CUDA。地图渲染使用 WebGPU。稍后可在设置和渲染中调整各选项。选择会保存在此浏览器中。','La aceleración de Orogen usa NVIDIA CUDA. El renderizado del mapa usa WebGPU. Puedes ajustar cada opción después en Ajustes y Renderizado. Tus elecciones se guardan en este navegador.','A aceleração do Orogen usa NVIDIA CUDA. A renderização do mapa usa WebGPU. Você pode ajustar cada opção depois em Configurações e Renderização. Suas escolhas são salvas neste navegador.'],
    cpu:['Use CPU options','Utiliser les options CPU','使用 CPU 选项','Usar opciones de CPU','Usar opções de CPU'], gpu:['Enable GPU acceleration','Activer l’accélération GPU','启用 GPU 加速','Activar aceleración GPU','Ativar aceleração por GPU'],
  };
  const normalize = value => {
    const base = String(value || '').toLowerCase().split(/[-_]/)[0];
    return languages.includes(base) ? base : 'en';
  };
  function initialLocale() {
    try { return normalize(localStorage.getItem(key) || navigator.language); }
    catch { return normalize(navigator.language); }
  }
  let currentLocale = initialLocale();
  window.NeuralEarthLocale = {
    locale() { return currentLocale; },
    text(name) { const words = name.startsWith('amplification.') ? amplificationText[name.slice(14)] : name.startsWith('v2.') ? v2Text[name.slice(3)] : name.startsWith('mapTools.') ? mapToolsText[name.slice(9)] : name.startsWith('hydrology.') ? hydrologyText[name.slice(10)] : name.startsWith('models.') ? modelText[name.slice(7)] : name.startsWith('welcome.') ? welcomeText[name.slice(8)] : shareText[name]; return words ? words[languages.indexOf(currentLocale)] : name; },
    error(message) { const words = shareErrors[message]; return words ? words[languages.indexOf(currentLocale)] : message; },
  };
  function install() {
    const header = document.querySelector('header');
    if (!header || document.getElementById('ui-language')) return;
    const select = document.createElement('select');
    select.id = 'ui-language';
    for (const [value, label] of [['en','English'],['fr','Français'],['zh','简体中文'],['es','Español'],['pt','Português']]) {
      select.add(new Option(label, value));
    }
    header.append(select);
    function updateLayer(locale) {
      const index = languages.indexOf(locale);
      const panel = document.getElementById('layerPanel');
      if (!panel) return;
      const selected = panel.querySelector('.layerChoices button[aria-pressed="true"]');
      const name = selected?.textContent || panel.querySelector('.selectedLayer')?.textContent;
      const display = panel.querySelector('.selectedLayer');
      if (display && name) display.textContent = name;
      const summary = panel.querySelector('summary');
      if (summary) { summary.title = layerLabel[index] + ' · ' + name; summary.setAttribute('aria-label', summary.title); }
    }
    function apply(locale) {
      currentLocale = locale;
      const fantasy = document.getElementById('fantasySettings');
      if (fantasy) {
        const index = languages.indexOf(locale);
        fantasy.querySelector('legend').textContent = fantasyText.title[index];
        for (const name of ['density','size','spacing']) {
          const input = fantasy.querySelector('#brush_'+name);
          if (input) { input.parentElement.firstChild.textContent = fantasyText[name][index]; input.setAttribute('aria-label',fantasyText[name][index]); }
        }
        const button = fantasy.querySelector('#fantasyExport'); if (button) button.textContent = fantasyText.export[index];
      }
      const forest = document.getElementById('render3dSettings');
      if (forest) {
        const index = languages.indexOf(locale);
        forest.querySelector('legend').textContent = forestText.title[index];
        for (const [id,name] of [['forestEnabled','enabled'],['forestDensity','density'],['forestClearings','clearings']]) {
          const input = forest.querySelector('#'+id), label = input?.closest('label');
          if (label?.firstChild?.nodeType === Node.TEXT_NODE) label.firstChild.textContent = forestText[name][index]+' ';
          label?.querySelectorAll('input').forEach(control => control.setAttribute('aria-label',forestText[name][index]));
        }
        forest.querySelector('small').textContent = forestText.hint[index];
      }
      const words = messages[locale];
      select.value = locale;
      select.setAttribute('aria-label', words.language);
      // These controls may have icon-only contents added by TerrainToolbar.
      for (const [id, name] of Object.entries({mapView:'map',globeView:'globe',minus:'zoomOut',plus:'zoomIn',fit:'fit',initialView:'initial',native:'native',measureDistance:'measure',random:'random',apply:'generate'})) {
        const element = document.getElementById(id);
        if (!element) continue;
        element.title = words[name];
        element.setAttribute('aria-label', words[name]);
        // The atlas theme puts visible labels in a second span after the icon.
        const visibleLabel = element.querySelector('span:not(.atlasIcon)');
        if (visibleLabel && !visibleLabel.id) visibleLabel.textContent = words[name];
      }
      document.getElementById('seed')?.setAttribute('aria-label', words.seed);
      const help = header.querySelector('.help');
      if (help) help.textContent = words.help;
      const loading = document.getElementById('loadingLabel');
      if (loading) loading.textContent = words.loading;
      for (const [id, name] of [['renderPanel','rendering'],['runtimePanel','tools']]) {
        const summary = document.querySelector('#' + id + ' > summary');
        if (summary) {
          summary.title = words[name]; summary.setAttribute('aria-label', words[name]);
          const visibleLabel = summary.querySelector('span:not(.atlasIcon)');
          if (visibleLabel) visibleLabel.textContent = words[name];
        }
      }
      document.querySelectorAll('.panelClose').forEach(button => button.setAttribute('aria-label', words.close));
      const index = languages.indexOf(locale);
      for (const [selector, name] of Object.entries({
        '#runtimePanel .panelHeading strong':'worldPreparation', '#prepareWorld':'prepare', '#stopPreparation':'pause',
        '#runtimePanel .toolPanel > p':'preparationHint', '#renderPanel fieldset > legend':'loadRefine', '#mapDiagnostics > summary':'mapDetails',
      })) {
        const element = document.querySelector(selector);
        if (element) element.textContent = panelText[name][index];
      }
      const renderHeading = document.querySelector('#renderPanel .panelHeading strong');
      if (renderHeading) renderHeading.textContent = words.rendering;
      for (const [id, name] of Object.entries({prepareEntireWorld:'prepareEntire',prefetch:'prefetch',refinementDepth:'depth',renderSea:'renderSea',seaMaxLod:'seaMax',cacheBudgetGiB:'cacheBudget',cacheLodGap:'cacheGap',gpuRender:'gpuRender',coarseGpu:'coarseGpu'})) {
        const label = document.getElementById(id)?.closest('label');
        const node = label?.firstChild;
        if (node && node.nodeType === Node.TEXT_NODE) node.textContent = panelText[name][index] + ' ';
      }
      for (const heading of document.querySelectorAll('#layerPanel .layerGroups fieldset > legend')) {
        if (!heading.dataset.localeSource) heading.dataset.localeSource = heading.textContent;
        const translations = layerSections[heading.dataset.localeSource];
        if (translations) heading.textContent = translations[index];
      }
      for (const button of document.querySelectorAll('#layerPanel .layerChoices button')) {
        const translations = layerNames[button.dataset.mode];
        if (translations) {
          if (!button.dataset.localeTitle) button.dataset.localeTitle = button.title;
          button.textContent = translations[index];
          // Preserve details such as units and the wind color encoding until
          // their complete descriptions have also been translated.
          const original = button.dataset.localeTitle;
          button.title = locale === 'en' || original === translations[index] ? original : `${translations[index]} · ${original}`;
        }
      }
      const layerHeading = document.querySelector('#layerPanel .panelHeading strong');
      if (layerHeading) layerHeading.textContent = layerLabel[index];
      updateLayer(locale);
      for (const [id, name, heading] of [['generationPanel','generation','generationHeading'],['climatePanel','climate','climate'],['noisePanel','noise','noise']]) {
        const summary = document.querySelector('#' + id + ' > summary');
        if (summary) {
          summary.title = generationText[name][index]; summary.setAttribute('aria-label', summary.title);
          const visibleLabel = summary.querySelector('span:not(.atlasIcon):not(#generationSummary)');
          if (visibleLabel) visibleLabel.textContent = summary.title;
        }
        const title = document.querySelector('#' + id + ' .panelHeading strong');
        if (title) title.textContent = generationText[heading][index];
      }
      for (const [selector, name] of Object.entries({'#sourceSettings > legend':'worldSource','#worldGeometry > legend':'worldCreation','#proceduralParameters > legend':'sourceNoise'})) {
        const element = document.querySelector(selector);
        if (element) element.textContent = generationText[name][index];
      }
      for (const [selector, name] of Object.entries({
        '#orogenParameters > legend':'reliefOptions', '#erosionSettings > legend':'erosionHeading',
        '#erosionSettings > small':'erosionHelp', '#cityErosionParameters > summary':'cityGpu',
        '#cityErosionParameters > small':'cityHelp', '#orogenGpuParameters > summary':'gpuHeading',
        '#orogenGpuParameters > small':'gpuHelp', '#continentPresetHelp':'presetHelp',
        '#climateIntro':'climateIntro', '#neuralRefinement > legend':'noise',
        '#neuralRefinement > small':'snrHelp', '#adaptiveSnr > summary':'adaptiveSnr',
        '#lodSnrSettings > summary':'lodSnr', '#lodSnrSettings > small':'lodHelp',
        '#proceduralParameters > small':'proceduralHelp',
      })) {
        const element = document.querySelector(selector);
        if (element) element.textContent = generationText[name][index];
      }
      for (const [id, name] of Object.entries({generatorType:'generator',customSource:'customSource',worldTopology:'geometry',worldDiameter:'diameter',continentalStyle:'continentPreset',continentalStrength:'continentStrength',macroScale:'smoothing',dropWater:'landSea',reliefPipeline:'erosion',sourceChannel:'channel'})) {
        const control = document.getElementById(id);
        const label = control?.closest('label');
        if (label?.firstChild?.nodeType === Node.TEXT_NODE) label.firstChild.textContent = generationText[name][index] + ' ';
        if (control && control.hasAttribute('aria-label')) control.setAttribute('aria-label', generationText[name][index]);
      }
      for (const [value, name] of [['sphere','sphere'],['plane','plane']]) {
        const option = document.querySelector('#worldTopology option[value="' + value + '"]');
        if (option) option.textContent = generationText[name][index];
      }
      const geometryHelp = document.querySelector('#worldGeometry > small');
      if (geometryHelp) geometryHelp.textContent = generationText.geometryHelp[index];
      for (const [id, values] of Object.entries(generationOptions)) {
        const select = document.getElementById(id);
        if (!select) continue;
        for (const option of select.options) if (values[option.value]) option.textContent = values[option.value][index];
      }
      const material = document.getElementById('renderMaterialSettings');
      if (material) {
        const legend = material.querySelector('legend'); if (legend) legend.textContent = materialText.title[index];
        const hint = material.querySelector(':scope > small'); if (hint) hint.textContent = materialText.hint[index];
        const reset = document.getElementById('renderMaterialReset'); if (reset) reset.textContent = materialText.reset[index];
        for (const name of ['season','variation','forest','moisture','rock_slope','snow','vegetation_tint','rock_tint','snow_color']) {
          const input = document.getElementById('render_' + name);
          const label = input?.closest('label');
          if (label?.firstChild?.nodeType === Node.TEXT_NODE) label.firstChild.textContent = materialText[name][index];
          label?.querySelectorAll('input').forEach(control => control.setAttribute('aria-label', materialText[name][index]));
        }
      }
      for (const [selector, name] of [['#applyGeneration, [data-shared-action="applyGeneration"]','apply'],['#resetGeneration, [data-shared-action="resetGeneration"]','reset']]) {
        document.querySelectorAll(selector).forEach(button => { button.textContent = generationText[name][index]; });
      }
      const share = document.getElementById('sharePanel');
      if (share) {
        const summary = share.querySelector('summary');
        if (summary) {
          summary.title = shareText.share[index]; summary.setAttribute('aria-label', summary.title);
          const visible = summary.querySelector('span:not(.atlasIcon)');
          if (visible) visible.textContent = shareText.share[index];
        }
        const title = share.querySelector('.panelHeading strong'); if (title) title.textContent = shareText.title[index];
        const hint = share.querySelector('.toolPanel > p'); if (hint) hint.textContent = shareText.hint[index];
        const uid = document.getElementById('generationUid')?.closest('label')?.firstChild;
        if (uid?.nodeType === Node.TEXT_NODE) uid.textContent = shareText.uid[index] + ' ';
        for (const [selector, name] of [['#copyShare','copyCode'],['#copyShareLink','copyLink'],['label[for="openShareCode"]','openLabel'],['#openShare','open'],['#unlockShareSize','fullWindow']]) {
          const element = share.querySelector(selector); if (element) element.textContent = shareText[name][index];
        }
        document.getElementById('shareCode')?.setAttribute('aria-label', shareText.code[index]);
        document.getElementById('openShareCode')?.setAttribute('aria-label', shareText.openLabel[index]);
      }
    }
    apply(initialLocale());
    select.addEventListener('change', () => {
      const locale = normalize(select.value);
      try { localStorage.setItem(key, locale); } catch { /* Storage can be unavailable. */ }
      apply(locale);
      window.dispatchEvent(new Event('neural-locale-change'));
    });
    document.getElementById('mapMode')?.addEventListener('change', () => updateLayer(normalize(select.value)));
    window.addEventListener('storage', event => { if (event.key === key) { apply(normalize(event.newValue)); window.dispatchEvent(new Event('neural-locale-change')); } });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install, {once:true});
  else install();
})();
