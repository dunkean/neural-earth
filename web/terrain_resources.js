/* Resource layers: legend, season control, GPU packages and cell inspector.
 * Tiles are physical tiles shaded by the WGSL evaluator (terrain_resources_gpu.js);
 * this module loads one package per world and mode and explains the view.
 * Inspector values are those of the LOD-10 authority cell. */
window.TerrainResources=(()=>{
  const PREFIX='resources-';
  let specs={},panel=null,body=null,inspector=null,seasonRow=null,progress=null,ticket=0,loadTicket=0;
  let renderer=null,redraw=()=>{},world=null,packages=new Map(),current=null,season=.5,lastLoadMs=0,lastInspection=null;
  const languages=['fr','en','zh','es','pt'];
  const locale=()=>{try{const value=(localStorage.getItem('neural-earth-locale-v1')||navigator.language||'en').toLowerCase().split(/[-_]/)[0];return languages.includes(value)?value:'en';}catch{return 'en';}};
  const t=(fr,en,zh,es,pt)=>({fr,en,zh,es,pt})[locale()];
  const translated={
    "Abattage et débardage possibles": ["Felling and timber extraction possible", "可采伐与集材", "Tala y desembosque posibles", "Abate e extração de madeira possíveis"],
    "Arc de subduction": ["Subduction arc", "俯冲弧", "Arco de subducción", "Arco de subducção"],
    "Ardoise et schiste · potentiel": ["Slate and schist · potential", "板岩与片岩 · 潜力", "Pizarra y esquisto · potencial", "Ardósia e xisto · potencial"],
    "Argile · potentiel": ["Clay · potential", "粘土·潜力", "Arcilla · potencial", "Argila · potencial"],
    "Argilo-marneux": ["Clay-marl", "泥灰岩", "Marga arcillosa", "Argila-marga"],
    "Basalte et volcanites": ["Basalt and volcanic rocks", "玄武岩及火山岩", "Basalto y rocas volcánicas", "Basalto e rochas vulcânicas"],
    "Bassin arrière-arc": ["Back-arc basin", "弧后盆地", "Cuenca de retroarco", "Bacia de retroarco"],
    "Bois · accroissement annuel": ["Wood · annual increment", "林木 · 年生长量", "Madera · incremento anual", "Madeira · incremento anual"],
    "Bois · stock sur pied initial": ["Wood · initial standing volume", "林木 · 初始立木蓄积量", "Madera · existencias iniciales en pie", "Madeira · volume inicial em pé"],
    "Calcaire": ["Limestone", "石灰石", "piedra caliza", "Calcário"],
    "Charbon": ["Coal", "煤炭", "Carbón", "Carvão"],
    "Chaux (calcaire) · potentiel": ["Lime (limestone) · potential", "石灰（石灰石）·潜力", "Cal (piedra caliza) · potencial", "Cal (calcário) · potencial"],
    "Collision continentale": ["Continental collision", "大陆碰撞", "colisión continental", "Colisão continental"],
    "Contexte tectonique · composition": ["Tectonic context composition", "构造背景组成", "Composición del contexto tectónico.", "Composição do contexto tectônico"],
    "Contexte · Arc de subduction": ["Context · Subduction arc", "构造背景 · 俯冲弧", "Contexto · Arco de subducción", "Contexto · Arco de subducção"],
    "Contexte · Bassin arrière-arc": ["Context · Back-arc basin", "构造背景 · 弧后盆地", "Contexto · Cuenca de retroarco", "Contexto · Bacia de retroarco"],
    "Contexte · Collision continentale": ["Context · Continental collision", "构造背景 · 大陆碰撞", "Contexto · colisión continental", "Contexto · Colisão continental"],
    "Contexte · Craton": ["Context · Craton", "构造背景 · 克拉通", "Contexto · Cratón", "Contexto · Cráton"],
    "Contexte · Domaine océanique": ["Context · Oceanic domain", "构造背景 · 海洋领域", "Contexto · Dominio oceánico", "Contexto · Domínio oceânico"],
    "Contexte · Marge passive": ["Context · Passive margin", "构造背景 · 被动大陆边缘", "Contexto · Margen pasivo", "Contexto · Margem passiva"],
    "Contexte · Ophiolite / croûte océanique obductée": ["Context · Ophiolite / obducted oceanic crust", "构造背景 · 蛇绿岩／仰冲洋壳", "Contexto · Ofiolita / corteza oceánica obducida", "Contexto · Ofiolito / crosta oceânica obductada"],
    "Contexte · Plate-forme sédimentaire": ["Context · Sedimentary platform", "构造背景 · 沉积台地", "Contexto · Plataforma sedimentaria", "Contexto · Plataforma sedimentar"],
    "Contexte · Point chaud": ["Context · Hotspot", "构造背景 · 地幔热点", "Contexto · Punto caliente", "Contexto · Ponto quente"],
    "Contexte · Rift continental": ["Context · Continental rift", "构造背景 · 大陆裂谷", "Contexto · Rift continental", "Contexto · Rifte continental"],
    "Couvert ligneux potentiel (Render v2)": ["Potential woody cover (Render v2)", "潜在木本植物覆盖率（Render v2）", "Cobertura leñosa potencial (Render v2)", "Cobertura lenhosa potencial (Render v2)"],
    "Craton": ["Craton", "克拉通", "Cratón", "Cráton"],
    "Cu sédimentaire (30b)": ["Sedimentary Cu (30b)", "沉积铜 (30b)", "Cu sedimentario (30b)", "Cu sedimentar (30b)"],
    "Céréales · aptitude (Ecocrop × sols × relief)": ["Cereals · suitability (Ecocrop × soils × terrain)", "谷物 · 适宜性（Ecocrop × 土壤 × 地形）", "Cereales · aptitud (Ecocrop × suelos × relieve)", "Cereais · aptidão (Ecocrop × solos × relevo)"],
    "Céréales · classes GAEZ (NS, vmS, mS, MS, S, VS)": ["Cereals · GAEZ classes (NS, vmS, mS, MS, S, VS)", "谷物 · GAEZ 类别（NS、vmS、mS、MS、S、VS）", "Cereales · Clases GAEZ (NS, vmS, mS, MS, S, VS)", "Cereais · Classes GAEZ (NS, vmS, mS, MS, S, VS)"],
    "Céréales · production nette potentielle (assolement)": ["Cereals · potential net production (crop rotation)", "谷物·潜在净产量（轮作）", "Cereales · producción neta potencial (rotación de cultivos)", "Cereais · produção líquida potencial (rotação de culturas)"],
    "Céréales · rapport récolte/semence": ["Cereals · harvest/seed ratio", "谷物·收获/种子比", "Cereales · relación cosecha/semilla", "Cereais · relação colheita/sementes"],
    "Céréales · rendement brut médiéval": ["Cereals · medieval gross yield", "谷物 · 中世纪单位面积总产量", "Cereales · rendimiento bruto medieval", "Cereais · produtividade bruta medieval"],
    "Domaine océanique": ["Oceanic domain", "海洋领域", "Dominio oceánico", "Domínio oceânico"],
    "Eau · écoulement annuel": ["Water · annual runoff", "水 · 年径流量", "Agua · escorrentía anual", "Água · escoamento anual"],
    "Fer des marais · renouvellement": ["Bog iron · replenishment", "沼铁矿 · 再生量", "Hierro de pantano · renovación", "Ferro dos pântanos · renovação"],
    "Fer latéritique": ["Lateritic iron", "红土铁", "hierro laterítico", "Ferro laterítico"],
    "Fer oolithique (34f)": ["Oolitic iron (34f)", "鲕粒铁 (34f)", "Hierro oolítico (34f)", "Ferro oolítico (34f)"],
    "Fer rubané Superior/Algoma (34a, 28b)": ["Superior/Algoma banded iron formations (34a, 28b)", "苏必利尔／阿尔戈马型条带状铁建造 (34a, 28b)", "Formaciones de hierro bandeado Superior/Algoma (34a, 28b)", "Formações ferríferas bandadas Superior/Algoma (34a, 28b)"],
    "Filons polymétalliques Pb-Zn-Ag (22c)": ["Polymetallic Pb-Zn-Ag veins (22c)", "多金属铅锌银矿脉 (22c)", "Vetas polimetálicas de Pb-Zn-Ag (22c)", "Veios polimetálicos Pb-Zn-Ag (22c)"],
    "Fourrures · potentiel": ["Furs · potential", "皮草·潜力", "Pieles · potencial", "Peles · potencial"],
    "Gibier · densité (Damuth, 80 kg)": ["Game animals · density (Damuth, 80 kg)", "猎物动物 · 密度（Damuth，80 千克）", "Fauna cinegética · densidad (Damuth, 80 kg)", "Fauna cinegética · densidade (Damuth, 80 kg)"],
    "Gibier · prélèvement durable": ["Game animals · sustainable offtake", "猎物动物 · 可持续猎获量", "Fauna cinegética · extracción sostenible", "Fauna cinegética · abate sustentável"],
    "Gisements discrets (sites) · famille, tonnage, accessibilité": ["Discrete deposits · type, tonnage, accessibility", "离散矿床 · 类型、吨位、可采性", "Yacimientos discretos · tipo, tonelaje, accesibilidad", "Jazidas discretas · tipo, tonelagem, acessibilidade"],
    "Granite et roches plutoniques acides": ["Granite and acidic plutonic rocks", "花岗岩和酸性深成岩", "Granito y rocas plutónicas ácidas.", "Granito e rochas plutônicas ácidas"],
    "Grès": ["Sandstone", "砂岩", "arenisca", "Arenito"],
    "Hauteur de canopée (Render v2)": ["Canopy Height (Render v2)", "冠层高度（渲染 v2）", "Altura del dosel (Render v2)", "Altura do dossel (renderização v2)"],
    "MS 40–60 %": ["Moderate 40–60 %", "中等适宜性 40–60 %", "Moderada 40–60 %", "Moderada 40–60 %"],
    "Marge passive": ["Passive margin", "被动大陆边缘", "Margen pasivo", "Margem passiva"],
    "Mois de récolte (fin de la plus longue saison)": ["Harvest month (end of longest season)", "收获月（最长季节结束）", "Mes de la cosecha (fin de la temporada más larga)", "Mês de colheita (fim da época mais longa)"],
    "Métamorphique (schiste, gneiss, ardoise)": ["Metamorphic (schist, gneiss, slate)", "变质岩（片岩、片麻岩、板岩）", "Metamórfico (esquisto, gneis, pizarra)", "Metamórfico (xisto, gnaisse, ardósia)"],
    "NS": ["Unsuitable", "不适宜", "No apto", "Não apto"],
    "Neige au sol (Render v2)": ["Snow on the ground (Render v2)", "地面上的雪（渲染 v2）", "Nieve en el suelo (Render v2)", "Neve no chão (Render v2)"],
    "Neige pérenne (Render v2)": ["Perennial Snow (Render v2)", "常年积雪（渲染 v2）", "Nieve perenne (Render v2)", "Neve perene (renderização v2)"],
    "Ophiolite / croûte océanique obductée": ["Ophiolite / obducted oceanic crust", "蛇绿岩／仰冲洋壳", "Ofiolita / corteza oceánica obducida", "Ofiolito / crosta oceânica obductada"],
    "Or orogénique en quartz (36a)": ["Orogenic gold in quartz (36a)", "石英中的造山金 (36a)", "Oro orogénico en cuarzo (36a)", "Ouro orogênico em quartzo (36a)"],
    "Part bois d'œuvre du stock": ["Sawtimber share of standing volume", "立木蓄积中的用材比例", "Proporción de madera de aserrío en las existencias", "Proporção de madeira para serraria no estoque"],
    "Part résineuse (Render v2)": ["Conifer share (Render v2)", "针叶树比例（Render v2）", "Proporción de coníferas (Render v2)", "Proporção de coníferas (Render v2)"],
    "Pb-Zn type Mississippi (32a–b)": ["Mississippi Valley-type Pb–Zn (32a–b)", "密西西比河谷型铅锌矿床 (32a–b)", "Pb–Zn de tipo valle del Mississippi (32a–b)", "Pb–Zn do tipo vale do Mississippi (32a–b)"],
    "Pierre de taille · potentiel": ["Dimension stone · potential", "建筑石材 · 潜力", "Piedra de sillería · potencial", "Pedra de cantaria · potencial"],
    "Placers aurifères (39a)": ["Gold placers (39a)", "砂金矿床 (39a)", "Placeres auríferos (39a)", "Depósitos aluvionares de ouro (39a)"],
    "Placers stannifères (39e)": ["Tin placers (39e)", "砂锡矿床 (39e)", "Placeres de estaño (39e)", "Depósitos aluvionares de estanho (39e)"],
    "Plate-forme sédimentaire": ["Sedimentary platform", "沉积台地", "Plataforma sedimentaria", "Plataforma sedimentar"],
    "Point chaud": ["Hotspot", "地幔热点", "Punto caliente", "Ponto quente"],
    "Porphyre Cu-Au (Cox & Singer 17/20c)": ["Cu-Au Porphyry (Cox & Singer 17/20c)", "铜金斑岩 (Cox & Singer 17/20c)", "Pórfido Cu-Au (Cox & Singer 17/20c)", "Porfírio Cu-Au (Cox & Singer 17/20c)"],
    "Potentiel · argent (zones)": ["Potential · Silver (zones)", "潜力 · 银（区域）", "Potencial · Plata (zonas)", "Potencial · Prata (zonas)"],
    "Potentiel · charbon (zones)": ["Potential · Coal (zones)", "潜力 · 煤炭（区域）", "Potencial · Carbón (zonas)", "Potencial · Carvão (zonas)"],
    "Potentiel · cuivre (zones)": ["Potential · Copper (zones)", "潜力 · 铜（区域）", "Potencial · Cobre (zonas)", "Potencial · Cobre (zonas)"],
    "Potentiel · fer (zones)": ["Potential · Iron (zones)", "潜力 · 铁（区域）", "Potencial · Hierro (zonas)", "Potencial · Ferro (zonas)"],
    "Potentiel · gypse (zones)": ["Potential · Gypsum (zones)", "潜力 · 石膏（区域）", "Potencial · Yeso (zonas)", "Potencial · Gesso (zonas)"],
    "Potentiel · or (zones)": ["Potential · Gold (zones)", "潜力 · 金（区域）", "Potencial · Oro (zonas)", "Potencial · Ouro (zonas)"],
    "Potentiel · plomb (zones)": ["Potential · Lead (zones)", "潜力 · 铅（区域）", "Potencial · Plomo (zonas)", "Potencial · Chumbo (zonas)"],
    "Potentiel · sel (zones)": ["Potential · Salt (zones)", "潜力 · 盐（区域）", "Potencial · Sal (zonas)", "Potencial · Sal (zonas)"],
    "Potentiel · tungstène (zones)": ["Potential · Tungsten (zones)", "潜力 · 钨（区域）", "Potencial · Wolframio (zonas)", "Potencial · Tungstênio (zonas)"],
    "Potentiel · zinc (zones)": ["Potential · Zinc (zones)", "潜力 · 锌（区域）", "Potencial · Zinc (zonas)", "Potencial · Zinco (zonas)"],
    "Potentiel · étain (zones)": ["Potential · Tin (zones)", "潜力 · 锡（区域）", "Potencial · Estaño (zonas)", "Potencial · Estanho (zonas)"],
    "Production primaire mensuelle": ["Monthly primary production", "每月初级产量", "Producción primaria mensual", "Produção primária mensal"],
    "Production primaire nette (modèle Miami de Lieth)": ["Net primary production (Lieth’s Miami model)", "净初级生产（利斯的迈阿密模型）", "Producción primaria neta (modelo de Miami de Lieth)", "Produção primária líquida (modelo Miami de Lieth)"],
    "Prospectivité · Charbon (zones)": ["Prospectivity · Coal (zones)", "成矿远景 · 煤炭（区域）", "Prospectividad · Carbón (zonas)", "Prospectividade · Carvão (zonas)"],
    "Prospectivité · Cu sédimentaire (30b) (zones)": ["Prospectivity · Sedimentary Cu (30b) (zones)", "成矿远景 · 沉积铜 (30b)（区域）", "Prospectividad · Cu sedimentario (30b) (zonas)", "Prospectividade · Cu sedimentar (30b) (zonas)"],
    "Prospectivité · Fer latéritique (zones)": ["Prospectivity · Lateritic iron (zones)", "成矿远景 · 红土铁（区域）", "Prospectividad · hierro laterítico (zonas)", "Prospectividade · Ferro laterítico (zonas)"],
    "Prospectivité · Fer oolithique (34f) (zones)": ["Prospectivity · Oolitic iron (34f) (zones)", "成矿远景 · 鲕粒铁 (34f)（区域）", "Prospectividad · Hierro oolítico (34f) (zonas)", "Prospectividade · Ferro oolítico (34f) (zonas)"],
    "Prospectivité · Fer rubané Superior/Algoma (34a, 28b) (zones)": ["Prospectivity · Superior/Algoma banded iron formations (34a, 28b) (zones)", "成矿远景 · 苏必利尔／阿尔戈马型条带状铁建造 (34a, 28b)（区域）", "Prospectividad · Formaciones de hierro bandeado Superior/Algoma (34a, 28b) (zonas)", "Prospectividade · Formações ferríferas bandadas Superior/Algoma (34a, 28b) (zonas)"],
    "Prospectivité · Filons polymétalliques Pb-Zn-Ag (22c) (zones)": ["Prospectivity · Polymetallic Pb-Zn-Ag veins (22c) (zones)", "成矿远景 · 多金属铅锌银矿脉 (22c)（区域）", "Prospectividad · Vetas polimetálicas de Pb-Zn-Ag (22c) (zonas)", "Prospectividade · Veios polimetálicos Pb-Zn-Ag (22c) (zonas)"],
    "Prospectivité · Or orogénique en quartz (36a) (zones)": ["Prospectivity · Orogenic gold in quartz (36a) (zones)", "成矿远景 · 石英中的造山金 (36a)（区域）", "Prospectividad · Oro orogénico en cuarzo (36a) (zonas)", "Prospectividade · Ouro orogênico em quartzo (36a) (zonas)"],
    "Prospectivité · Pb-Zn type Mississippi (32a–b) (zones)": ["Prospectivity · Mississippi Valley-type Pb–Zn (32a–b) (zones)", "成矿远景 · 密西西比河谷型铅锌矿床 (32a–b)（区域）", "Prospectividad · Pb–Zn de tipo valle del Mississippi (32a–b) (zonas)", "Prospectividade · Pb–Zn do tipo vale do Mississippi (32a–b) (zonas)"],
    "Prospectivité · Placers aurifères (39a) (zones)": ["Prospectivity · Gold placers (39a) (zones)", "成矿远景 · 砂金矿床 (39a)（区域）", "Prospectividad · Placeres auríferos (39a) (zonas)", "Prospectividade · Depósitos aluvionares de ouro (39a) (zonas)"],
    "Prospectivité · Placers stannifères (39e) (zones)": ["Prospectivity · Tin placers (39e) (zones)", "成矿远景 · 砂锡矿床 (39e)（区域）", "Prospectividad · Placeres de estaño (39e) (zonas)", "Prospectividade · Depósitos aluvionares de estanho (39e) (zonas)"],
    "Prospectivité · Porphyre Cu-Au (Cox & Singer 17/20c) (zones)": ["Prospectivity · Cu-Au Porphyry (Cox & Singer 17/20c) (zones)", "成矿远景 · 铜金斑岩 (Cox & Singer 17/20c)（区域）", "Prospectividad · Pórfido Cu-Au (Cox & Singer 17/20c) (zonas)", "Prospectividade · Porfírio Cu-Au (Cox & Singer 17/20c) (zonas)"],
    "Prospectivité · Sel et gypse évaporitiques (35a) (zones)": ["Prospectivity · Evaporitic salt and gypsum (35a) (zones)", "成矿远景 · 蒸发盐和石膏 (35a)（区域）", "Prospectividad · Sal evaporítica y yeso (35a) (zonas)", "Prospectividade · Sal evaporativo e gesso (35a) (zonas)"],
    "Prospectivité · Skarn Fe-Cu (18b) (zones)": ["Prospectivity · Skarn Fe-Cu (18b) (zones)", "成矿远景 · 矽卡岩铁铜 (18b)（区域）", "Prospectividad · Skarn Fe-Cu (18b) (zonas)", "Prospectividade · Skarn Fe-Cu (18b) (zonas)"],
    "Prospectivité · Sn-W de granite, filons et greisens (15a–c) (zones)": ["Prospectivity · Sn-W of granite, veins and greisens (15a–c) (zones)", "成矿远景 · 花岗岩、矿脉和云英岩的 Sn-W (15a–c)（区域）", "Prospectividad · Sn-W de granito, vetas y greisens (15a–c) (zonas)", "Prospectividade · Sn-W de granito, veios e greisens (15a – c) (zonas)"],
    "Prospectivité · Sulfures massifs volcanogènes (24a, 28a) (zones)": ["Prospectivity · Volcanogenic massive sulphides (24a, 28a) (zones)", "成矿远景 · 火山成因块状硫化物 (24a, 28a)（区域）", "Prospectividad · Sulfuros masivos vulcanógenos (24a, 28a) (zonas)", "Prospectividade · Sulfetos maciços vulcanogênicos (24a, 28a) (zonas)"],
    "Prospectivité · Épithermal Au-Ag (25a–c) (zones)": ["Prospectivity · Epithermal Au–Ag (25a–c) (zones)", "成矿远景 · 浅成低温热液金银矿床 (25a–c)（区域）", "Prospectividad · Au–Ag epitermal (25a–c) (zonas)", "Prospectividade · Au–Ag epitérmico (25a–c) (zonas)"],
    "Pâture · capacité annuelle avec foin": ["Pasture · annual carrying capacity with hay", "牧场 · 含干草补饲的年载畜量", "Pasto · capacidad de carga anual con heno", "Pastagem · capacidade de suporte anual com feno"],
    "Pâture · capacité du mois": ["Pasture · monthly carrying capacity", "牧场 · 月载畜量", "Pasto · capacidad de carga mensual", "Pastagem · capacidade de suporte mensal"],
    "Pâture · capacité du mois le plus maigre": ["Pasture · carrying capacity in the leanest month", "牧场 · 饲草最少月份的载畜量", "Pasto · capacidad de carga del mes de mayor escasez", "Pastagem · capacidade de suporte no mês de maior escassez"],
    "Pâture · potentiel de transhumance": ["Pasture · transhumance potential", "牧场 · 季节性转场放牧潜力", "Pasto · potencial de trashumancia", "Pastagem · potencial de transumância"],
    "Pêche continentale (plaines inondables)": ["Inland fishing (floodplains)", "内陆捕鱼（洪泛区）", "Pesca continental (llanuras aluviales)", "Pesca interior (várzeas)"],
    "Pêche côtière (plateau < 200 m)": ["Coastal fisheries (shelf < 200 m)", "近海渔业（陆架水深 < 200 米）", "Pesca costera (plataforma continental < 200 m)", "Pesca costeira (plataforma continental < 200 m)"],
    "Pêche du mois": ["Monthly fish yield", "月渔获量", "Producción pesquera mensual", "Produção pesqueira mensal"],
    "Rift continental": ["Continental rift", "大陆裂谷", "Rift continental", "Rifte continental"],
    "Roche mère · composition (8 classes)": ["Parent rock composition (8 classes)", "母岩成分（8类）", "Composición de la roca madre (8 clases)", "Composição da rocha mãe (8 classes)"],
    "Roche · Argilo-marneux": ["Rock · Clay-marl", "岩石 · 泥灰岩", "Roca · Marga arcillosa", "Rocha · Argila-marga"],
    "Roche · Basalte et volcanites": ["Rock · Basalt and volcanic rocks", "岩石 · 玄武岩及火山岩", "Roca · Basalto y rocas volcánicas", "Rocha · Basalto e rochas vulcânicas"],
    "Roche · Calcaire": ["Rock · Limestone", "岩石 · 石灰石", "Roca · piedra caliza", "Rocha · Calcário"],
    "Roche · Granite et roches plutoniques acides": ["Rock · Granite and acidic plutonic rocks", "岩石 · 花岗岩和酸性深成岩", "Roca · Granito y rocas plutónicas ácidas.", "Rocha · Granito e rochas plutônicas ácidas"],
    "Roche · Grès": ["Rock · Sandstone", "岩石 · 砂岩", "Roca · arenisca", "Rocha · Arenito"],
    "Roche · Métamorphique (schiste, gneiss, ardoise)": ["Rock · Metamorphic (schist, gneiss, slate)", "岩石 · 变质岩（片岩、片麻岩、板岩）", "Roca · Metamórfico (esquisto, gneis, pizarra)", "Rocha · Metamórfico (xisto, gnaisse, ardósia)"],
    "Roche · Ultrabasique": ["Rock · Ultrabasic", "岩石 · 超基性岩", "Roca · Ultrabásica", "Rocha · Ultrabásica"],
    "Roche · Évaporites": ["Rock · Evaporites", "岩石 · 蒸发岩", "Roca · Evaporitas", "Rocha · Evaporitos"],
    "Récupérable médiéval · argent (gisements)": ["Recoverable medieval silver (deposits)", "可回收的中世纪白银（存款）", "Plata medieval recuperable (depósitos)", "Prata medieval recuperável (depósitos)"],
    "Récupérable médiéval · charbon (gisements)": ["Recoverable medieval coal (deposits)", "可回收的中世纪煤炭（矿床）", "Carbón medieval recuperable (depósitos)", "Carvão medieval recuperável (depósitos)"],
    "Récupérable médiéval · cuivre (gisements)": ["Recoverable medieval copper (deposits)", "可回收的中世纪铜（矿床）", "Cobre medieval recuperable (depósitos)", "Cobre medieval recuperável (depósitos)"],
    "Récupérable médiéval · fer (gisements)": ["Recoverable medieval iron (deposits)", "可回收的中世纪铁（矿床）", "Hierro medieval recuperable (depósitos)", "Ferro medieval recuperável (depósitos)"],
    "Récupérable médiéval · gypse (gisements)": ["Recoverable medieval · gypsum (deposits)", "可回收的中世纪·石膏（矿床）", "medieval recuperable · yeso (depósitos)", "Medieval recuperável · gesso (depósitos)"],
    "Récupérable médiéval · or (gisements)": ["Recoverable medieval gold (deposits)", "可回收的中世纪黄金（存款）", "Oro medieval recuperable (depósitos)", "Ouro medieval recuperável (depósitos)"],
    "Récupérable médiéval · plomb (gisements)": ["Recoverable medieval lead (deposits)", "可回收的中世纪铅（矿床）", "Plomo medieval recuperable (depósitos)", "Chumbo medieval recuperável (depósitos)"],
    "Récupérable médiéval · sel (gisements)": ["Salvageable medieval · salt (deposits)", "可抢救的中世纪·盐（存款）", "Salvable medieval · sal (depósitos)", "Medieval aproveitável · sal (depósitos)"],
    "Récupérable médiéval · tungstène (gisements)": ["Recoverable medieval tungsten (deposits)", "可回收的中世纪钨（矿床）", "Tungsteno medieval recuperable (depósitos)", "Tungstênio medieval recuperável (depósitos)"],
    "Récupérable médiéval · zinc (gisements)": ["Recoverable medieval zinc (deposits)", "可回收的中世纪锌（矿床）", "Zinc medieval recuperable (depósitos)", "Zinco medieval recuperável (depósitos)"],
    "Récupérable médiéval · étain (gisements)": ["Recoverable medieval · tin (deposits)", "可回收的中世纪·锡（矿床）", "medieval recuperable · estaño (depósitos)", "Medieval recuperável · estanho (depósitos)"],
    "S 60–80 %": ["Suitable 60–80 %", "适宜 60–80 %", "Apta 60–80 %", "Apta 60–80 %"],
    "Sables et graviers · potentiel": ["Sand and gravel · potential", "砂石·潜力", "Arena y grava · potencial", "Areia e cascalho · potencial"],
    "Saison de croissance (T ≥ 5 °C, ETR ≥ ½ ETP)": ["Growing season (T ≥ 5 °C, actual ET ≥ ½ potential ET)", "生长季（温度 ≥ 5 °C，实际蒸散量 ≥ 潜在蒸散量的一半）", "Temporada de crecimiento (T ≥ 5 °C, ETR ≥ ½ ETP)", "Estação de crescimento (T ≥ 5 °C, ETR ≥ ½ ETP)"],
    "Sel et gypse évaporitiques (35a)": ["Evaporitic salt and gypsum (35a)", "蒸发盐和石膏 (35a)", "Sal evaporítica y yeso (35a)", "Sal evaporativo e gesso (35a)"],
    "Sel marin du mois": ["Sea salt of the month", "当月海盐", "Sal marina del mes", "Sal marinho do mês"],
    "Sel marin solaire": ["Solar-evaporated sea salt", "日晒海盐", "Sal marina por evaporación solar", "Sal marinho por evaporação solar"],
    "Skarn Fe-Cu (18b)": ["Skarn Fe-Cu (18b)", "矽卡岩铁铜 (18b)", "Skarn Fe-Cu (18b)", "Skarn Fe-Cu (18b)"],
    "Sn-W de granite, filons et greisens (15a–c)": ["Sn-W of granite, veins and greisens (15a–c)", "花岗岩、矿脉和云英岩的 Sn-W (15a–c)", "Sn-W de granito, vetas y greisens (15a–c)", "Sn-W de granito, veios e greisens (15a – c)"],
    "Stock contenu · argent (gisements)": ["Stock content · money (deposits)", "库存内容·货币（存款）", "Contenido de existencias · dinero (depósitos)", "Conteúdo de estoque · dinheiro (depósitos)"],
    "Stock contenu · charbon (gisements)": ["Stock contained · coal (deposits)", "所含库存·煤炭（矿床）", "Stock contenido · carbón (depósitos)", "Estoque contido · carvão (depósitos)"],
    "Stock contenu · cuivre (gisements)": ["Stock contained · copper (deposits)", "所含库存·铜（矿床）", "Stock contenido · cobre (depósitos)", "Estoque contido · cobre (depósitos)"],
    "Stock contenu · fer (gisements)": ["Stock content · iron (deposits)", "库存含量·铁（存款）", "Contenido de existencias · hierro (depósitos)", "Conteúdo de estoque · ferro (depósitos)"],
    "Stock contenu · gypse (gisements)": ["Stock content · gypsum (deposits)", "库存含量·石膏（矿床）", "Contenido en stock · yeso (depósitos)", "Conteúdo de estoque · gesso (depósitos)"],
    "Stock contenu · or (gisements)": ["Stock contained · gold (deposits)", "所含库存·黄金（存款）", "Stock contenido · oro (depósitos)", "Estoque contido · ouro (depósitos)"],
    "Stock contenu · plomb (gisements)": ["Stock content · lead (deposits)", "库存含量·铅（矿床）", "Contenido de stock · plomo (depósitos)", "Conteúdo de estoque · lead (depósitos)"],
    "Stock contenu · sel (gisements)": ["Stock content · salt (deposits)", "库存含量·盐（沉积物）", "Contenido de stock · sal (depósitos)", "Conteúdo de estoque · sal (depósitos)"],
    "Stock contenu · tungstène (gisements)": ["Stock content · tungsten (deposits)", "库存含量·钨（矿床）", "Contenido en existencias · tungsteno (depósitos)", "Conteúdo de estoque · tungstênio (depósitos)"],
    "Stock contenu · zinc (gisements)": ["Stock content · zinc (deposits)", "库存含量·锌（矿床）", "Contenido en existencias · zinc (depósitos)", "Conteúdo de estoque · zinco (depósitos)"],
    "Stock contenu · étain (gisements)": ["Stock content · tin (deposits)", "库存含量·锡（存款）", "Contenido en existencias · estaño (depósitos)", "Conteúdo de estoque · estanho (depósitos)"],
    "Sulfures massifs volcanogènes (24a, 28a)": ["Volcanogenic massive sulphides (24a, 28a)", "火山成因块状硫化物 (24a, 28a)", "Sulfuros masivos vulcanógenos (24a, 28a)", "Sulfetos maciços vulcanogênicos (24a, 28a)"],
    "Terres arables potentielles · part de la cellule": ["Potential arable land · share of cell area", "潜在耕地 · 网格面积占比", "Tierras cultivables potenciales · proporción del área de la celda", "Terras aráveis potenciais · proporção da área da célula"],
    "Tourbe · part de la cellule": ["Peat · share of cell area", "泥炭 · 网格面积占比", "Turba · proporción del área de la celda", "Turfa · proporção da área da célula"],
    "Tourbe · volume": ["Peat volume", "泥炭量", "Volumen de turba", "Volume de turfa"],
    "UGB/km²": ["LSU/km²", "标准牲畜单位/平方公里", "UGM/km²", "UGB/km²"],
    "Ultrabasique": ["Ultrabasic", "超基性岩", "Ultrabásica", "Ultrabásica"],
    "VS ≥80 %": ["Very suitable ≥80 %", "高度适宜 ≥80 %", "Muy apta ≥80 %", "Muito apta ≥80 %"],
    "classe": ["class", "类别", "clase", "classe"],
    "fraction": ["fraction", "占比", "fracción", "fração"],
    "g MS/m²/an": ["g DM/m²/year", "克干物质/平方米/年", "g MS/m²/año", "g MS/m²/ano"],
    "g MS/m²/mois": ["g DM/m²/month", "克干物质/平方米/月", "g MS/m²/mes", "g MS/m²/mês"],
    "grain/grain": ["grain/grain", "谷物/谷物", "grano/grano", "grão/grão"],
    "ind./km²": ["individuals/km²", "个体/平方公里", "ind./km²", "ind./km²"],
    "indice 0–1": ["index 0–1", "指数 0–1", "índice 0–1", "índice 0–1"],
    "kg/km²": ["kg/km²", "公斤/平方公里", "kg/km²", "kg/km²"],
    "kg/km²/an": ["kg/km²/year", "公斤/平方公里/年", "kg/km²/año", "kg/km²/ano"],
    "kg/km²/mois": ["kg/km²/month", "公斤/平方公里/月", "kg/km²/mes", "kg/km²/mês"],
    "m": ["m", "米", "m", "m"],
    "mS 20–40 %": ["Marginal 20–40 %", "低适宜性 20–40 %", "Marginal 20–40 %", "Marginal 20–40 %"],
    "mm/an": ["mm/year", "毫米/年", "mm/año", "mm/ano"],
    "mm/mois": ["mm/month", "毫米/月", "mm/mes", "mm/mês"],
    "mois": ["month", "月", "mes", "mês"],
    "m³/ha": ["m³/ha", "立方米/公顷", "m³/ha", "m³/ha"],
    "m³/ha/an": ["m³/ha/year", "立方米/公顷/年", "m³/ha/año", "m³/ha/ano"],
    "m³/km²": ["m³/km²", "立方米/平方公里", "m³/km²", "m³/km²"],
    "symbole": ["symbol", "符号", "símbolo", "símbolo"],
    "t Fe/km²/an": ["t Fe/km²/year", "吨铁/平方公里/年", "t Fe/km²/año", "t Fe/km²/ano"],
    "t/ha": ["t/ha", "吨/公顷", "t/ha", "t/ha"],
    "t/km²": ["t/km²", "吨/平方公里", "t/km²", "t/km²"],
    "t/km²/an": ["t/km²/year", "吨/平方公里/年", "t/km²/año", "t/km²/ano"],
    "t/km²/mois": ["t/km²/month", "吨/平方公里/月", "t/km²/mes", "t/km²/mês"],
    "vmS <20 %": ["Very marginal <20 %", "极低适宜性 <20 %", "Muy marginal <20 %", "Muito marginal <20 %"],
    "Écoulement du mois": ["Monthly runoff", "月径流量", "Escorrentía mensual", "Escoamento mensal"],
    "Épithermal Au-Ag (25a–c)": ["Epithermal Au–Ag (25a–c)", "浅成低温热液金银矿床 (25a–c)", "Au–Ag epitermal (25a–c)", "Au–Ag epitérmico (25a–c)"],
    "Évaporites": ["Evaporites", "蒸发岩", "Evaporitas", "Evaporitos"],
    "derived": ["derived", "派生", "derivado", "derivado"],
    "synthetic": ["synthetic", "合成", "sintético", "sintético"],
    "derived+synthetic": ["derived + synthetic", "派生 + 合成", "derivado + sintético", "derivado + sintético"],
    "Tous les gisements": ["All deposits", "全部矿床", "Todos los yacimientos", "Todas as jazidas"],
    "Diagnostic": ["Diagnostics", "诊断", "Diagnóstico", "Diagnóstico"],
    "Fer": ["Iron", "铁", "Hierro", "Ferro"],
    "Cuivre": ["Copper", "铜", "Cobre", "Cobre"],
    "Étain": ["Tin", "锡", "Estaño", "Estanho"],
    "Plomb": ["Lead", "铅", "Plomo", "Chumbo"],
    "Zinc": ["Zinc", "锌", "Zinc", "Zinco"],
    "Argent": ["Silver", "银", "Plata", "Prata"],
    "Or": ["Gold", "金", "Oro", "Ouro"],
    "Sel": ["Salt", "盐", "Sal", "Sal"],
    "Gypse": ["Gypsum", "石膏", "Yeso", "Gesso"],
    "Tungstène": ["Tungsten", "钨", "Wolframio", "Tungstênio"],
    "Métal stellaire": ["Star metal", "星辰金属", "Metal estelar", "Metal estelar"],
    "Fer des marais": ["Bog iron", "沼铁矿", "Hierro de pantano", "Ferro dos pântanos"],
    "Petit fer sédimentaire": ["Small sedimentary iron deposit", "小型沉积铁矿床", "Pequeño yacimiento de hierro sedimentario", "Pequena jazida de ferro sedimentar"],
    "Petits filons Pb–Ag": ["Small Pb–Ag veins", "小型铅银矿脉", "Pequeñas vetas de Pb–Ag", "Pequenos veios de Pb–Ag"],
    "Petits filons Sn–W": ["Small Sn–W veins", "小型锡钨矿脉", "Pequeñas vetas de Sn–W", "Pequenos veios de Sn–W"],
    "Carrière de gypse": ["Gypsum quarry", "石膏采石场", "Cantera de yeso", "Pedreira de gesso"],
    "Source salée / saumure": ["Saline spring / brine", "盐泉／卤水", "Manantial salino / salmuera", "Nascente salina / salmoura"],
    "Petit affleurement de charbon": ["Small coal outcrop", "小型煤层露头", "Pequeño afloramiento de carbón", "Pequeno afloramento de carvão"],
    "tonnes récupérables": ["recoverable tonnes", "可采吨数", "toneladas recuperables", "toneladas recuperáveis"],
    "derived (tirages adressés)": ["derived (keyed sampling)", "推导（键控抽样）", "derivado (muestreo por claves)", "derivado (amostragem por chaves)"],
    "Prospectivité · Fer des marais (zones)": ["Prospectivity · Bog iron (zones)", "成矿远景 · 沼铁矿（区域）", "Prospectividad · Hierro de pantano (zonas)", "Prospectividade · Ferro dos pântanos (zonas)"],
    "Prospectivité · Petit fer sédimentaire (zones)": ["Prospectivity · Small sedimentary iron deposit (zones)", "成矿远景 · 小型沉积铁矿床（区域）", "Prospectividad · Pequeño yacimiento de hierro sedimentario (zonas)", "Prospectividade · Pequena jazida de ferro sedimentar (zonas)"],
    "Prospectivité · Petits filons Pb–Ag (zones)": ["Prospectivity · Small Pb–Ag veins (zones)", "成矿远景 · 小型铅银矿脉（区域）", "Prospectividad · Pequeñas vetas de Pb–Ag (zonas)", "Prospectividade · Pequenos veios de Pb–Ag (zonas)"],
    "Prospectivité · Petits filons Sn–W (zones)": ["Prospectivity · Small Sn–W veins (zones)", "成矿远景 · 小型锡钨矿脉（区域）", "Prospectividad · Pequeñas vetas de Sn–W (zonas)", "Prospectividade · Pequenos veios de Sn–W (zonas)"],
    "Prospectivité · Carrière de gypse (zones)": ["Prospectivity · Gypsum quarry (zones)", "成矿远景 · 石膏采石场（区域）", "Prospectividad · Cantera de yeso (zonas)", "Prospectividade · Pedreira de gesso (zonas)"],
    "Prospectivité · Source salée / saumure (zones)": ["Prospectivity · Saline spring / brine (zones)", "成矿远景 · 盐泉／卤水（区域）", "Prospectividad · Manantial salino / salmuera (zonas)", "Prospectividade · Nascente salina / salmoura (zonas)"],
    "Prospectivité · Petit affleurement de charbon (zones)": ["Prospectivity · Small coal outcrop (zones)", "成矿远景 · 小型煤层露头（区域）", "Prospectividad · Pequeño afloramiento de carbón (zonas)", "Prospectividade · Pequeno afloramento de carvão (zonas)"],
    "Fantasy · métal stellaire": ["Fantasy · star metal", "奇幻 · 星辰金属", "Fantasía · metal estelar", "Fantasia · metal estelar"],
    "Prospectivité · Fantasy · métal stellaire (zones)": ["Prospectivity · Fantasy · star metal (zones)", "成矿远景 · 奇幻 · 星辰金属（区域）", "Prospectividad · Fantasía · metal estelar (zonas)", "Prospectividade · Fantasia · metal estelar (zonas)"],
    "Potentiel · métal stellaire (zones)": ["Potential · Star metal (zones)", "潜力 · 星辰金属（区域）", "Potencial · Metal estelar (zonas)", "Potencial · Metal estelar (zonas)"],
    "kg": ["kg", "千克", "kg", "kg"],
    "t": ["t", "吨", "t", "t"],
    "km": ["km", "千米", "km", "km"],
    "km²": ["km²", "平方公里", "km²", "km²"],
    "proxy": ["proxy", "代理值", "indicador indirecto", "indicador indireto"]
  };
  const frenchStatus={derived:'dérivé',synthetic:'synthétique','derived+synthetic':'dérivé + synthétique','derived (tirages adressés)':'dérivé (tirages adressés)'};
  const label=(value,language=locale())=>language==='fr'?(frenchStatus[value]||value):(translated[value]?.[languages.indexOf(language)-1]||value);
  const hasTranslation=value=>Object.hasOwn(translated,value)&&translated[value].length===4&&translated[value].every(v=>typeof v==='string'&&v.trim());
  const esc=value=>String(value).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const fmt=value=>value==null?'—':Math.abs(value)>=1e4||(Math.abs(value)>0&&Math.abs(value)<1e-2)?Number(value).toExponential(2):Number(value).toLocaleString(locale(),{maximumFractionDigits:3});
  function mount(){
    if(panel)return;
    panel=document.createElement('div');panel.id='resourceLegend';panel.hidden=true;
    panel.style.cssText='max-width:360px;margin-bottom:8px;line-height:1.5;font-size:12px;pointer-events:auto';
    panel.innerHTML='<div data-resource-body></div><div data-resource-progress hidden style="margin:4px 0"><div style="height:4px;background:#ffffff22;border-radius:2px"><div style="height:4px;width:0;background:#9cc9ff;border-radius:2px;transition:width .4s"></div></div><small></small></div><label data-resource-season hidden style="display:block;margin-top:4px">Saison <input type="range" min="0" max="1" step="0.01" value="0.5" style="width:150px;vertical-align:middle"> <span></span></label><p data-resource-inspector style="margin:6px 0 0;white-space:pre-line">Cliquer près d’un symbole pour lire la ressource récupérable.</p><a href="/docs/RESOURCE_LAYERS.md" target="_blank" rel="noopener" style="color:#9cc9ff">Méthode des ressources</a>';
    body=panel.querySelector('[data-resource-body]');progress=panel.querySelector('[data-resource-progress]');inspector=panel.querySelector('[data-resource-inspector]');seasonRow=panel.querySelector('[data-resource-season]');
    inspector.style.maxHeight='min(40vh,280px)';inspector.style.overflowY='auto';
    inspector.dataset.empty='true';
    const input=seasonRow.querySelector('input');
    input.oninput=()=>{season=Number(input.value);syncSeason();renderer?.setResourceSeason?.(season);redraw();};
    (document.getElementById('hud')||document.body).prepend(panel);
    localizePanel();
  }
  function syncSeason(){const m=season*12-.5,i=((Math.round(m)%12)+12)%12;const month=new Intl.DateTimeFormat(locale(),{month:'short',timeZone:'UTC'}).format(new Date(Date.UTC(2020,i,1)));seasonRow.querySelector('span').textContent=month+' '+t('(calendrier nord)','(northern calendar)','（北半球日历）','(calendario norte)','(calendário norte)');seasonRow.querySelector('input').value=String(season);}
  function localizePanel(){if(!panel)return;seasonRow.firstChild.textContent=t('Saison ','Season ','季节 ','Estación ','Estação ');panel.querySelector('a').textContent=t('Méthode des ressources','Resource method','资源方法','Método de recursos','Método dos recursos');if(inspector.dataset.empty==='true')inspector.textContent=t('Cliquer sur la carte pour inspecter la cellule LOD 10 (30,72 km).','Click the map to inspect the LOD 10 cell (30.72 km).','点击地图以查看 LOD 10 网格（30.72 公里）。','Haz clic en el mapa para inspeccionar la celda LOD 10 (30,72 km).','Clique no mapa para inspecionar a célula LOD 10 (30,72 km).');else if(lastInspection)renderInspection(lastInspection);if(!panel.hidden&&specs[document.getElementById('mapMode')?.value])body.innerHTML=legend(specs[document.getElementById('mapMode').value]);syncSeason();
    const groups={geo:t('Ressources · géologie','Resources · geology','资源 · 地质','Recursos · geología','Recursos · geologia'),dep:t('Ressources · gisements','Resources · deposits','资源 · 矿床','Recursos · yacimientos','Recursos · jazidas'),agri:t('Ressources · agriculture et élevage','Resources · farming and livestock','资源 · 农业与畜牧','Recursos · agricultura y ganadería','Recursos · agricultura e pecuária'),bio:t('Ressources · forêt, faune, pêche, eau','Resources · forest, wildlife, fish, water','资源 · 森林、野生动物、渔业、水','Recursos · bosque, fauna, pesca, agua','Recursos · floresta, fauna, pesca, água'),mat:t('Ressources · matériaux, sel, tourbe','Resources · materials, salt, peat','资源 · 材料、盐、泥炭','Recursos · materiales, sal, turba','Recursos · materiais, sal, turfa'),diagnostic:label('Diagnostic'),season:t('Ressources · saisons','Resources · seasons','资源 · 季节','Recursos · estaciones','Recursos · estações')};
    for(const [id,spec] of Object.entries(specs)){const name=label(spec.label),option=document.querySelector('#mapMode option[value="'+id+'"]'),button=document.querySelector('#layerPanel button[data-mode="'+id+'"]');if(option)option.textContent=name;if(button){button.textContent=name;button.title=name;const heading=button.closest('fieldset')?.querySelector('legend'),group=spec.group;if(heading&&groups[group])heading.textContent=groups[group];}}
    const selected=document.getElementById('mapMode')?.value;if(selected&&specs[selected]){const name=label(specs[selected].label),node=document.querySelector('#layerPanel .selectedLayer');if(node)node.textContent=name;}
  }
  window.addEventListener('neural-locale-change',localizePanel);
  function legend(spec){
    const status=`<small>${t('Statut','Status','状态','Estado','Estado')} : ${esc(label(spec.status))}${spec.proxy?' · <b>'+esc(label('proxy'))+'</b> '+t('d’un champ de rendu (Render v2)','of a rendering field (Render v2)','渲染字段的代理（Render v2）','de un campo de renderizado (Render v2)','de um campo de renderização (Render v2)'):''} · ${t('détail calculé sur GPU','detail computed on GPU','细节由 GPU 计算','detalle calculado en GPU','detalhe calculado na GPU')}</small>`;
    if(spec.classes&&spec.kind!=='ramp'){
      const items=spec.classes.map(([name,color])=>`<span style="display:inline-block;margin-right:10px"><i style="display:inline-block;width:9px;height:9px;margin-right:4px;background:#${color};border:1px solid #0006"></i>${esc(label(name))}</span>`).join('');
      const note=spec.kind==='points'?`<br><small>${t('Couleur : famille. Aire ∝ 9 + 5 × log₁₀(tonnes récupérables), bornée à 10⁰–10⁸ t. Diamètres : 6 px (≤1 t), 9 px (100 t), 11 px (10 000 t), 14 px (≥100 Mt), hors contour de 1 px. Les grands sites passent devant. Fondu vers l’emprise réelle au zoom proche.','Colour: family. Area ∝ 9 + 5 × log₁₀(recoverable tonnes), clamped to 10⁰–10⁸ t. Diameters: 6 px (≤1 t), 9 px (100 t), 11 px (10,000 t), 14 px (≥100 Mt), plus a 1 px outline. Larger sites are drawn on top. Fades to the real footprint at close zoom.','颜色：矿床类型。面积 ∝ 9 + 5 × log₁₀(可开采吨数)，限定在 10⁰–10⁸ 吨。直径：6 像素（≤1 吨）、9 像素（100 吨）、11 像素（1 万吨）、14 像素（≥1 亿吨），另加 1 像素轮廓。较大的矿点绘制在上层。近距离缩放时过渡为真实范围。','Color: familia. Área ∝ 9 + 5 × log₁₀(toneladas recuperables), limitada a 10⁰–10⁸ t. Diámetros: 6 px (≤1 t), 9 px (100 t), 11 px (10 000 t), 14 px (≥100 Mt), más un contorno de 1 px. Los sitios grandes se dibujan delante. Transición a la huella real con zoom cercano.','Cor: família. Área ∝ 9 + 5 × log₁₀(toneladas recuperáveis), limitada a 10⁰–10⁸ t. Diâmetros: 6 px (≤1 t), 9 px (100 t), 11 px (10 000 t), 14 px (≥100 Mt), mais um contorno de 1 px. Os sítios maiores ficam à frente. Transição para a área real com zoom próximo.')}</small>`:spec.kind==='mix'?`<br><small>${spec.label.startsWith('Contexte')?t('Contexte dominant : les fractions sont des appartenances floues (distance aux frontières de plaques), contacts déformés.','Dominant context: fractions are fuzzy memberships based on distance to plate boundaries, with deformed contacts.','主导环境：比例为基于板块边界距离的模糊隶属度，接触带经过变形。','Contexto dominante: las fracciones son pertenencias difusas según la distancia a los límites de placas, con contactos deformados.','Contexto dominante: as frações são associações difusas conforme a distância aos limites de placas, com contatos deformados.'):t('Géologie réalisée : les mêmes unités à tous les zooms (provinces issues des contextes tectoniques, ceintures le long de la structure) ; les parts de chaque cellule de 30 km en sont mesurées.','Realized geology: the same units at all zoom levels (provinces from tectonic contexts, belts along structures); each 30 km cell’s shares are measured from them.','已实现地质：所有缩放级别使用相同单元（由构造环境形成的省区及沿构造分布的带）；每个 30 公里网格的比例由此测得。','Geología realizada: las mismas unidades en todos los niveles de zoom (provincias de contextos tectónicos y cinturones a lo largo de la estructura); se miden sus proporciones en cada celda de 30 km.','Geologia realizada: as mesmas unidades em todos os níveis de zoom (províncias de contextos tectônicos e cinturões ao longo da estrutura); suas proporções são medidas em cada célula de 30 km.')}</small>`:'';
      return `<strong>${esc(label(spec.label))}</strong><br>${items}${note}${spec.kind==='points'?`<br><small>${t('Regroupement à moins de 22 px : masse récupérable sommée, compteur de sites, couleur du membre le plus massif. Cliquer pour lire tous les membres ; le zoom les sépare.','Groups within 22 px: summed recoverable mass, site count, colour of the largest member. Click to list all members; zoom in to split them.','22 像素内聚合：可采质量求和，显示矿点数，以最大成员的颜色表示。点击查看全部成员；放大后逐步分开。','Agrupación a menos de 22 px: masa recuperable sumada, número de sitios y color del miembro de mayor masa. Haz clic para ver todos los miembros; amplía para separarlos.','Agrupamento a menos de 22 px: massa recuperável somada, número de sítios e cor do membro de maior massa. Clique para listar todos os membros; aproxime para separá-los.')}</small>`:''}<br>${status}`;
    }
    const stops=spec.colors.map((c,i)=>`#${c} ${Math.round(100*i/(spec.colors.length-1))}%`).join(',');
    return `<strong>${esc(label(spec.label))}</strong><div style="height:10px;margin:4px 0;background:linear-gradient(90deg,${stops})"></div>`+
      `<div style="display:flex;justify-content:space-between"><span>${fmt(spec.range[0])}</span><span>${esc(label(spec.unit))}${spec.log?' '+t('(échelle log)','(log scale)','（对数刻度）','(escala log)','(escala log)'):''}</span><span>${fmt(spec.range[1])}</span></div>${note(spec)}${status}`;
  }
  // Loading state: bar, stage and estimate; the relief map stays visible underneath.
  function showProgress(text,fraction,error){
    if(!progress)return;
    progress.hidden=text==null;if(text==null)return;
    progress.querySelector('small').textContent=text;progress.querySelector('small').style.color=error?'#ff9c9c':'';
    progress.firstElementChild.firstElementChild.style.width=Math.round(100*Math.max(.03,Math.min(1,fraction||0)))+'%';
  }
  const wait=ms=>new Promise(ok=>setTimeout(ok,ms));
  // What a mineral layer shows: a potential (zones) or the stock of the deposits.
  function note(spec){
    if(spec.group==='diagnostic')return `<small>${t('Diagnostic : intensité de placement (pondérée par la dotation typique pour un métal). Ce champ sert à tirer les sites ; ce n’est pas une carte des ressources présentes.','Diagnostic: placement intensity (weighted by typical endowment for a metal). This field is used to draw sites; it is not a map of the resources present.','诊断：布点强度（按金属的典型禀赋加权）。此字段用于抽取矿点，并非现有资源分布图。','Diagnóstico: intensidad de colocación (ponderada por la dotación típica de un metal). Este campo sirve para sortear los sitios; no es un mapa de los recursos presentes.','Diagnóstico: intensidade de posicionamento (ponderada pela dotação típica de um metal). Este campo serve para sortear os sítios; não é um mapa dos recursos presentes.')}</small><br>`;
    if(spec.label.includes('(gisements)'))return `<small>${t('Stock des gisements : emprises lobées suivant la structure et le relief régional ; le relief fin accentue les filons et fonds de vallée sans modifier la densité conservée. Noyaux normalisés par l’aire sphérique (tolérance numérique mesurée).','Deposit stock: lobed footprints follow structure and regional terrain; fine terrain emphasizes veins and valley floors without changing conserved density. Kernels are normalized by spherical area (measured numerical tolerance).','矿床存量：分叶范围遵循构造和区域地形；精细地形突出矿脉和谷底，不改变守恒密度。核按球面面积归一化（已测量数值容差）。','Existencias de yacimientos: huellas lobuladas siguen la estructura y el relieve regional; el relieve fino acentúa vetas y valles sin cambiar la densidad conservada. Núcleos normalizados por área esférica (tolerancia numérica medida).','Estoque das jazidas: áreas lobadas seguem a estrutura e o relevo regional; o relevo fino destaca veios e vales sem alterar a densidade conservada. Núcleos normalizados pela área esférica (tolerância numérica medida).')}</small><br>`;
    if(spec.label.startsWith('Potentiel'))return `<small>${t('Ceintures tectoniques et provinces fertiles : dotation typique par km² (densité × prospectivité × tonnage médian × teneur), indépendante du tirage des gisements. Échelle linéaire ; les faibles valeurs restent discrètes.','Tectonic belts and fertile provinces: typical endowment per km² (density × prospectivity × median tonnage × grade), independent of deposit sampling. Linear scale; low values remain subtle.','构造带和富矿区：每平方公里的典型禀赋（密度 × 成矿潜力 × 中位吨位 × 品位），独立于矿床抽样。线性刻度；低值保持不显著。','Cinturones tectónicos y provincias fértiles: dotación típica por km² (densidad × prospectividad × tonelaje mediano × ley), independiente del sorteo de yacimientos. Escala lineal; los valores bajos son discretos.','Cinturões tectônicos e províncias férteis: dotação típica por km² (densidade × prospectividade × tonelagem mediana × teor), independente do sorteio de jazidas. Escala linear; valores baixos permanecem discretos.')}</small><br>`;
    return '';
  }
  async function load(mode){
    if(!renderer?.setResources||!world)return;
    const key=world.seed+'|'+mode,mine=++loadTicket,started=performance.now();
    current=null;   // until this package is on the GPU
    // No package bound: the renderer keeps the relief map (not a blank layer).
    renderer.resetResources?.();renderer.markResourceTiles?.();redraw();
    try{
      let pkg=packages.get(key);
      while(!pkg){
        showProgress(t('Chargement de la couche…','Loading layer…','正在加载图层…','Cargando capa…','Carregando camada…'),.05);
        const r=await fetch(`/api/resources/${encodeURIComponent(world.seed)}/gpu?mode=${encodeURIComponent(mode)}&world_profile=${encodeURIComponent(world.generation_profile||'orogen')}`);
        if(mine!==loadTicket)return;
        if(r.status===202){            // world resources still compiling in the background
          const s=await r.json();
          if(s.state==='error')throw Error(s.error||t('échec du calcul des ressources','resource computation failed','资源计算失败','falló el cálculo de recursos','falha no cálculo dos recursos'));
          showProgress(t('Calcul des ressources du monde…','Computing world resources…','正在计算世界资源…','Calculando recursos del mundo…','Calculando recursos do mundo…')+(s.waiting_for?t(' — cette couche attend une étape suivante',' — this layer is waiting for a later stage',' — 此图层正在等待后续阶段',' — esta capa espera una etapa posterior',' — esta camada aguarda uma etapa posterior'):''),s.progress);
          await wait(1000);if(mine!==loadTicket)return;continue;
        }
        if(!r.ok)throw Error((await r.json()).error);
        const bytes=await r.arrayBuffer(),length=new DataView(bytes).getUint32(0,true);
        const header=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,length)));
        pkg={header,data:window.TerrainResourcesGPU.parse(header,bytes.slice(4+length))};
        packages.set(key,pkg);if(packages.size>6)packages.delete(packages.keys().next().value);
      }
      if(mine!==loadTicket)return;
      renderer.setResources(pkg.header,pkg.data,{season});current=mode;lastLoadMs=performance.now()-started;showProgress(null);redraw();
    }catch(error){if(mine===loadTicket){showProgress(t('Couche indisponible : ','Layer unavailable: ','图层不可用：','Capa no disponible: ','Camada indisponível: ')+error.message,1,true);}}
  }
  // Start the world's background compile as soon as the world opens.
  async function prepare(){
    if(!world)return;
    try{await fetch(`/api/resources/${encodeURIComponent(world.seed)}/status?world_profile=${encodeURIComponent(world.generation_profile||'orogen')}`);}catch(_){}
  }
  function sync(mode){
    mount();
    const spec=specs[mode],on=!!(mode&&mode.startsWith(PREFIX)&&spec);
    panel.hidden=!on;
    if(!on){current=null;return;}
    body.innerHTML=legend(spec);seasonRow.hidden=!spec.seasonal;localizePanel();
    if(current!==mode)load(mode);
  }
  function setWorld(data){
    world=data;specs=data?.resource_layers||{};packages.clear();current=null;lastInspection=null;if(inspector)inspector.dataset.empty='true';
    if(Object.keys(specs).length)prepare();
    for(const button of document.querySelectorAll('#layerPanel button[data-mode^="'+PREFIX+'"]'))button.hidden=!specs[button.dataset.mode];
    for(const option of document.querySelectorAll('#mapMode option[value^="'+PREFIX+'"]')){option.hidden=!specs[option.value];option.disabled=!specs[option.value];}
    sync(document.getElementById('mapMode')?.value);localizePanel();
  }
  function setRenderer(r,onRedraw){renderer=r;redraw=onRedraw||redraw;current=null;sync(document.getElementById('mapMode')?.value);}
  async function inspect({x,y,seed,profile,fetchJSON,mpp=30}){
    mount();if(panel.hidden)return false;
    const aggregation=renderer?.res?.aggregation,source=renderer?.res?.source;
    let cluster=null,memberCells=null,memberIndices=null;
    if(aggregation&&source?.deposits){
      const b=renderer.res.header.world_bounds,span=b?b[2]-b[0]:0;
      const distance=g=>{let dx=x-g.x;if(span)dx-=Math.floor(dx/span+.5)*span;return Math.hypot(dx,y-g.y);};
      const candidates=aggregation.groups.filter(g=>g.members.length>1&&distance(g)<=12*mpp).sort((a,b)=>distance(a)-distance(b));
      cluster=candidates[0];
      if(cluster){
        x=cluster.x;y=cluster.y;
        const cells=new Int32Array(source.deposits.buffer,source.deposits.byteOffset,source.deposits.length);
        memberCells=new Set(cluster.members.map(i=>`${cells[i*12]},${cells[i*12+1]}`));
        if(renderer.res.header.site_indices){const values=new Float32Array(source.deposits.buffer,source.deposits.byteOffset,source.deposits.length);memberIndices=new Set(cluster.members.map(i=>values[i*12+10]));}
      }
    }
    const mine=++ticket;lastInspection=null;inspector.dataset.empty='false';inspector.textContent=t('Lecture de la cellule…','Reading cell…','正在读取网格…','Leyendo celda…','Lendo célula…');
    try{
      const response=await (fetchJSON||fetch)(`/api/resources/${encodeURIComponent(seed)}/inspect?x=${x}&y=${y}&radius_m=${Math.max(1000,(cluster?23:9)*mpp)}&cluster=${cluster?1:0}&mode=${encodeURIComponent(document.getElementById('mapMode').value)}&world_profile=${encodeURIComponent(profile)}`);
      const data=await response.json();if(mine!==ticket)return true;
      if(response.status===202){inspector.textContent=t('Calcul des ressources du monde en cours…','World resource computation in progress…','世界资源计算进行中…','Cálculo de recursos del mundo en curso…','Cálculo dos recursos do mundo em andamento…');return true;}
      if(!response.ok)throw Error(data.error);
      if(cluster){data.cluster={count:cluster.members.length,mass:cluster.mass};data.deposits=data.deposits.filter(d=>memberIndices?memberIndices.has(d.site_index):memberCells.has(d.symbol_cell.join(',')));}
      lastInspection=data;renderInspection(data);
    }catch(error){if(mine===ticket)inspector.textContent=t('Inspection impossible : ','Inspection unavailable: ','无法查看：','Inspección no disponible: ','Inspeção indisponível: ')+error.message;}
    return true;
  }
  function renderInspection(data){
      const mode=document.getElementById('mapMode').value,value=data.values?.[mode];
      const lines=[`${t('Cellule LOD','LOD cell','LOD 网格','Celda LOD','Célula LOD')} ${data.cell.lod} (${data.cell.i}, ${data.cell.j}) · ${fmt(data.cell.area_km2)} km²`,
        `${t('Contexte dominant','Dominant context','主导环境','Contexto dominante','Contexto dominante')} : ${label(data.dominant.context)} · ${t('roche dominante','dominant rock','主导岩石','roca dominante','rocha dominante')} : ${label(data.dominant.rock)}`];
      if(data.cluster)lines.unshift(`${t('Groupe','Cluster','聚合矿点','Grupo','Grupo')} : ${fmt(data.cluster.count)} ${t('gisements','deposits','个矿床','yacimientos','jazidas')} · ${fmt(data.cluster.mass)} ${label('tonnes récupérables')}`);
      if(value)lines.push(`${label(value.label)} : ${fmt(value.value)} ${label(value.unit)}${value.proxy?' ('+label('proxy')+')':''}${specs[mode]?.seasonal?' '+t('(moyenne annuelle)','(annual average)','（年平均）','(media anual)','(média anual)'):''}`);
      const selected=mode.replace('resources-dep-','').replaceAll('-','_');
      const near=data.deposits.filter(d=>selected==='points'||!mode.startsWith('resources-dep-')||d.medieval_recoverable[selected]>0);
      if(selected==='zinc'||selected==='tungsten')lines.push(t('Quantité dans le minerai triable, pas de métal raffiné ; usage médiéval du tungstène provisoire.','Quantity in sortable ore, not refined metal; medieval use of tungsten is provisional.','可分选矿石中的含量，并非精炼金属；钨的中世纪用途为暂定。','Cantidad en el mineral seleccionable, no metal refinado; uso medieval del wolframio provisional.','Quantidade no minério selecionável, não metal refinado; uso medieval do tungstênio provisório.'));
      for(const d of data.cluster?near:near.slice(0,6)){
        const resources=Object.entries(d.medieval_recoverable).filter(([m,v])=>v>0&&(selected==='points'||!mode.startsWith('resources-dep-')||m===selected));
        for(const [m,v] of resources){
          const grade=d.contained[m]/(d.contained_units[m]==='kg'?1000:1)/d.ore_t;
          const name=label(specs['resources-dep-'+m.replaceAll('_','-')]?.label||m);
          lines.push(`• ${label(d.label)} · ${name} ${t('à','at','距离','a','a')} ${fmt(d.distance_m/1000)} km : ${fmt(v)} ${label(d.contained_units[m])} ${t('récupérables','recoverable','可开采','recuperables','recuperáveis')} ; ${t('teneur','grade','品位','ley','teor')} ${fmt(grade*100)} % ; ${t('profondeur','depth','深度','profundidad','profundidade')} ${fmt(d.depth_top_m)} m ; ${t('emprise','footprint','范围','huella','área')} ${d.footprint_m.map(fmt).join(' × ')} m`);
        }
      }
      if(!near.length)lines.push(t('Aucun gisement récupérable de cette ressource à proximité.','No recoverable deposit of this resource nearby.','附近没有该资源的可开采矿床。','Ningún yacimiento recuperable de este recurso cerca.','Nenhuma jazida recuperável deste recurso por perto.'));
      inspector.textContent=lines.join('\n');
  }
  return {PREFIX,label,hasTranslation,mount,sync,setWorld,setRenderer,inspect,specs:()=>specs,season:()=>season,
    state:()=>({current,loading:loadTicket,lastLoadMs,progress:progress&&!progress.hidden?progress.textContent:null,header:renderer?.res?.header?.mode||null,generation:renderer?.resGeneration||0,lastUploadMs:renderer?.lastResourceUploadMs||0,inspection:lastInspection?{cluster:lastInspection.cluster||null,members:lastInspection.deposits.length}:null})};
})();
document.getElementById('mapMode')?.addEventListener('change',event=>window.TerrainResources.sync(event.target.value));
