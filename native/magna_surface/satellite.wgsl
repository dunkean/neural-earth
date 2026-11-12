struct mu_TerrainParams {
    uQuad: vec4<f32>,
    uCanvas: vec2<f32>,
    uOrigin: vec2<f32>,
    uPx: f32,
    uDpr: f32,
    uMapW: f32,
    uFieldN: i32,
    uFieldOrigin: vec2<f32>,
    uFieldCell: f32,
    uMaxLod: i32,
    uAlpha: f32,
    uBase: i32,
    uHasCave: i32,
    uHasLake: i32,
    uLakeN: f32,
    uHasOverlay: i32,
    uUseWaterLayer: i32,
    uCityWater: vec2<f32>,
    uWaterVisible: i32,
    uRelief: i32,
    uPaper: vec3<f32>,
    uSea: vec3<f32>,
    uBackground: vec3<f32>,
    uFrameColor: vec3<f32>,
    uContourColor: vec3<f32>,
    uShore: vec3<f32>,
    uCoastalSea: i32,
    uSeaBand: f32,
    uCoastOrigin: vec2<f32>,
    uCoastWidth: f32,
    uCoastN: i32,
    uStops: array<vec3<f32>, 8>,
    uStopT: array<f32, 8>,
    uStopCount: i32,
    uLight: vec3<f32>,
    uLight2_: vec3<f32>,
    uExaggeration: f32,
    uFlatLight: f32,
    uRockStrength: f32,
    uShade: f32,
    uMinH: f32,
    uMaxH: f32,
    uGlobalMin: f32,
    uFloor: f32,
    uCoastal: i32,
    uFrame: i32,
    uContours: i32,
    uStep: f32,
    uThinW: f32,
    uIndexW: f32,
    uThinA: f32,
    uIndexA: f32,
    uFlatSlope: f32,
    uRefine: i32,
    uSourceCell: f32,
    uSeed: u32,
    uEnhanced3d: i32,
    uGroundCapture: i32,
    uNaturalWater: i32,
    uNight: i32,
    uRefinementMode: i32,
    uRefinedRect: vec4<f32>,
    uPreviousRefinedRect: vec4<f32>,
    uRefinedN: i32,
    uPreviousRefinedN: i32,
    uRefinedAlpha: f32,
    uDetailA: vec4<f32>,
    uDetailB: vec3<f32>,
    uOctaves: i32,
    uV2Season: f32,
    uV2WaterVisible: i32,
    uMaterialZone: i32,
    uV2Calendar: i32,
    uV2CalLeaf: vec4<f32>,
    uV2CalIce: vec4<f32>,
    uV2CalThaw: vec4<f32>,
    uV2CalClimate: vec4<f32>,
    uV2CalSnow: vec4<f32>,
    uForestTrees: vec4<f32>,
    uForestRect: vec3<f32>,
    uHasForestBank: i32,
    uForestBankDensity: f32,
    uHasOccupation: i32,
    uOccupationRect: vec3<f32>,
    uOccupationN: f32,
    uHasForestBeach: i32,
    uForestBeachRect: vec3<f32>,
    uQuality: i32,
    uMaterialSeed: u32,
    uMaterialClimate: vec2<f32>,
    uGroundCover: vec4<f32>,
    uHasLandDetail: i32,
    uLandDetailRect: vec3<f32>,
    uLandDetailN: f32,
    uHasAux: i32,
    uPaperMap: i32,
    uV2GlacierSources: array<vec4<f32>, 8>,
    uWaterStops: array<vec3<f32>, 8>,
    uWaterStopT: array<f32, 8>,
    uWaterStopCount: i32,
    uWaterSeason: vec2<f32>,
    uShoreRingRect: vec3<f32>,
}

struct mu_GroundMaterial {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    occlusion: f32,
}

struct mu_ForestStand {
    p: vec2<f32>,
    cover: f32,
    height: f32,
    needle: f32,
    tone: f32,
    kappa: f32,
    trop: f32,
    dry: f32,
    open: f32,
    forced: f32,
    allow: f32,
    hedge: f32,
    orchard: bool,
    rows: vec2<f32>,
}

struct mu_ForestPatch {
    base: vec2<i32>,
    centre: vec2<f32>,
    glade: vec3<f32>,
}

struct mu_ForestCrown {
    c: vec2<f32>,
    r: f32,
    h: f32,
    depth: f32,
    needle: bool,
    alive: bool,
    riparian: bool,
    edge: f32,
    id: u32,
}

struct mu_ForestHit {
    cover: f32,
    top: f32,
    second: f32,
    slope: vec2<f32>,
    rel: f32,
    local: vec2<f32>,
    r: f32,
    needle: bool,
    riparian: bool,
    edge: f32,
    id: u32,
    mu_level: i32,
    near: f32,
    ink: f32,
}

struct mu_ForestPrep {
    st: mu_ForestStand,
    fp: f32,
    lod: f32,
    standLod: f32,
    reveal: f32,
    openA: f32,
    openW: f32,
    forcedA: f32,
    hedgeA: f32,
    humid: f32,
    cold: f32,
    radius: f32,
    crownVis: f32,
    fringe: f32,
    galleryNear: f32,
    valid: bool,
}

struct mu_ForestCanopy {
    cover: f32,
    shade: f32,
    near: f32,
    rel: f32,
    slope: vec2<f32>,
    edge: f32,
    tint: f32,
    ink: f32,
    needle: bool,
    riparian: bool,
}

struct mu_V2Env {
    temperature: f32,
    potential: f32,
    moisture: f32,
    exposure: f32,
    mineral: f32,
    sand: f32,
    clay: f32,
    organic: f32,
    rock: f32,
    grass: f32,
    woody: f32,
    height: f32,
    needle: f32,
    leaf: f32,
    snow: f32,
    ice: f32,
    tone: f32,
    glacier: f32,
    world: vec2<f32>,
    slope: f32,
    footprint: f32,
    seed: u32,
}

struct mu_FragmentOutput {
    @location(0) mu_outColor: vec4<f32>,
}

const mu_ROT: array<f32, 12> = array<f32, 12>(0.3f, 1.1f, 2.2f, 0.7f, 2.9f, 1.7f, 0.1f, 2.5f, 1.4f, 2f, 0.5f, 2.7f);
const mu_ELEV: array<vec3<f32>, 6> = array<vec3<f32>, 6>(vec3<f32>(0f, 0.1254902f, 0.3764706f), vec3<f32>(0f, 0.627451f, 0.27450982f), vec3<f32>(1f, 0.9019608f, 0f), vec3<f32>(0.92156863f, 0.13725491f, 0.078431375f), vec3<f32>(0.5019608f, 0.5019608f, 0.5019608f), vec3(1f));
const mu_V2_ZONE_FOREST: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(1f, 0.12f, 0.45f, 0.25f), vec4<f32>(0.62f, 0.3f, 0.45f, 0f), vec4<f32>(1f, 0.55f, 0.55f, 0.15f), vec4<f32>(1f, 0.92f, 0.75f, 0.45f), vec4<f32>(1f, 0.9f, 0.85f, 0.05f), vec4<f32>(0.05f, 0.98f, 0.95f, 0.45f), vec4<f32>(0.22f, 0.2f, 0.5f, 0.08f), vec4<f32>(0.12f, 0f, 1f, 0f), vec4<f32>(0.42f, 0f, 1f, 0.1f), vec4<f32>(1f, 0f, 1f, 0.55f), vec4<f32>(0.55f, 0.08f, 0.7f, 1f));
const mu_V2_ZONE_SOIL: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0f, 0f, 0.05f, 1f), vec4<f32>(0.35f, 0f, -0.1f, 0.75f), vec4<f32>(0f, 0f, 0.05f, 1f), vec4<f32>(0.6f, 0f, 0.12f, 0.95f), vec4<f32>(0.3f, 0f, 0f, 1f), vec4<f32>(0.45f, 0f, 0.15f, 0.9f), vec4<f32>(0f, 0.06f, 0f, 1f), vec4<f32>(0.25f, 0.55f, -0.2f, 0.12f), vec4<f32>(0f, 0.1f, 0f, 1f), vec4<f32>(0f, 0f, 0.1f, 1f), vec4<f32>(-0.4f, 0f, 0.45f, 1f));
const mu_V2_ZONE_SWARD: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0.29f, 0.39f, 0.17f, 0.35f), vec4<f32>(0.5f, 0.5f, 0.36f, 0.6f), vec4<f32>(0.33f, 0.4f, 0.2f, 0.3f), vec4<f32>(0.3f, 0.36f, 0.21f, 0.4f), vec4<f32>(0.33f, 0.45f, 0.19f, 0.45f), vec4<f32>(0.44f, 0.44f, 0.33f, 0.6f), vec4<f32>(0.58f, 0.56f, 0.34f, 0.65f), vec4<f32>(0.66f, 0.6f, 0.46f, 0.7f), vec4<f32>(0.72f, 0.61f, 0.33f, 0.8f), vec4<f32>(0.22f, 0.37f, 0.14f, 0.45f), vec4<f32>(0.31f, 0.38f, 0.2f, 0.45f));
const mu_V2_ZONE_BARE: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0.42f, 0.36f, 0.27f, 0.2f), vec4<f32>(0.78f, 0.75f, 0.66f, 0.72f), vec4<f32>(0.4f, 0.35f, 0.27f, 0.2f), vec4<f32>(0.52f, 0.5f, 0.48f, 0.55f), vec4<f32>(0.6f, 0.6f, 0.58f, 0.55f), vec4<f32>(0.34f, 0.33f, 0.31f, 0.55f), vec4<f32>(0.58f, 0.52f, 0.4f, 0.4f), vec4<f32>(0.77f, 0.69f, 0.55f, 0.82f), vec4<f32>(0.67f, 0.46f, 0.31f, 0.5f), vec4<f32>(0.52f, 0.34f, 0.23f, 0.4f), vec4<f32>(0.23f, 0.21f, 0.15f, 0.5f));
const mu_V2_ZONE_CANOPY: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(1f, 1f, 1f, 0.1f), vec4<f32>(1.06f, 1.03f, 0.86f, 0.55f), vec4<f32>(0.97f, 1f, 1f, 0.05f), vec4<f32>(0.86f, 0.95f, 1.02f, 0.1f), vec4<f32>(0.9f, 0.97f, 1f, 0.05f), vec4<f32>(1.05f, 1f, 0.9f, 0.15f), vec4<f32>(1.05f, 1.03f, 0.86f, 0.2f), vec4<f32>(1f, 1.06f, 0.86f, 0f), vec4<f32>(1.1f, 1.05f, 0.76f, 0.6f), vec4<f32>(0.86f, 1.1f, 0.86f, 0.1f), vec4<f32>(1f, 1.02f, 0.9f, 0.2f));
const mu_V2_ZONE_WET: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0.27f, 0.38f, 0.18f, 0f), vec4<f32>(0.4f, 0.42f, 0.28f, 0f), vec4<f32>(0.28f, 0.36f, 0.2f, 0.5f), vec4<f32>(0.36f, 0.33f, 0.22f, 1f), vec4<f32>(0.3f, 0.4f, 0.2f, 3.5f), vec4<f32>(0.42f, 0.4f, 0.27f, 2f), vec4<f32>(0.42f, 0.46f, 0.27f, 0f), vec4<f32>(0.3f, 0.4f, 0.2f, 0f), vec4<f32>(0.36f, 0.42f, 0.22f, 0f), vec4<f32>(0.36f, 0.56f, 0.24f, 0f), vec4<f32>(0.3f, 0.29f, 0.18f, 0f));
const mu_V2_SEASON_HERB: array<vec3<f32>, 44> = array<vec3<f32>, 44>(vec3<f32>(0.86f, 0.8f, 0.7f), vec3<f32>(0.98f, 1.12f, 0.9f), vec3<f32>(0.96f, 1.06f, 0.92f), vec3<f32>(1.06f, 0.96f, 0.78f), vec3<f32>(0.9f, 1.08f, 0.86f), vec3<f32>(0.92f, 1.12f, 0.84f), vec3<f32>(1.18f, 1.02f, 0.74f), vec3<f32>(1.1f, 0.98f, 0.78f), vec3<f32>(0.85f, 0.8f, 0.72f), vec3<f32>(0.96f, 1.12f, 0.9f), vec3<f32>(0.97f, 1.05f, 0.93f), vec3<f32>(1.08f, 0.94f, 0.74f), vec3<f32>(0.8f, 0.76f, 0.72f), vec3<f32>(0.94f, 1.04f, 0.88f), vec3<f32>(0.98f, 1.04f, 0.96f), vec3<f32>(1.08f, 0.94f, 0.76f), vec3<f32>(0.85f, 0.82f, 0.76f), vec3<f32>(0.98f, 1.1f, 0.9f), vec3<f32>(1f, 1.06f, 0.96f), vec3<f32>(1.07f, 0.95f, 0.78f), vec3<f32>(0.9f, 0.88f, 0.86f), vec3<f32>(0.92f, 0.92f, 0.84f), vec3<f32>(0.98f, 1.02f, 0.88f), vec3<f32>(1.12f, 0.9f, 0.72f), vec3<f32>(0.92f, 0.86f, 0.76f), vec3<f32>(0.86f, 1.14f, 0.82f), vec3<f32>(1.06f, 0.98f, 0.84f), vec3<f32>(1.04f, 0.94f, 0.8f), vec3<f32>(0.9f, 0.9f, 0.92f), vec3<f32>(0.96f, 1.02f, 0.9f), vec3<f32>(1.08f, 1f, 0.88f), vec3<f32>(1.02f, 0.97f, 0.9f), vec3<f32>(1.12f, 1f, 0.74f), vec3<f32>(1.04f, 1.02f, 0.8f), vec3<f32>(0.72f, 1.14f, 0.74f), vec3<f32>(0.92f, 1.08f, 0.78f), vec3<f32>(1.1f, 1.02f, 0.82f), vec3<f32>(1f, 1.02f, 0.96f), vec3<f32>(0.92f, 1.06f, 0.92f), vec3<f32>(0.96f, 1.04f, 0.94f), vec3<f32>(0.92f, 0.84f, 0.7f), vec3<f32>(0.92f, 1.14f, 0.9f), vec3<f32>(0.96f, 1.07f, 0.92f), vec3<f32>(1.08f, 0.95f, 0.74f));
const mu_V2_SEASON_LEAF: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0f, 0.62f, 1f, 0.85f), vec4<f32>(0.82f, 0.92f, 1f, 0.95f), vec4<f32>(0f, 0.55f, 1f, 0.8f), vec4<f32>(0f, 0.4f, 1f, 0.75f), vec4<f32>(0f, 0.4f, 1f, 0.75f), vec4<f32>(0f, 0.3f, 1f, 0.65f), vec4<f32>(0f, 0.6f, 1f, 0.7f), vec4<f32>(0.9f, 1f, 1f, 0.95f), vec4<f32>(0.3f, 0.55f, 1f, 0.8f), vec4<f32>(0.85f, 0.95f, 1f, 1f), vec4<f32>(0f, 0.62f, 1f, 0.85f));
const mu_V2_SEASON_ICE: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0.55f, 0.1f, 0f, 0f), vec4<f32>(0.75f, 0.3f, 0f, 0.05f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0.82f, 0.82f, 0.76f, 0.8f), vec4<f32>(0.3f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0.1f, 0f, 0f, 0f));
const mu_V2_SEASON_WET: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(0.2f, 0.6f, 1.6f, 1.1f), vec4<f32>(0.7f, 0.9f, 1.3f, 1.2f), vec4<f32>(0.9f, 1.35f, 0.75f, 1f));
const mu_V2_SEASON_THAW: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0f, 5.3f, 0f, 3f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 10.5f, 0f, 4f), vec4<f32>(0f, 4.5f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 5.3f, 0f, 3f));
const mu_V2_SEASON_CLIMATE: array<vec3<f32>, 11> = array<vec3<f32>, 11>(vec3<f32>(0.1f, 1f, 0f), vec3<f32>(-0.75f, 0f, 0f), vec3<f32>(0.25f, 6f, 0.1f), vec3<f32>(0f, 7f, 0.12f), vec3<f32>(0.15f, 9f, 0.35f), vec3<f32>(0.2f, 6f, 0f), vec3<f32>(0.3f, 5f, 0f), vec3<f32>(0f, 0f, 0f), vec3<f32>(0.95f, 0f, 0f), vec3<f32>(0.45f, 0f, 0f), vec3<f32>(0f, 2f, 0f));
const mu_ROCK_STRATA: f32 = 1f;
const mu_ROCK_FACETS: f32 = 1f;
const mu_ROCK_STREAKS: f32 = 1f;
const mu_groundWavelength: array<f32, 4> = array<f32, 4>(8f, 2.2f, 0.32f, 0.055f);
const mu_groundAmplitude: array<f32, 4> = array<f32, 4>(0.17f, 0.04f, 0.012f, 0.002f);
const mu_groundAspect: array<f32, 4> = array<f32, 4>(0.35f, 0.24f, 0.85f, 0.72f);
const mu_groundFilterWidth: array<f32, 4> = array<f32, 4>(1f, 0.5f, 0.5f, 1f);
const mu_groundRotationC: array<f32, 4> = array<f32, 4>(0.8f, -0.6f, 0.28f, -0.96f);
const mu_groundRotationS: array<f32, 4> = array<f32, 4>(0.6f, 0.8f, 0.96f, 0.28f);
const mu_groundVarianceFactor: array<f32, 4> = array<f32, 4>(0.2f, 1.8f, 1.3f, 0.4f);
const mu_forestTauTable: array<f32, 33> = array<f32, 33>(0f, 0.0211f, 0.0446f, 0.066f, 0.0881f, 0.1113f, 0.1338f, 0.1573f, 0.182f, 0.2051f, 0.2303f, 0.2538f, 0.2796f, 0.3045f, 0.3293f, 0.3546f, 0.3816f, 0.4091f, 0.435f, 0.4636f, 0.4895f, 0.5188f, 0.5472f, 0.5668f, 0.586f, 0.6063f, 0.6289f, 0.6524f, 0.6805f, 0.7112f, 0.751f, 0.8099f, 1f);
const mu_forestPatchCdf: array<f32, 17> = array<f32, 17>(0f, 0f, 0.0001f, 0.001f, 0.0085f, 0.0401f, 0.1202f, 0.2729f, 0.5009f, 0.7274f, 0.8801f, 0.9598f, 0.9914f, 0.999f, 0.9999f, 1f, 1f);
const mu_forestCrownCdf: array<f32, 17> = array<f32, 17>(0f, 0f, 0f, 0.0005f, 0.0063f, 0.0336f, 0.1125f, 0.2742f, 0.4998f, 0.7261f, 0.8874f, 0.9665f, 0.9936f, 0.9994f, 1f, 1f, 1f);
const mu_FOREST_PATCH: f32 = 56f;
const mu_FOREST_LOBE: f32 = 14f;
const mu_FOREST_LONE: f32 = 24f;
const mu_FOREST_GLADE: f32 = 210f;
const mu_FOREST_GLADE_FRAME: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0.83f, 0.39f), vec2<f32>(-0.21f, 1.27f));
const mu_V2_CROWN_AMBIENT: vec2<f32> = vec2<f32>(0.3766294f, 0.34189487f);
const mu_V2_CROWN_RESPONSE: array<vec4<f32>, 19> = array<vec4<f32>, 19>(vec4<f32>(0.05495809f, 0.0013764778f, 0.08137881f, 0.0020390463f), vec4<f32>(0.066055395f, 0.001541067f, 0.08850629f, 0.0020669752f), vec4<f32>(0.07783681f, 0.0017119818f, 0.095201634f, 0.0020981387f), vec4<f32>(0.09020807f, 0.0018873867f, 0.10143083f, 0.0021297745f), vec4<f32>(0.10311157f, 0.0020662295f, 0.10719746f, 0.002161111f), vec4<f32>(0.11649999f, 0.002247855f, 0.112457864f, 0.002190355f), vec4<f32>(0.13033146f, 0.002432014f, 0.11722252f, 0.002217979f), vec4<f32>(0.14456406f, 0.002618785f, 0.121541254f, 0.0022458134f), vec4<f32>(0.15913282f, 0.0028080365f, 0.12548193f, 0.0022760672f), vec4<f32>(0.17388634f, 0.002997934f, 0.12920947f, 0.0023133466f), vec4<f32>(0.18827625f, 0.003177238f, 0.13293578f, 0.002362944f), vec4<f32>(0.20132901f, 0.0033263683f, 0.13726775f, 0.0024406172f), vec4<f32>(0.21284945f, 0.0034477315f, 0.14394931f, 0.0025913042f), vec4<f32>(0.22274995f, 0.0035451497f, 0.15064499f, 0.0027281195f), vec4<f32>(0.2309552f, 0.0036214865f, 0.15619417f, 0.0028264131f), vec4<f32>(0.23740272f, 0.0036788688f, 0.1605546f, 0.002895476f), vec4<f32>(0.24204347f, 0.0037188304f, 0.16369313f, 0.0029412308f), vec4<f32>(0.24484213f, 0.0037424085f, 0.16558585f, 0.002967349f), vec4<f32>(0.24577738f, 0.0037502022f, 0.16621836f, 0.0029758415f));
const mu_V2_ICE_ZOOM: f32 = 5f;
const mu_V2_SEA_ICE_ZOOM: f32 = 8f;
const mu_V2_SEASON_LAG: f32 = 0.08f;
const mu_V2_FRESH_ICE: array<vec4<f32>, 11> = array<vec4<f32>, 11>(vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(1f, 0f, 0f, 0f), vec4<f32>(1f, 0.087f, 0f, 0f), vec4<f32>(1f, 0.301f, 0f, 0.023f), vec4<f32>(1f, 1f, 0f, 0.817f), vec4<f32>(0.913f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0f, 0f, 0f, 0f), vec4<f32>(0.183f, 0f, 0f, 0f));
const mu_V2_WATER_ARIDITY: array<f32, 11> = array<f32, 11>(0f, 0.3f, 0f, 0f, 0f, 0f, 0.5f, 1f, 0.72f, 0f, 0f);
const mu_V2_WATER_TEA: array<f32, 11> = array<f32, 11>(0f, 0f, 0.3f, 0.65f, 0f, 0.25f, 0f, 0f, 0f, 0f, 0.55f);
const mu_V2_WATER_GLACIAL: array<f32, 11> = array<f32, 11>(0f, 0f, 0f, 0f, 1f, 0.2f, 0f, 0f, 0f, 0f, 0f);
const mu_V2_WATER_SEDIMENT: array<f32, 11> = array<f32, 11>(0.45f, 0.35f, 0.45f, 0.2f, 0.25f, 0.1f, 0.3f, 0.1f, 1f, 1f, 0.7f);

const mu_uMode:i32 = 3;
const mu_uForest:i32 = 1;
const mu_uRenderV2_:i32 = 1;
const mu_uForestBrushes:i32 = 0;




var<private> mu_outColor: vec4<f32>;








var<private> mu_surfaceGroundWaterCached: bool = false;
var<private> mu_surfaceGroundWaterPoint: vec2<f32> = vec2(0f);
var<private> mu_surfaceGroundWaterSample: f32 = 0f;
var<private> mu_v2Foliage: f32 = 1f;
var<private> mu_v2CrownGuide: vec3<f32> = vec3(0f);
var<private> mu_v2ZoneWet: f32 = 0f;
var<private> mu_v2ZoneBare: f32 = 0f;
var<private> mu_v2FloorModulation: f32 = 1f;
var<private> mu_v2MacroNormal: vec3<f32> = vec3<f32>(0f, 1f, 0f);
var<private> mu_v2MacroCanopy: vec3<f32> = vec3(0f);
var<private> mu_v2GroundSnow: f32 = 0f;
var<private> mu_v2MeanEnabled: f32 = 0f;
var<private> mu_v2ForestReveal: f32 = 0f;





var<private> mu_renderGuideColor: vec3<f32> = vec3(0f);
var<private> mu_renderGuideOpen: vec3<f32> = vec3(0f);
var<private> mu_renderGuideClosed: vec3<f32> = vec3(0f);
var<private> mu_renderGuideBaseOpen: vec3<f32> = vec3(0f);
var<private> mu_renderGuideBaseClosed: vec3<f32> = vec3(0f);
var<private> mu_renderGuideSediment: vec3<f32> = vec3(0f);
var<private> mu_renderGuideWoods: f32 = 0f;
var<private> mu_renderGuideBeach: f32 = 0f;
var<private> mu_renderGuideFlats: f32 = 0f;
var<private> mu_renderGuideShore: f32 = 1000000000f;
var<private> mu_renderGuideOpenness: f32 = 0f;
var<private> mu_surfaceFall: vec2<f32> = vec2<f32>(1f, 0f);
var<private> mu_surfaceRockTone: vec3<f32> = vec3<f32>(0.43f, 0.42f, 0.39f);
var<private> mu_surfaceSoilTone: vec3<f32> = vec3<f32>(0.4f, 0.32f, 0.22f);
var<private> mu_surfaceHerbTone: vec3<f32> = vec3(1f);
var<private> mu_surfaceMicroTone: vec3<f32> = vec3(1f);
var<private> mu_surfaceMicroSlope: vec2<f32> = vec2(0f);
var<private> mu_surfaceRise: f32 = 0f;
var<private> mu_surfaceBedded: f32 = 0f;
var<private> mu_surfaceStrataWeight: f32 = 0f;
var<private> mu_surfaceFacetWeight: f32 = 1f;
var<private> mu_surfaceStreakWeight: f32 = 1f;
var<private> mu_surfaceGroundSlope: vec2<f32> = vec2(0f);
var<private> mu_surfaceGroundGradient: vec2<f32> = vec2(0f);
var<private> mu_surfaceGroundEnv: vec4<f32> = vec4(0f);
var<private> mu_surfaceGroundHeight: f32 = 0f;
var<private> mu_surfaceGroundShore: f32 = 1000000000f;
var<private> mu_surfaceGroundBeach: f32 = 0f;
var<private> mu_surfaceGroundNormalGain: f32 = 1f;
var<private> mu_surfaceGroundHasInputs: bool = false;
var<private> mu_surfaceGroundPoint: vec2<f32> = vec2(0f);
var<private> mu_surfaceGroundOccupation: vec3<f32> = vec3(0f);
var<private> mu_surfaceGroundBank: f32 = 1000000000f;
var<private> mu_surfaceGroundWet: f32 = 0f;
var<private> mu_surfaceGroundBankCached: bool = false;
var<private> mu_surfaceGroundBankPoint: vec2<f32> = vec2(0f);
var<private> mu_surfaceGroundBankSample: vec4<f32> = vec4(0f);

var<private> mu_groundOccupationCached: bool = false;
var<private> mu_groundOccupationPoint: vec2<f32> = vec2(0f);
var<private> mu_groundOccupationSample: vec3<f32> = vec3(0f);
var<private> mu_forestResolvedCover: f32 = 0f;
var<private> mu_forestResolvedShade: f32 = 0f;
var<private> mu_forestMapCover: f32 = 0f;
var<private> mu_paperCommonsShare: f32 = 0f;
var<private> mu_paperTendedShare: f32 = 0f;
var<private> mu_v2IceWhole: f32 = 0.31f;
var<private> mu_v2MeanOpen: mu_GroundMaterial;
var<private> mu_v2MeanLitter: mu_GroundMaterial;
var<private> mu_v2MeanCrown: mu_GroundMaterial;
var<private> mu_v2MeanWeights: vec3<f32> = vec3<f32>(1f, 0f, 0f);
var<private> mu_v2Here: mu_V2Env;

var<private> mu_gl_FragCoord_1: vec4<f32>;


var<private> mu_global: mu_TerrainParams;

fn mu_v2Zoned()->bool{return false;}

fn mu_v2Zone()->i32{return 0;}

fn mu_v2SeasonHerb()->vec3<f32>{return vec3<f32>(1.);}

fn mu_forestWaterCoverage(p:vec2<f32>)->f32{return magnaWaterCover(p);}

fn mu_sstep(a_2: f32, b_4: f32, x: f32) -> f32 {
    var a_3: f32;
    var b_5: f32;
    var x_1: f32;
    var t_8: f32;

    a_3 = a_2;
    b_5 = b_4;
    x_1 = x;
    let _e268 = x_1;
    let _e269 = a_3;
    let _e271 = b_5;
    let _e272 = a_3;
    t_8 = clamp(((_e268 - _e269) / (_e271 - _e272)), 0f, 1f);
    let _e279 = t_8;
    let _e280 = t_8;
    let _e284 = t_8;
    return ((_e279 * _e280) * (3f - (2f * _e284)));
}

fn mu_surfaceHash(c_3: vec3<i32>, salt: u32) -> u32 {
    var c_4: vec3<i32>;
    var salt_1: u32;
    var v_6: vec3<u32>;
    var h_3: u32;

    c_4 = c_3;
    salt_1 = salt;
    let _e266 = c_4;
    v_6 = vec3<u32>(_e266);
    let _e269 = v_6;
    let _e273 = v_6;
    let _e278 = v_6;
    let _e283 = salt_1;
    h_3 = ((((_e269.x * 2654435769u) ^ (_e273.y * 2246822507u)) ^ (_e278.z * 3266489909u)) ^ _e283);
    let _e286 = h_3;
    let _e287 = h_3;
    h_3 = ((_e286 ^ (_e287 >> 16u)) * 2146121005u);
    let _e293 = h_3;
    let _e294 = h_3;
    h_3 = ((_e293 ^ (_e294 >> 15u)) * 2221713035u);
    let _e300 = h_3;
    let _e301 = h_3;
    return (_e300 ^ (_e301 >> 16u));
}

fn mu_surfaceCorner(c_5: vec3<i32>, f_7: vec3<f32>, o: vec3<i32>, salt_2: u32) -> f32 {
    var c_6: vec3<i32>;
    var f_8: vec3<f32>;
    var o_1: vec3<i32>;
    var salt_3: u32;
    var h_4: u32;
    var d_2: vec3<f32>;
    var g_5: vec3<f32>;

    c_6 = c_5;
    f_8 = f_7;
    o_1 = o;
    salt_3 = salt_2;
    let _e270 = c_6;
    let _e271 = o_1;
    let _e273 = salt_3;
    let _e274 = mu_surfaceHash((_e270 + _e271), _e273);
    h_4 = _e274;
    let _e276 = f_8;
    let _e277 = o_1;
    d_2 = (_e276 - vec3<f32>(_e277));
    let _e281 = h_4;
    let _e285 = h_4;
    let _e291 = h_4;
    g_5 = ((vec3<f32>(f32((_e281 & 1023u)), f32(((_e285 >> 10u) & 1023u)), f32(((_e291 >> 20u) & 1023u))) / vec3(511.5f)) - vec3(1f));
    let _e305 = g_5;
    let _e307 = d_2;
    let _e310 = g_5;
    let _e312 = d_2;
    let _e316 = g_5;
    let _e318 = d_2;
    return (((_e305.x * _e307.x) + (_e310.y * _e312.y)) + (_e316.z * _e318.z));
}

fn mu_surfaceNoise(p_9: vec3<f32>, salt_4: u32) -> f32 {
    var p_10: vec3<f32>;
    var salt_5: u32;
    var fl: vec3<f32>;
    var c_7: vec3<i32>;
    var f_9: vec3<f32>;
    var u_6: vec3<f32>;
    var a_6: vec4<f32> = vec4(0f);
    var b_8: vec4<f32> = vec4(0f);
    var local_30: i32;
    var corners: i32;
    var k_4: i32 = 0i;
    var v_7: f32;
    var j_2: i32;
    var x0_: f32;
    var x1_: f32;
    var y0_: f32;
    var x2_: f32;
    var x3_: f32;
    var y1_: f32;

    p_10 = p_9;
    salt_5 = salt_4;
    let _e266 = p_10;
    fl = floor(_e266);
    let _e269 = fl;
    c_7 = vec3<i32>(_e269);
    let _e272 = p_10;
    let _e273 = fl;
    f_9 = (_e272 - _e273);
    let _e276 = f_9;
    let _e277 = f_9;
    let _e279 = f_9;
    let _e281 = f_9;
    let _e282 = f_9;
    u_6 = (((_e276 * _e277) * _e279) * ((_e281 * ((_e282 * 6f) - vec3(15f))) + vec3(10f)));
    let _e300 = u_6;
    if (_e300.z == 0f) {
        local_30 = 4i;
    } else {
        local_30 = 8i;
    }
    let _e307 = local_30;
    corners = _e307;
    loop {
        let _e311 = k_4;
        let _e312 = corners;
        if !((_e311 < _e312)) {
            break;
        }
        {
            let _e318 = c_7;
            let _e319 = f_9;
            let _e320 = k_4;
            let _e323 = k_4;
            let _e329 = k_4;
            let _e334 = salt_5;
            let _e335 = mu_surfaceCorner(_e318, _e319, vec3<i32>((_e320 & 1i), ((_e323 >> 1u) & 1i), (_e329 >> 2u)), _e334);
            v_7 = _e335;
            let _e337 = k_4;
            j_2 = (_e337 & 3i);
            let _e341 = k_4;
            if (_e341 < 4i) {
                {
                    let _e344 = j_2;
                    if (_e344 == 0i) {
                        let _e348 = v_7;
                        a_6.x = _e348;
                    } else {
                        let _e349 = j_2;
                        if (_e349 == 1i) {
                            let _e353 = v_7;
                            a_6.y = _e353;
                        } else {
                            let _e354 = j_2;
                            if (_e354 == 2i) {
                                let _e358 = v_7;
                                a_6.z = _e358;
                            } else {
                                let _e360 = v_7;
                                a_6.w = _e360;
                            }
                        }
                    }
                }
            } else {
                {
                    let _e361 = j_2;
                    if (_e361 == 0i) {
                        let _e365 = v_7;
                        b_8.x = _e365;
                    } else {
                        let _e366 = j_2;
                        if (_e366 == 1i) {
                            let _e370 = v_7;
                            b_8.y = _e370;
                        } else {
                            let _e371 = j_2;
                            if (_e371 == 2i) {
                                let _e375 = v_7;
                                b_8.z = _e375;
                            } else {
                                let _e377 = v_7;
                                b_8.w = _e377;
                            }
                        }
                    }
                }
            }
        }
        continuing {
            let _e315 = k_4;
            k_4 = (_e315 + 1i);
        }
    }
    let _e378 = a_6;
    let _e380 = a_6;
    let _e382 = a_6;
    let _e385 = u_6;
    x0_ = (_e378.x + ((_e380.y - _e382.x) * _e385.x));
    let _e390 = a_6;
    let _e392 = a_6;
    let _e394 = a_6;
    let _e397 = u_6;
    x1_ = (_e390.z + ((_e392.w - _e394.z) * _e397.x));
    let _e402 = x0_;
    let _e403 = x1_;
    let _e404 = x0_;
    let _e406 = u_6;
    y0_ = (_e402 + ((_e403 - _e404) * _e406.y));
    let _e411 = u_6;
    if (_e411.z == 0f) {
        {
            let _e415 = y0_;
            return (_e415 * 5.25f);
        }
    }
    let _e418 = b_8;
    let _e420 = b_8;
    let _e422 = b_8;
    let _e425 = u_6;
    x2_ = (_e418.x + ((_e420.y - _e422.x) * _e425.x));
    let _e430 = b_8;
    let _e432 = b_8;
    let _e434 = b_8;
    let _e437 = u_6;
    x3_ = (_e430.z + ((_e432.w - _e434.z) * _e437.x));
    let _e442 = x2_;
    let _e443 = x3_;
    let _e444 = x2_;
    let _e446 = u_6;
    y1_ = (_e442 + ((_e443 - _e444) * _e446.y));
    let _e451 = y0_;
    let _e452 = y1_;
    let _e453 = y0_;
    let _e455 = u_6;
    return ((_e451 + ((_e452 - _e453) * _e455.z)) * 5.25f);
}

fn mu_surfacePlane(p_11: vec2<f32>, salt_6: u32) -> vec3<f32> {
    var p_12: vec2<f32>;
    var salt_7: u32;
    var fl_1: vec2<f32>;
    var c_8: vec2<i32>;
    var f_10: vec2<f32>;
    var u_7: vec2<f32>;
    var du_1: vec2<f32>;
    var h0_: u32;
    var h1_: u32;
    var h2_: u32;
    var h3_: u32;
    var g0_: vec2<f32>;
    var g1_: vec2<f32>;
    var g2_: vec2<f32>;
    var g3_: vec2<f32>;
    var a_7: f32;
    var b_9: f32;
    var d_3: f32;
    var e: f32;
    var k_5: f32;
    var gradient_2: vec2<f32>;

    p_12 = p_11;
    salt_7 = salt_6;
    let _e266 = p_12;
    fl_1 = floor(_e266);
    let _e269 = fl_1;
    c_8 = vec2<i32>(_e269);
    let _e272 = p_12;
    let _e273 = fl_1;
    f_10 = (_e272 - _e273);
    let _e276 = f_10;
    let _e277 = f_10;
    let _e279 = f_10;
    let _e281 = f_10;
    let _e282 = f_10;
    u_7 = (((_e276 * _e277) * _e279) * ((_e281 * ((_e282 * 6f) - vec2(15f))) + vec2(10f)));
    let _e295 = f_10;
    let _e297 = f_10;
    let _e299 = f_10;
    let _e300 = f_10;
    du_1 = (((30f * _e295) * _e297) * ((_e299 * (_e300 - vec2(2f))) + vec2(1f)));
    let _e310 = c_8;
    let _e315 = salt_7;
    let _e316 = mu_surfaceHash(vec3<i32>(_e310.x, _e310.y, 0i), _e315);
    h0_ = _e316;
    let _e318 = c_8;
    let _e322 = (_e318 + vec2<i32>(1i, 0i));
    let _e327 = salt_7;
    let _e328 = mu_surfaceHash(vec3<i32>(_e322.x, _e322.y, 0i), _e327);
    h1_ = _e328;
    let _e330 = c_8;
    let _e334 = (_e330 + vec2<i32>(0i, 1i));
    let _e339 = salt_7;
    let _e340 = mu_surfaceHash(vec3<i32>(_e334.x, _e334.y, 0i), _e339);
    h2_ = _e340;
    let _e342 = c_8;
    let _e346 = (_e342 + vec2<i32>(1i, 1i));
    let _e351 = salt_7;
    let _e352 = mu_surfaceHash(vec3<i32>(_e346.x, _e346.y, 0i), _e351);
    h3_ = _e352;
    let _e354 = h0_;
    let _e358 = h0_;
    g0_ = ((vec2<f32>(f32((_e354 & 1023u)), f32(((_e358 >> 10u) & 1023u))) / vec2(511.5f)) - vec2(1f));
    let _e372 = h1_;
    let _e376 = h1_;
    g1_ = ((vec2<f32>(f32((_e372 & 1023u)), f32(((_e376 >> 10u) & 1023u))) / vec2(511.5f)) - vec2(1f));
    let _e390 = h2_;
    let _e394 = h2_;
    g2_ = ((vec2<f32>(f32((_e390 & 1023u)), f32(((_e394 >> 10u) & 1023u))) / vec2(511.5f)) - vec2(1f));
    let _e408 = h3_;
    let _e412 = h3_;
    g3_ = ((vec2<f32>(f32((_e408 & 1023u)), f32(((_e412 >> 10u) & 1023u))) / vec2(511.5f)) - vec2(1f));
    let _e426 = g0_;
    let _e427 = f_10;
    a_7 = dot(_e426, _e427);
    let _e430 = g1_;
    let _e431 = f_10;
    b_9 = dot(_e430, (_e431 - vec2<f32>(1f, 0f)));
    let _e438 = g2_;
    let _e439 = f_10;
    d_3 = dot(_e438, (_e439 - vec2<f32>(0f, 1f)));
    let _e446 = g3_;
    let _e447 = f_10;
    e = dot(_e446, (_e447 - vec2<f32>(1f, 1f)));
    let _e454 = a_7;
    let _e455 = b_9;
    let _e457 = d_3;
    let _e459 = e;
    k_5 = (((_e454 - _e455) - _e457) + _e459);
    let _e462 = g0_;
    let _e463 = g1_;
    let _e464 = g0_;
    let _e466 = u_7;
    let _e470 = g2_;
    let _e471 = g0_;
    let _e473 = u_7;
    let _e477 = g0_;
    let _e478 = g1_;
    let _e480 = g2_;
    let _e482 = g3_;
    let _e484 = u_7;
    let _e487 = u_7;
    let _e491 = du_1;
    let _e492 = b_9;
    let _e493 = a_7;
    let _e495 = k_5;
    let _e496 = u_7;
    let _e500 = d_3;
    let _e501 = a_7;
    let _e503 = k_5;
    let _e504 = u_7;
    gradient_2 = ((((_e462 + ((_e463 - _e464) * _e466.x)) + ((_e470 - _e471) * _e473.y)) + (((((_e477 - _e478) - _e480) + _e482) * _e484.x) * _e487.y)) + (_e491 * vec2<f32>(((_e492 - _e493) + (_e495 * _e496.y)), ((_e500 - _e501) + (_e503 * _e504.x)))));
    let _e512 = a_7;
    let _e513 = b_9;
    let _e514 = a_7;
    let _e516 = u_7;
    let _e520 = d_3;
    let _e521 = a_7;
    let _e523 = u_7;
    let _e527 = k_5;
    let _e528 = u_7;
    let _e531 = u_7;
    let _e535 = gradient_2;
    return (vec3<f32>((((_e512 + ((_e513 - _e514) * _e516.x)) + ((_e520 - _e521) * _e523.y)) + ((_e527 * _e528.x) * _e531.y)), _e535.x, _e535.y) * 5.25f);
}

fn mu_surfaceBed(z_2: f32, salt_8: u32) -> f32 {
    var z_3: f32;
    var salt_9: u32;
    var i_5: f32;
    var a_8: f32;
    var b_10: f32;

    z_3 = z_2;
    salt_9 = salt_8;
    let _e295 = z_3;
    i_5 = floor(_e295);
    let _e298 = i_5;
    let _e303 = salt_9;
    let _e304 = mu_surfaceHash(vec3<i32>(i32(_e298), 0i, 0i), _e303);
    a_8 = f32((_e304 & 1023u));
    let _e309 = i_5;
    let _e316 = salt_9;
    let _e317 = mu_surfaceHash(vec3<i32>((i32(_e309) + 1i), 0i, 0i), _e316);
    b_10 = f32((_e317 & 1023u));
    let _e322 = a_8;
    let _e323 = b_10;
    let _e324 = a_8;
    let _e328 = z_3;
    let _e329 = i_5;
    let _e331 = mu_sstep(0.3f, 0.7f, (_e328 - _e329));
    return (((_e322 + ((_e323 - _e324) * _e331)) / 1023f) - 0.5f);
}

fn mu_surfaceBand(world_22: vec2<f32>, wavelength_1: f32, salt_10: u32) -> vec3<f32> {
    var world_23: vec2<f32>;
    var wavelength_2: f32;
    var salt_11: u32;
    var n_3: vec3<f32>;

    world_23 = world_22;
    wavelength_2 = wavelength_1;
    salt_11 = salt_10;
    let _e298 = world_23;
    let _e302 = world_23;
    let _e307 = world_23;
    let _e311 = world_23;
    let _e316 = wavelength_2;
    let _e319 = salt_11;
    let _e320 = mu_surfacePlane((vec2<f32>(((0.8f * _e298.x) + (0.6f * _e302.y)), ((0.8f * _e307.y) - (0.6f * _e311.x))) / vec2(_e316)), _e319);
    n_3 = _e320;
    let _e322 = n_3;
    let _e325 = n_3;
    let _e329 = n_3;
    let _e334 = n_3;
    let _e338 = n_3;
    let _e343 = wavelength_2;
    let _e345 = (vec2<f32>(((0.8f * _e325.y) - (0.6f * _e329.z)), ((0.6f * _e334.y) + (0.8f * _e338.z))) / vec2(_e343));
    return vec3<f32>(_e322.x, _e345.x, _e345.y);
}

fn mu_surfaceCrack(n_4: f32, fp: f32, wavelength_3: f32) -> f32 {
    var n_5: f32;
    var fp_1: f32;
    var wavelength_4: f32;
    var k_6: f32;

    n_5 = n_4;
    fp_1 = fp;
    wavelength_4 = wavelength_3;
    let _e299 = wavelength_4;
    let _e301 = wavelength_4;
    let _e303 = fp_1;
    let _e304 = fp_1;
    k_6 = min(40f, (((0.02f * _e299) * _e301) / max((_e303 * _e304), 0.0001f)));
    let _e313 = k_6;
    let _e317 = k_6;
    let _e319 = n_5;
    let _e321 = n_5;
    return (inverseSqrt((1f + (1.6f * _e313))) - exp2(((-(_e317) * _e319) * _e321)));
}

fn mu_surfaceFacets(q_2: vec2<f32>, salt_12: u32) -> vec4<f32> {
    var q_3: vec2<f32>;
    var salt_13: u32;
    var fl_2: vec2<f32>;
    var c_9: vec2<i32>;
    var f_11: vec2<f32>;
    var first: f32 = 9f;
    var second: f32 = 9f;
    var id: u32 = 0u;
    var j_3: i32 = -1i;
    var i_6: i32;
    var h_5: u32;
    var d_4: vec2<f32>;
    var r_4: f32;

    q_3 = q_2;
    salt_13 = salt_12;
    let _e295 = q_3;
    fl_2 = floor(_e295);
    let _e298 = fl_2;
    c_9 = vec2<i32>(_e298);
    let _e301 = q_3;
    let _e302 = fl_2;
    f_11 = (_e301 - _e302);
    loop {
        let _e314 = j_3;
        if !((_e314 <= 1i)) {
            break;
        }
        {
            i_6 = -1i;
            loop {
                let _e324 = i_6;
                if !((_e324 <= 1i)) {
                    break;
                }
                {
                    let _e331 = c_9;
                    let _e332 = i_6;
                    let _e333 = j_3;
                    let _e335 = (_e331 + vec2<i32>(_e332, _e333));
                    let _e340 = salt_13;
                    let _e341 = mu_surfaceHash(vec3<i32>(_e335.x, _e335.y, 0i), _e340);
                    h_5 = _e341;
                    let _e343 = i_6;
                    let _e345 = j_3;
                    let _e352 = h_5;
                    let _e356 = h_5;
                    let _e368 = f_11;
                    d_4 = (((vec2<f32>(f32(_e343), f32(_e345)) + vec2(0.1f)) + ((0.8f * vec2<f32>(f32((_e352 & 255u)), f32(((_e356 >> 8u) & 255u)))) / vec2(255f))) - _e368);
                    let _e371 = d_4;
                    let _e372 = d_4;
                    r_4 = dot(_e371, _e372);
                    let _e375 = r_4;
                    let _e376 = first;
                    if (_e375 < _e376) {
                        {
                            let _e378 = first;
                            second = _e378;
                            let _e379 = r_4;
                            first = _e379;
                            let _e380 = h_5;
                            id = _e380;
                        }
                    } else {
                        let _e381 = r_4;
                        let _e382 = second;
                        if (_e381 < _e382) {
                            {
                                let _e384 = r_4;
                                second = _e384;
                            }
                        }
                    }
                }
                continuing {
                    let _e328 = i_6;
                    i_6 = (_e328 + 1i);
                }
            }
        }
        continuing {
            let _e318 = j_3;
            j_3 = (_e318 + 1i);
        }
    }
    let _e385 = id;
    let _e395 = second;
    let _e397 = first;
    let _e400 = id;
    let _e406 = id;
    let _e416 = ((vec2<f32>(f32(((_e400 >> 24u) & 15u)), f32((_e406 >> 28u))) / vec2(15f)) - vec2(0.5f));
    return vec4<f32>(((f32(((_e385 >> 16u) & 255u)) / 255f) - 0.5f), (sqrt(_e395) - sqrt(_e397)), _e416.x, _e416.y);
}

fn mu_surfaceStones(q_4: vec2<f32>, salt_14: u32) -> vec2<f32> {
    var q_5: vec2<f32>;
    var salt_15: u32;

    q_5 = q_4;
    salt_15 = salt_14;
    let _e295 = q_5;
    let _e296 = salt_15;
    let _e297 = mu_surfaceFacets(_e295, _e296);
    return _e297.xy;
}

fn mu_surfaceJoint(edge: f32, pixel: f32) -> f32 {
    var edge_1: f32;
    var pixel_1: f32;
    var a_9: f32;

    edge_1 = edge;
    pixel_1 = pixel;
    let _e295 = pixel_1;
    a_9 = max(_e295, 0.004f);
    let _e301 = a_9;
    let _e304 = a_9;
    let _e306 = edge_1;
    let _e307 = mu_sstep((0.03f - _e301), (0.03f + _e304), _e306);
    let _e312 = pixel_1;
    let _e313 = mu_sstep(0.03f, 0.07f, _e312);
    return ((1f - _e307) * (1f - _e313));
}

fn mu_groundHash(cell_1: vec3<i32>, seed_2: u32) -> u32 {
    var cell_2: vec3<i32>;
    var seed_3: u32;
    var p_13: vec3<u32>;
    var h_8: u32;

    cell_2 = cell_1;
    seed_3 = seed_2;
    let _e295 = cell_2;
    p_13 = vec3<u32>(_e295);
    let _e298 = p_13;
    let _e302 = p_13;
    let _e307 = p_13;
    let _e312 = seed_3;
    h_8 = ((((_e298.x * 2654435769u) ^ (_e302.y * 2246822507u)) ^ (_e307.z * 3266489909u)) ^ _e312);
    let _e315 = h_8;
    let _e316 = h_8;
    h_8 = ((_e315 ^ (_e316 >> 16u)) * 2146121005u);
    let _e322 = h_8;
    let _e323 = h_8;
    h_8 = ((_e322 ^ (_e323 >> 15u)) * 2221713035u);
    let _e329 = h_8;
    let _e330 = h_8;
    return (_e329 ^ (_e330 >> 16u));
}

fn mu_groundGravel(point: vec2<f32>, seed_4: u32) -> vec4<f32> {
    var point_1: vec2<f32>;
    var seed_5: u32;
    var cell_3: vec2<i32>;
    var f_12: vec2<f32>;
    var sum_2: f32 = 0f;
    var tone: f32 = 0f;
    var gradient_3: vec2<f32> = vec2(0f);
    var j_4: i32 = -1i;
    var i_7: i32;
    var neighbour: vec2<i32>;
    var h_9: u32;
    var offset: vec2<f32>;
    var d_5: vec2<f32>;
    var angle_1: i32;
    var local_31: f32;
    var local_32: f32;
    var local_33: f32;
    var c_10: f32;
    var local_34: f32;
    var local_35: f32;
    var s_2: f32;
    var uv_3: vec2<f32>;
    var radius: vec2<f32>;
    var inverse: vec2<f32>;
    var q_6: f32;
    var t_11: f32;
    var weight_1: f32;
    var attenuation: f32;

    point_1 = point;
    seed_5 = seed_4;
    let _e295 = point_1;
    cell_3 = vec2<i32>(floor(_e295));
    let _e299 = point_1;
    f_12 = fract(_e299);
    loop {
        let _e312 = j_4;
        if !((_e312 <= 1i)) {
            break;
        }
        i_7 = -1i;
        loop {
            let _e322 = i_7;
            if !((_e322 <= 1i)) {
                break;
            }
            {
                let _e329 = cell_3;
                let _e330 = i_7;
                let _e331 = j_4;
                neighbour = (_e329 + vec2<i32>(_e330, _e331));
                let _e335 = neighbour;
                let _e338 = neighbour;
                let _e341 = seed_5;
                let _e342 = mu_groundHash(vec3<i32>(_e335.x, 0i, _e338.y), _e341);
                h_9 = _e342;
                let _e344 = h_9;
                if ((_e344 & 7u) > 4u) {
                    continue;
                }
                let _e349 = i_7;
                let _e350 = j_4;
                let _e357 = h_9;
                let _e361 = h_9;
                offset = ((vec2<f32>(f32(_e349), f32(_e350)) + vec2(0.5f)) + ((vec2<f32>(f32((_e357 & 255u)), f32(((_e361 >> 8u) & 255u))) / vec2(255f)) - vec2(0.5f)));
                let _e376 = f_12;
                let _e377 = offset;
                d_5 = (_e376 - _e377);
                let _e380 = h_9;
                angle_1 = i32(((_e380 >> 24u) & 3u));
                let _e387 = angle_1;
                if (_e387 == 0i) {
                    local_33 = 1f;
                } else {
                    let _e391 = angle_1;
                    if (_e391 == 2i) {
                        local_32 = 0f;
                    } else {
                        let _e395 = angle_1;
                        if (_e395 == 1i) {
                            local_31 = 0.70710677f;
                        } else {
                            local_31 = -0.70710677f;
                        }
                        let _e402 = local_31;
                        local_32 = _e402;
                    }
                    let _e404 = local_32;
                    local_33 = _e404;
                }
                let _e406 = local_33;
                c_10 = _e406;
                let _e408 = angle_1;
                if (_e408 == 0i) {
                    local_35 = 0f;
                } else {
                    let _e412 = angle_1;
                    if (_e412 == 2i) {
                        local_34 = 1f;
                    } else {
                        local_34 = 0.70710677f;
                    }
                    let _e418 = local_34;
                    local_35 = _e418;
                }
                let _e420 = local_35;
                s_2 = _e420;
                let _e422 = c_10;
                let _e423 = d_5;
                let _e426 = s_2;
                let _e427 = d_5;
                let _e431 = s_2;
                let _e433 = d_5;
                let _e436 = c_10;
                let _e437 = d_5;
                uv_3 = vec2<f32>(((_e422 * _e423.x) + (_e426 * _e427.y)), ((-(_e431) * _e433.x) + (_e436 * _e437.y)));
                let _e445 = h_9;
                let _e451 = h_9;
                radius = (vec2(0.1f) + ((0.3f * vec2<f32>(f32(((_e445 >> 16u) & 15u)), f32(((_e451 >> 20u) & 15u)))) / vec2(15f)));
                let _e466 = radius;
                let _e467 = radius;
                inverse = (vec2(1f) / (_e466 * _e467));
                let _e472 = uv_3;
                let _e473 = uv_3;
                let _e475 = inverse;
                q_6 = dot((_e472 * _e473), _e475);
                let _e478 = q_6;
                if (_e478 >= 1f) {
                    continue;
                }
                let _e482 = q_6;
                t_11 = (1f - _e482);
                let _e487 = h_9;
                weight_1 = (0.6f + ((1.8f * f32((_e487 >> 28u))) / 15f));
                let _e496 = sum_2;
                let _e497 = weight_1;
                let _e498 = t_11;
                let _e500 = t_11;
                sum_2 = (_e496 + ((_e497 * _e498) * _e500));
                let _e503 = tone;
                let _e504 = weight_1;
                let _e505 = t_11;
                let _e507 = t_11;
                let _e509 = h_9;
                tone = (_e503 + (((_e504 * _e505) * _e507) * ((f32(((_e509 >> 12u) & 15u)) / 15f) - 0.5f)));
                let _e521 = gradient_3;
                let _e524 = weight_1;
                let _e526 = t_11;
                let _e528 = c_10;
                let _e529 = uv_3;
                let _e532 = inverse;
                let _e535 = s_2;
                let _e536 = uv_3;
                let _e539 = inverse;
                let _e543 = s_2;
                let _e544 = uv_3;
                let _e547 = inverse;
                let _e550 = c_10;
                let _e551 = uv_3;
                let _e554 = inverse;
                gradient_3 = (_e521 + (((-4f * _e524) * _e526) * vec2<f32>((((_e528 * _e529.x) * _e532.x) - ((_e535 * _e536.y) * _e539.y)), (((_e543 * _e544.x) * _e547.x) + ((_e550 * _e551.y) * _e554.y)))));
            }
            continuing {
                let _e326 = i_7;
                i_7 = (_e326 + 1i);
            }
        }
        continuing {
            let _e316 = j_4;
            j_4 = (_e316 + 1i);
        }
    }
    let _e561 = sum_2;
    attenuation = exp(-(_e561));
    let _e566 = attenuation;
    let _e568 = gradient_3;
    let _e570 = attenuation;
    let _e572 = tone;
    let _e573 = attenuation;
    let _e575 = gradient_3;
    let _e577 = attenuation;
    return vec4<f32>((1f - _e566), (_e568.x * _e570), (_e572 * _e573), (_e575.y * _e577));
}

fn mu_surfaceGroundCap(d_6: vec2<f32>, radius_1: vec2<f32>) -> vec3<f32> {
    var d_7: vec2<f32>;
    var radius_2: vec2<f32>;
    var inverse_1: vec2<f32>;
    var t_12: f32;

    d_7 = d_6;
    radius_2 = radius_1;
    let _e296 = radius_2;
    let _e297 = radius_2;
    inverse_1 = (vec2(1f) / (_e296 * _e297));
    let _e303 = d_7;
    let _e304 = d_7;
    let _e306 = inverse_1;
    t_12 = max((1f - dot((_e303 * _e304), _e306)), 0f);
    let _e312 = t_12;
    let _e313 = t_12;
    let _e317 = t_12;
    let _e319 = d_7;
    let _e321 = inverse_1;
    let _e322 = (((-4f * _e317) * _e319) * _e321);
    return vec3<f32>((_e312 * _e313), _e322.x, _e322.y);
}

fn mu_surfaceGroundCover(world_26: vec2<f32>, slope_1: f32, fp_4: f32, rock: f32, grass: f32, salt_18: u32, broad: vec3<f32>, fine: vec3<f32>) -> vec3<f32> {
    var world_27: vec2<f32>;
    var slope_2: f32;
    var fp_5: f32;
    var rock_1: f32;
    var grass_1: f32;
    var salt_19: u32;
    var broad_1: vec3<f32>;
    var fine_1: vec3<f32>;
    var tone_1: vec3<f32> = vec3(0f);
    var supplied: bool;
    var local_36: vec3<f32>;
    var occupation_2: vec3<f32>;
    var local_37: i32;
    var local_38: i32;
    var cls: i32;
    var local_39: f32;
    var snow: f32;
    var exposed: f32;
    var moisture: f32;
    var green: f32;
    var local_40: f32;
    var land: f32;
    var light: vec2<f32>;
    var local_41: vec2<f32>;
    var herb_1: f32;
    var swardKeep: f32;
    var local_42: f32;
    var local_43: f32;
    var density: f32;
    var phase: f32;
    var temperature: f32;
    var fineKeep: f32;
    var s1_: vec3<f32>;
    var local_44: vec3<f32>;
    var s2_: vec3<f32>;
    var amount: f32;
    var grain: f32;
    var sand: f32;
    var angle_2: f32;
    var wind: vec2<f32>;
    var flow: vec3<f32>;
    var phase_1: f32;
    var rate: vec2<f32>;
    var keep: f32;
    var ripple: f32;
    var rippleRate: vec2<f32>;
    var rippleKeep: f32;
    var local_45: f32;
    var bank: f32;
    var distance_: f32;
    var bankKeep: f32;
    var coarse: f32;
    var k_7: f32;
    var mediumKeep: f32;
    var fineKeep_1: f32;
    var level_1: i32 = 0i;
    var local_46: f32;
    var local_47: f32;
    var keep_1: f32;
    var local_48: f32;
    var local_49: f32;
    var cell_4: f32;
    var local_50: f32;
    var local_51: f32;
    var height: f32;
    var cap: vec4<f32>;
    var g_6: vec2<f32>;
    var amount_1: f32;
    var scree: f32;
    var blockKeep: f32;
    var q_7: vec2<f32>;
    var h_10: u32;
    var radius_3: vec2<f32>;
    var reach: f32;
    var centre: vec2<f32>;
    var d_8: vec2<f32>;
    var cap_1: vec3<f32>;
    var amount_2: f32;
    var shadowRadius: vec2<f32>;
    var shadowD: vec2<f32>;
    var shadowUv: vec2<f32>;
    var shadow: f32;
    var centredShadow: f32;
    var tip: f32;
    var shadowStrength: f32;

    world_27 = world_26;
    slope_2 = slope_1;
    fp_5 = fp_4;
    rock_1 = rock;
    grass_1 = grass;
    salt_19 = salt_18;
    broad_1 = broad;
    fine_1 = fine;
    mu_surfaceGroundSlope = vec2(0f);
    let _e312 = mu_global.uQuality;
    if (_e312 > 0i) {
        let _e315 = tone_1;
        return _e315;
    }
    let _e316 = mu_global.uGroundCover;
    let _e329 = fp_5;
    let _e333 = mu_surfaceGroundHeight;
    if ((((((_e316.w < 0.5f) || (mu_uForest == 0i)) || (mu_uForestBrushes != 0i)) || (mu_uRenderV2_ > 1i)) || (_e329 >= 10f)) || (_e333 <= 0f)) {
        let _e337 = tone_1;
        return _e337;
    }
    let _e338 = fp_5;
    let _e341 = mu_surfaceGroundEnv;
    if ((_e338 >= 3f) && (_e341.z < 0.01f)) {
        let _e346 = tone_1;
        return _e346;
    }
    if (mu_uRenderV2_ == 1i) {
        let _e351 = rock_1;
        let _e356 = grass_1;
        let _e361 = mu_v2GroundSnow;
        mu_surfaceGroundNormalGain = mix(((0.02f + (0.43f * _e351)) + (0.07f * (1f - _e356))), 0.008f, _e361);
    }
    let _e363 = mu_surfaceGroundHasInputs;
    let _e364 = world_27;
    let _e365 = mu_surfaceGroundPoint;
    supplied = (_e363 && all((_e364 == _e365)));
    let _e370 = supplied;
    if _e370 {
        let _e371 = mu_surfaceGroundOccupation;
        local_36 = _e371;
    } else {
        local_36 = vec3(0f);
    }
    let _e375 = local_36;
    occupation_2 = _e375;
    let _e377 = supplied;
    if _e377 {
        let _e378 = occupation_2;
        local_38 = i32(_e378.z);
    } else {
        let _e381 = mu_global.uHasOccupation;
        if (_e381 == 1i) {
            local_37 = -1i;
        } else {
            local_37 = 0i;
        }
        let _e388 = local_37;
        local_38 = _e388;
    }
    let _e390 = local_38;
    cls = _e390;
    let _e392 = cls;
    let _e395 = cls;
    let _e399 = cls;
    let _e403 = cls;
    let _e407 = cls;
    let _e411 = cls;
    let _e415 = cls;
    let _e419 = cls;
    if ((((((((_e392 < 0i) || (_e395 == 1i)) || (_e399 == 4i)) || (_e403 == 5i)) || (_e407 == 7i)) || (_e411 == 8i)) || (_e415 == 9i)) || (_e419 == 10i)) {
        let _e423 = tone_1;
        return _e423;
    }
    if (mu_uRenderV2_ == 1i) {
        let _e426 = mu_v2GroundSnow;
        local_39 = _e426;
    } else {
        let _e427 = mu_surfaceGroundEnv;
        local_39 = _e427.w;
    }
    let _e430 = local_39;
    snow = clamp(_e430, 0f, 1f);
    let _e436 = snow;
    exposed = (1f - _e436);
    let _e439 = exposed;
    if (_e439 <= 0f) {
        let _e442 = tone_1;
        return _e442;
    }
    let _e443 = mu_surfaceGroundEnv;
    moisture = _e443.x;
    let _e446 = mu_surfaceGroundEnv;
    green = _e446.y;
    let _e449 = supplied;
    let _e450 = cls;
    if (_e449 && (_e450 > 0i)) {
        let _e456 = fp_5;
        let _e458 = occupation_2;
        let _e460 = occupation_2;
        let _e464 = mu_sstep(0f, max(0.3f, _e456), min(_e458.x, -(_e460.y)));
        local_40 = _e464;
    } else {
        local_40 = 1f;
    }
    let _e467 = local_40;
    land = _e467;
    let _e469 = exposed;
    let _e470 = land;
    let _e472 = mu_surfaceGroundWet;
    exposed = (_e469 * (_e470 * (1f - _e472)));
    let _e476 = mu_global.uGroundCover;
    light = _e476.xz;
    let _e479 = light;
    let _e480 = light;
    if (dot(_e479, _e480) > 0.0001f) {
        let _e484 = light;
        local_41 = normalize(_e484);
    } else {
        local_41 = vec2<f32>(1f, 0f);
    }
    let _e490 = local_41;
    light = _e490;
    let _e491 = grass_1;
    let _e493 = rock_1;
    let _e496 = exposed;
    let _e501 = slope_2;
    let _e502 = mu_sstep(0.35f, 0.7f, _e501);
    herb_1 = (((_e491 * (1f - _e493)) * _e496) * (1f - _e502));
    let _e509 = fp_5;
    let _e510 = mu_sstep(0.125f, 0.25f, _e509);
    swardKeep = (1f - _e510);
    let _e513 = herb_1;
    let _e516 = swardKeep;
    if ((_e513 > 0.01f) && (_e516 > 0f)) {
        {
            let _e524 = moisture;
            let _e525 = mu_sstep(0.2f, 1.6f, _e524);
            let _e528 = cls;
            if (_e528 == 2i) {
                local_43 = 1f;
            } else {
                let _e532 = cls;
                if (_e532 == 3i) {
                    local_42 = 0.8f;
                } else {
                    local_42 = 0.72f;
                }
                let _e538 = local_42;
                local_43 = _e538;
            }
            let _e540 = local_43;
            density = ((0.38f + (0.55f * _e525)) * _e540);
            if (mu_uRenderV2_ == 1i) {
                {
                    let _e548 = mu_global.uV2Season;
                    phase = (0.5f - (0.5f * cos((6.2831855f * _e548))));
                    let _e554 = mu_global.uMaterialClimate;
                    let _e557 = mu_surfaceGroundHeight;
                    let _e562 = phase;
                    temperature = ((_e554.x - (0.0065f * _e557)) + (6f * ((2f * _e562) - 1f)));
                    let _e571 = moisture;
                    let _e572 = mu_sstep(0.35f, 1.3f, _e571);
                    let _e575 = temperature;
                    let _e576 = mu_sstep(1f, 13f, _e575);
                    green = (_e572 * _e576);
                }
            }
            let _e581 = fp_5;
            let _e582 = mu_sstep(0.035f, 0.07f, _e581);
            fineKeep = (1f - _e582);
            let _e585 = world_27;
            let _e587 = salt_19;
            let _e590 = mu_surfaceBand(_e585, 1f, (_e587 + 817u));
            s1_ = _e590;
            let _e592 = fineKeep;
            if (_e592 > 0f) {
                let _e595 = world_27;
                let _e597 = salt_19;
                let _e600 = mu_surfaceBand(_e595, 0.28f, (_e597 + 818u));
                local_44 = _e600;
            } else {
                local_44 = vec3(0f);
            }
            let _e604 = local_44;
            s2_ = _e604;
            let _e606 = herb_1;
            let _e607 = density;
            amount = (_e606 * _e607);
            let _e610 = mu_surfaceGroundSlope;
            let _e611 = amount;
            let _e613 = swardKeep;
            let _e615 = s1_;
            let _e619 = fineKeep;
            let _e621 = s2_;
            mu_surfaceGroundSlope = (_e610 + (_e611 * (((0.03f * _e613) * _e615.yz) + ((0.012f * _e619) * _e621.yz))));
            let _e628 = swardKeep;
            let _e630 = s1_;
            let _e634 = fineKeep;
            let _e636 = s2_;
            grain = (((0.6f * _e628) * _e630.x) + ((0.4f * _e634) * _e636.x));
            let _e641 = tone_1;
            let _e643 = amount;
            let _e645 = grain;
            let _e655 = green;
            tone_1 = (_e641 + (((0.07f * _e643) * _e645) * mix(vec3<f32>(0.85f, 0.7f, 0.32f), vec3<f32>(0.55f, 0.85f, 0.38f), vec3(_e655))));
        }
    }
    let _e662 = mu_surfaceGroundEnv;
    let _e664 = mu_sstep(0.3f, 0.7f, _e662.z);
    let _e666 = rock_1;
    let _e670 = grass_1;
    let _e673 = exposed;
    let _e678 = slope_2;
    let _e679 = mu_sstep(0.35f, 0.7f, _e678);
    sand = ((((_e664 * (1f - _e666)) * (1f - _e670)) * _e673) * (1f - _e679));
    let _e683 = sand;
    if (_e683 > 0.01f) {
        {
            let _e686 = salt_19;
            angle_2 = (f32((_e686 & 1023u)) * 0.006135923f);
            let _e693 = angle_2;
            let _e695 = angle_2;
            wind = vec2<f32>(cos(_e693), sin(_e695));
            let _e699 = world_27;
            let _e701 = salt_19;
            let _e704 = mu_surfaceBand(_e699, 1600f, (_e701 + 821u));
            flow = _e704;
            let _e706 = world_27;
            let _e707 = wind;
            let _e712 = flow;
            let _e717 = mu_surfaceGroundHeight;
            phase_1 = (((dot(_e706, _e707) * 0.0698f) + (2.5f * _e712.x)) + (0.07f * _e717));
            let _e721 = wind;
            let _e725 = flow;
            let _e730 = mu_surfaceGroundGradient;
            rate = (((_e721 * 0.0698f) + (2.5f * _e725.yz)) + (0.07f * _e730));
            let _e737 = fp_5;
            let _e738 = mu_sstep(5f, 10f, _e737);
            let _e743 = fp_5;
            let _e744 = rate;
            let _e749 = mu_sstep(0.125f, 0.25f, ((_e743 * length(_e744)) / 6.2831855f));
            keep = ((1f - _e738) * (1f - _e749));
            let _e753 = mu_surfaceGroundSlope;
            let _e754 = sand;
            let _e755 = keep;
            let _e759 = phase_1;
            let _e762 = rate;
            mu_surfaceGroundSlope = (_e753 + ((((_e754 * _e755) * 0.9f) * cos(_e759)) * _e762));
            let _e765 = tone_1;
            let _e766 = sand;
            let _e767 = keep;
            let _e771 = phase_1;
            tone_1 = (_e765 + vec3((((_e766 * _e767) * 0.05f) * sin(_e771))));
            let _e776 = fp_5;
            if (_e776 < 0.32f) {
                {
                    let _e779 = world_27;
                    let _e780 = wind;
                    let _e785 = flow;
                    let _e790 = mu_surfaceGroundHeight;
                    ripple = (((dot(_e779, _e780) * 4.4879894f) + (1400f * _e785.x)) + (0.6f * _e790));
                    let _e794 = wind;
                    let _e798 = flow;
                    let _e803 = mu_surfaceGroundGradient;
                    rippleRate = (((_e794 * 4.4879894f) + (1400f * _e798.yz)) + (0.6f * _e803));
                    let _e810 = fp_5;
                    let _e811 = mu_sstep(0.16f, 0.32f, _e810);
                    let _e816 = fp_5;
                    let _e817 = rippleRate;
                    let _e822 = mu_sstep(0.125f, 0.25f, ((_e816 * length(_e817)) / 6.2831855f));
                    rippleKeep = ((1f - _e811) * (1f - _e822));
                    let _e826 = mu_surfaceGroundSlope;
                    let _e827 = sand;
                    let _e828 = rippleKeep;
                    let _e832 = ripple;
                    let _e835 = rippleRate;
                    mu_surfaceGroundSlope = (_e826 + ((((_e827 * _e828) * 0.065f) * cos(_e832)) * _e835));
                }
            }
        }
    }
    let _e838 = supplied;
    if _e838 {
        let _e839 = mu_surfaceGroundBank;
        local_45 = _e839;
    } else {
        local_45 = 1000000000f;
    }
    let _e842 = local_45;
    bank = _e842;
    let _e844 = mu_surfaceGroundShore;
    let _e845 = bank;
    distance_ = min(_e844, _e845);
    let _e848 = exposed;
    let _e851 = distance_;
    let _e852 = mu_sstep(0f, 0.3f, _e851);
    let _e857 = distance_;
    let _e858 = mu_sstep(8f, 36f, _e857);
    let _e864 = slope_2;
    let _e865 = mu_sstep(0.5f, 0.9f, _e864);
    bankKeep = (((_e848 * _e852) * (1f - _e858)) * (1f - _e865));
    let _e869 = bankKeep;
    let _e870 = mu_surfaceGroundBeach;
    let _e873 = grass_1;
    bankKeep = (_e869 * max(_e870, (1f - (0.55f * _e873))));
    let _e878 = bankKeep;
    let _e881 = fp_5;
    if ((_e878 > 0.01f) && (_e881 < 1f)) {
        {
            let _e888 = distance_;
            let _e889 = mu_sstep(2f, 18f, _e888);
            coarse = (1f - _e889);
            let _e895 = fp_5;
            let _e896 = mu_sstep(0.4f, 1f, _e895);
            k_7 = (1f - _e896);
            let _e902 = fp_5;
            let _e903 = mu_sstep(0.12f, 0.3f, _e902);
            mediumKeep = (1f - _e903);
            let _e909 = fp_5;
            let _e910 = mu_sstep(0.01f, 0.026f, _e909);
            fineKeep_1 = (1f - _e910);
            loop {
                let _e915 = level_1;
                if !((_e915 < 3i)) {
                    break;
                }
                {
                    let _e922 = level_1;
                    if (_e922 == 0i) {
                        let _e925 = k_7;
                        let _e927 = mediumKeep;
                        local_47 = (_e925 * (1f - _e927));
                    } else {
                        let _e930 = level_1;
                        if (_e930 == 1i) {
                            let _e933 = mediumKeep;
                            let _e935 = fineKeep_1;
                            local_46 = (_e933 * (1f - _e935));
                        } else {
                            let _e938 = fineKeep_1;
                            local_46 = _e938;
                        }
                        let _e940 = local_46;
                        local_47 = _e940;
                    }
                    let _e942 = local_47;
                    keep_1 = _e942;
                    let _e944 = keep_1;
                    if (_e944 <= 0f) {
                        continue;
                    }
                    let _e947 = level_1;
                    if (_e947 == 0i) {
                        local_49 = 8f;
                    } else {
                        let _e951 = level_1;
                        if (_e951 == 1i) {
                            local_48 = 2.4f;
                        } else {
                            local_48 = 0.26f;
                        }
                        let _e957 = local_48;
                        local_49 = _e957;
                    }
                    let _e959 = local_49;
                    cell_4 = _e959;
                    let _e961 = level_1;
                    if (_e961 == 0i) {
                        local_51 = 0.75f;
                    } else {
                        let _e965 = level_1;
                        if (_e965 == 1i) {
                            local_50 = 0.32f;
                        } else {
                            local_50 = 0.03f;
                        }
                        let _e971 = local_50;
                        local_51 = _e971;
                    }
                    let _e973 = local_51;
                    height = _e973;
                    let _e975 = world_27;
                    let _e976 = cell_4;
                    let _e979 = salt_19;
                    let _e982 = level_1;
                    let _e985 = mu_groundGravel((_e975 / vec2(_e976)), ((_e979 + 827u) + u32(_e982)));
                    cap = _e985;
                    let _e987 = cap;
                    let _e989 = cell_4;
                    g_6 = (_e987.yw / vec2(_e989));
                    let _e993 = bankKeep;
                    let _e994 = keep_1;
                    let _e998 = coarse;
                    amount_1 = ((_e993 * _e994) * (0.45f + (0.55f * _e998)));
                    let _e1003 = mu_surfaceGroundSlope;
                    let _e1004 = amount_1;
                    let _e1005 = height;
                    let _e1007 = g_6;
                    mu_surfaceGroundSlope = (_e1003 + ((_e1004 * _e1005) * _e1007));
                    let _e1010 = tone_1;
                    let _e1011 = amount_1;
                    let _e1012 = height;
                    let _e1015 = g_6;
                    let _e1016 = light;
                    let _e1020 = cap;
                    let _e1025 = cap;
                    let _e1033 = cap;
                    tone_1 = (_e1010 + (_e1011 * ((vec3((((_e1012 * 0.2f) * dot(_e1015, _e1016)) - (0.1f * _e1020.x))) + (_e1025.z * vec3<f32>(0.28f, 0.25f, 0.2f))) + (_e1033.x * vec3<f32>(0.08f, 0.07f, 0.16f)))));
                }
                continuing {
                    let _e919 = level_1;
                    level_1 = (_e919 + 1i);
                }
            }
        }
    }
    let _e1045 = slope_2;
    let _e1046 = mu_sstep(0.22f, 0.42f, _e1045);
    let _e1050 = slope_2;
    let _e1051 = mu_sstep(1.1f, 1.7f, _e1050);
    let _e1056 = grass_1;
    let _e1060 = exposed;
    scree = (((_e1046 * (1f - _e1051)) * (1f - (0.85f * _e1056))) * _e1060);
    let _e1066 = fp_5;
    let _e1067 = mu_sstep(1.2f, 3f, _e1066);
    blockKeep = (1f - _e1067);
    let _e1070 = scree;
    let _e1073 = blockKeep;
    if ((_e1070 > 0.01f) && (_e1073 > 0f)) {
        {
            let _e1077 = world_27;
            q_7 = (_e1077 / vec2(16f));
            let _e1082 = q_7;
            let _e1084 = vec2<i32>(floor(_e1082));
            let _e1089 = salt_19;
            let _e1090 = mu_groundHash(vec3<i32>(_e1084.x, _e1084.y, 839i), _e1089);
            h_10 = _e1090;
            let _e1092 = h_10;
            let _e1100 = rock_1;
            if ((f32((_e1092 >> 24u)) / 255f) < (0.12f + (0.16f * _e1100))) {
                {
                    let _e1106 = h_10;
                    let _e1112 = h_10;
                    radius_3 = (vec2(0.09f) + ((0.12f * vec2<f32>(f32(((_e1106 >> 16u) & 15u)), f32(((_e1112 >> 20u) & 15u)))) / vec2(15f)));
                    let _e1126 = radius_3;
                    let _e1128 = radius_3;
                    reach = (max(_e1126.x, _e1128.y) * 1.9f);
                    let _e1134 = reach;
                    let _e1137 = reach;
                    let _e1140 = h_10;
                    let _e1144 = h_10;
                    centre = (vec2(_e1134) + (((1f - (2f * _e1137)) * vec2<f32>(f32((_e1140 & 255u)), f32(((_e1144 >> 8u) & 255u)))) / vec2(255f)));
                    let _e1158 = q_7;
                    let _e1160 = centre;
                    d_8 = (fract(_e1158) - _e1160);
                    let _e1163 = d_8;
                    let _e1164 = radius_3;
                    let _e1165 = mu_surfaceGroundCap(_e1163, _e1164);
                    cap_1 = _e1165;
                    let _e1167 = scree;
                    let _e1168 = blockKeep;
                    amount_2 = (_e1167 * _e1168);
                    let _e1171 = mu_surfaceGroundSlope;
                    let _e1172 = amount_2;
                    let _e1173 = cap_1;
                    mu_surfaceGroundSlope = (_e1171 + (((_e1172 * _e1173.yz) * 1.3f) / vec2(16f)));
                    let _e1183 = radius_3;
                    let _e1185 = radius_3;
                    let _e1190 = radius_3;
                    let _e1192 = radius_3;
                    shadowRadius = vec2<f32>((1.2f * max(_e1183.x, _e1185.y)), (0.95f * min(_e1190.x, _e1192.y)));
                    let _e1198 = d_8;
                    let _e1199 = light;
                    let _e1202 = shadowRadius;
                    shadowD = (_e1198 + ((_e1199 * 0.55f) * _e1202.x));
                    let _e1207 = shadowD;
                    let _e1208 = light;
                    let _e1210 = shadowD;
                    let _e1211 = light;
                    let _e1214 = light;
                    shadowUv = vec2<f32>(dot(_e1207, _e1208), dot(_e1210, vec2<f32>(-(_e1211.y), _e1214.x)));
                    let _e1220 = shadowUv;
                    let _e1221 = shadowRadius;
                    let _e1222 = mu_surfaceGroundCap(_e1220, _e1221);
                    shadow = _e1222.x;
                    let _e1225 = shadow;
                    let _e1226 = cap_1;
                    let _e1228 = shadowRadius;
                    let _e1230 = shadowRadius;
                    let _e1234 = radius_3;
                    let _e1236 = radius_3;
                    centredShadow = (_e1225 - ((_e1226.x * (_e1228.x * _e1230.y)) / (_e1234.x * _e1236.y)));
                    let _e1242 = cap_1;
                    let _e1244 = cap_1;
                    tip = (_e1242.x * (sqrt(_e1244.x) - 0.75f));
                    let _e1254 = mu_global.uGroundCover;
                    let _e1256 = mu_global.uGroundCover;
                    let _e1262 = mu_sstep(0.85f, 0.99f, (_e1254.y / max(length(_e1256.xyz), 0.001f)));
                    shadowStrength = (1f - _e1262);
                    let _e1265 = tone_1;
                    let _e1266 = amount_2;
                    let _e1268 = tip;
                    let _e1271 = centredShadow;
                    let _e1273 = shadowStrength;
                    tone_1 = (_e1265 + vec3((_e1266 * ((0.3f * _e1268) - ((0.1f * _e1271) * _e1273)))));
                }
            }
        }
    }
    let _e1279 = tone_1;
    return _e1279;
}

fn mu_surfaceChute(world_28: vec2<f32>, across: f32, along: f32, salt_20: u32) -> f32 {
    var world_29: vec2<f32>;
    var across_1: f32;
    var along_1: f32;
    var salt_21: u32;
    var step_: vec2<f32>;
    var n_6: f32 = 0f;
    var i_8: i32 = -2i;

    world_29 = world_28;
    across_1 = across;
    along_1 = along;
    salt_21 = salt_20;
    let _e299 = mu_surfaceFall;
    let _e300 = along_1;
    step_ = (_e299 * _e300);
    loop {
        let _e308 = i_8;
        if !((_e308 <= 2i)) {
            break;
        }
        let _e315 = n_6;
        let _e316 = world_29;
        let _e317 = step_;
        let _e318 = i_8;
        let _e322 = across_1;
        let _e324 = ((_e316 + (_e317 * f32(_e318))) / vec2(_e322));
        let _e329 = salt_21;
        let _e330 = mu_surfaceNoise(vec3<f32>(_e324.x, _e324.y, 0f), _e329);
        n_6 = (_e315 + _e330);
        continuing {
            let _e312 = i_8;
            i_8 = (_e312 + 1i);
        }
    }
    let _e332 = n_6;
    return (_e332 * 0.42f);
}

fn mu_surfaceMicroLocal(world_30: vec2<f32>, slope_3: f32, fp_6: f32, rock_2: f32, grass_2: f32, salt_22: u32) -> vec3<f32> {
    var world_31: vec2<f32>;
    var slope_4: f32;
    var fp_7: f32;
    var rock_3: f32;
    var grass_3: f32;
    var salt_23: u32;
    var fallLength: f32;
    var local_52: vec2<f32>;
    var k1_: f32;
    var k2_: f32;
    var k3_: f32;
    var k4_: f32;
    var n1_: vec3<f32> = vec3(0f);
    var n2_: vec3<f32> = vec3(0f);
    var n3_: vec3<f32> = vec3(0f);
    var n4_: vec3<f32> = vec3(0f);
    var k_8: i32 = 0i;
    var local_53: f32;
    var local_54: f32;
    var local_55: f32;
    var n_7: vec3<f32>;
    var mottle: f32;
    var soil: f32;
    var herb_2: f32;
    var tone_2: f32;
    var hue: vec3<f32>;
    var relief: f32;
    var detail_1: f32;
    var steep: f32;
    var wear: f32 = 0f;
    var k_9: i32 = 0i;
    var local_56: f32;
    var local_57: f32;
    var local_58: f32;
    var local_59: f32;
    var chute: f32;
    var blocky: f32;
    var block: f32 = 0f;
    var tilt: vec2<f32> = vec2(0f);
    var k_10: i32 = 0i;
    var local_60: f32;
    var local_61: f32;
    var size: f32;
    var f_13: vec4<f32>;
    var joint: f32;
    var face: f32;
    var rate_1: f32;
    var beds: vec3<f32> = vec3(0f);
    var k_11: i32 = 0i;
    var local_62: f32;
    var local_63: f32;
    var t_13: f32;
    var b1_: f32;
    var b2_: f32;
    var b3_: f32;
    var talus: f32;
    var clastic: f32;
    var k_12: i32 = 0i;
    var local_64: f32;
    var local_65: f32;
    var st: vec2<f32>;

    world_31 = world_30;
    slope_4 = slope_3;
    fp_7 = fp_6;
    rock_3 = rock_2;
    grass_3 = grass_2;
    salt_23 = salt_22;
    mu_surfaceMicroSlope = vec2(0f);
    mu_surfaceMicroTone = vec3(1f);
    mu_surfaceGroundSlope = vec2(0f);
    mu_surfaceGroundNormalGain = 1f;
    let _e310 = mu_global.uQuality;
    if (_e310 > 0i) {
        {
            let _e313 = fp_7;
            if (_e313 < 2f) {
                let _e318 = world_31;
                let _e320 = salt_23;
                let _e323 = mu_surfaceBand(_e318, 40f, (_e320 + 31u));
                let _e328 = fp_7;
                let _e329 = mu_sstep(1f, 2f, _e328);
                let _e334 = grass_3;
                mu_surfaceMicroTone = (vec3(1f) + ((_e323.x * (1f - _e329)) * (vec3(0.025f) + (_e334 * vec3<f32>(0.025f, 0.01f, -0.015f)))));
            }
            let _e344 = mu_surfaceMicroTone;
            return _e344;
        }
    }
    let _e345 = fp_7;
    if (_e345 >= 10f) {
        {
            let _e348 = mu_surfaceMicroTone;
            return _e348;
        }
    }
    let _e349 = mu_surfaceGroundGradient;
    fallLength = length(_e349);
    let _e352 = fallLength;
    if (_e352 > 0.0001f) {
        let _e355 = mu_surfaceGroundGradient;
        let _e356 = fallLength;
        local_52 = (_e355 / vec2(_e356));
    } else {
        local_52 = vec2<f32>(1f, 0f);
    }
    let _e363 = local_52;
    mu_surfaceFall = _e363;
    let _e367 = fp_7;
    let _e368 = mu_sstep(5f, 10f, _e367);
    k1_ = (1f - _e368);
    let _e374 = fp_7;
    let _e375 = mu_sstep(1.4f, 2.8f, _e374);
    k2_ = (1f - _e375);
    let _e381 = fp_7;
    let _e382 = mu_sstep(0.38f, 0.76f, _e381);
    k3_ = (1f - _e382);
    let _e388 = fp_7;
    let _e389 = mu_sstep(0.1f, 0.2f, _e388);
    k4_ = (1f - _e389);
    loop {
        let _e406 = k_8;
        if !((_e406 < 4i)) {
            break;
        }
        {
            let _e413 = k_8;
            let _e416 = k2_;
            let _e420 = k_8;
            let _e423 = k3_;
            let _e428 = k_8;
            let _e431 = k4_;
            if ((((_e413 == 1i) && (_e416 <= 0f)) || ((_e420 == 2i) && (_e423 <= 0f))) || ((_e428 == 3i) && (_e431 <= 0f))) {
                continue;
            }
            let _e436 = world_31;
            let _e437 = k_8;
            if (_e437 == 0i) {
                local_55 = 40f;
            } else {
                let _e441 = k_8;
                if (_e441 == 1i) {
                    local_54 = 11f;
                } else {
                    let _e445 = k_8;
                    if (_e445 == 2i) {
                        local_53 = 3f;
                    } else {
                        local_53 = 0.8f;
                    }
                    let _e451 = local_53;
                    local_54 = _e451;
                }
                let _e453 = local_54;
                local_55 = _e453;
            }
            let _e455 = local_55;
            let _e456 = salt_23;
            let _e459 = k_8;
            let _e462 = mu_surfaceBand(_e436, _e455, ((_e456 + 31u) + u32(_e459)));
            n_7 = _e462;
            let _e464 = k_8;
            if (_e464 == 0i) {
                let _e467 = n_7;
                n1_ = _e467;
            } else {
                let _e468 = k_8;
                if (_e468 == 1i) {
                    let _e471 = n_7;
                    n2_ = _e471;
                } else {
                    let _e472 = k_8;
                    if (_e472 == 2i) {
                        let _e475 = n_7;
                        n3_ = _e475;
                    } else {
                        let _e476 = n_7;
                        n4_ = _e476;
                    }
                }
            }
        }
        continuing {
            let _e410 = k_8;
            k_8 = (_e410 + 1i);
        }
    }
    let _e479 = k1_;
    let _e481 = n1_;
    let _e485 = k2_;
    let _e487 = n2_;
    let _e492 = k3_;
    let _e494 = n3_;
    let _e499 = k4_;
    let _e501 = n4_;
    mu_surfaceMicroSlope = (0.1f * (((((40f * _e479) * _e481.yz) + ((11f * _e485) * _e487.yz)) + ((3f * _e492) * _e494.yz)) + ((0.8f * _e499) * _e501.yz)));
    let _e507 = k1_;
    let _e509 = n1_;
    let _e513 = k2_;
    let _e515 = n2_;
    let _e520 = k3_;
    let _e522 = n3_;
    let _e527 = k4_;
    let _e529 = n4_;
    mottle = (((((0.5f * _e507) * _e509.x) + ((0.38f * _e513) * _e515.x)) + ((0.3f * _e520) * _e522.x)) + ((0.22f * _e527) * _e529.x));
    let _e535 = rock_3;
    let _e538 = grass_3;
    soil = ((1f - _e535) * (1f - _e538));
    let _e543 = rock_3;
    let _e545 = grass_3;
    herb_2 = ((1f - _e543) * _e545);
    let _e549 = mottle;
    let _e551 = soil;
    let _e554 = herb_2;
    let _e558 = rock_3;
    tone_2 = (1f + (_e549 * (((0.05f * _e551) + (0.075f * _e554)) + (0.035f * _e558))));
    let _e566 = mottle;
    let _e567 = soil;
    let _e574 = herb_2;
    hue = (vec3(1f) + (_e566 * ((_e567 * vec3<f32>(0.012f, 0f, -0.016f)) + (_e574 * vec3<f32>(0.075f, 0.01f, -0.06f)))));
    let _e585 = rock_3;
    if (_e585 > 0.02f) {
        {
            let _e592 = slope_4;
            let _e593 = mu_sstep(0.5f, 1.2f, _e592);
            relief = mix(1f, 0.65f, _e593);
            let _e597 = k3_;
            let _e599 = n3_;
            let _e602 = k4_;
            let _e604 = n4_;
            let _e608 = fp_7;
            let _e610 = mu_surfaceCrack((_e599.x + ((0.4f * _e602) * _e604.x)), _e608, 3f);
            let _e613 = k4_;
            let _e615 = n4_;
            let _e617 = fp_7;
            let _e619 = mu_surfaceCrack(_e615.x, _e617, 0.8f);
            detail_1 = (((0.24f * _e597) * _e610) + ((0.16f * _e613) * _e619));
            let _e625 = slope_4;
            let _e626 = mu_sstep(0.12f, 0.5f, _e625);
            steep = _e626;
            let _e628 = mu_surfaceStreakWeight;
            if (_e628 > 0.01f) {
                {
                    loop {
                        let _e635 = k_9;
                        if !((_e635 < 3i)) {
                            break;
                        }
                        {
                            let _e642 = k_9;
                            let _e645 = k1_;
                            let _e649 = k_9;
                            let _e652 = k2_;
                            if (((_e642 == 1i) && (_e645 <= 0f)) || ((_e649 == 2i) && (_e652 <= 0f))) {
                                continue;
                            }
                            let _e657 = world_31;
                            let _e658 = k_9;
                            if (_e658 == 0i) {
                                local_57 = 26f;
                            } else {
                                let _e662 = k_9;
                                if (_e662 == 1i) {
                                    local_56 = 9f;
                                } else {
                                    local_56 = 3.2f;
                                }
                                let _e668 = local_56;
                                local_57 = _e668;
                            }
                            let _e670 = local_57;
                            let _e671 = k_9;
                            if (_e671 == 0i) {
                                local_59 = 15f;
                            } else {
                                let _e675 = k_9;
                                if (_e675 == 1i) {
                                    local_58 = 5.4f;
                                } else {
                                    local_58 = 1.9f;
                                }
                                let _e681 = local_58;
                                local_59 = _e681;
                            }
                            let _e683 = local_59;
                            let _e684 = salt_23;
                            let _e687 = k_9;
                            let _e690 = mu_surfaceChute(_e657, _e670, _e683, ((_e684 + 61u) + u32(_e687)));
                            chute = _e690;
                            let _e692 = k_9;
                            if (_e692 == 0i) {
                                let _e699 = fp_7;
                                let _e700 = mu_sstep(14f, 30f, _e699);
                                let _e703 = chute;
                                wear = ((0.06f * (1f - _e700)) * _e703);
                            } else {
                                let _e705 = k_9;
                                if (_e705 == 1i) {
                                    let _e708 = wear;
                                    let _e710 = k1_;
                                    let _e712 = chute;
                                    wear = (_e708 + ((0.045f * _e710) * _e712));
                                } else {
                                    let _e715 = wear;
                                    let _e717 = k2_;
                                    let _e719 = chute;
                                    wear = (_e715 + ((0.04f * _e717) * _e719));
                                }
                            }
                        }
                        continuing {
                            let _e639 = k_9;
                            k_9 = (_e639 + 1i);
                        }
                    }
                    let _e722 = detail_1;
                    let _e723 = steep;
                    let _e724 = mu_surfaceStreakWeight;
                    let _e726 = wear;
                    detail_1 = (_e722 + ((_e723 * _e724) * _e726));
                }
            }
            let _e729 = mu_surfaceFacetWeight;
            let _e732 = slope_4;
            let _e733 = mu_sstep(0.1f, 0.4f, _e732);
            blocky = (_e729 * _e733);
            let _e736 = blocky;
            if (_e736 > 0.01f) {
                {
                    loop {
                        let _e746 = k_10;
                        if !((_e746 < 3i)) {
                            break;
                        }
                        {
                            let _e753 = k_10;
                            let _e756 = k2_;
                            let _e760 = k_10;
                            let _e763 = k3_;
                            if (((_e753 == 1i) && (_e756 <= 0f)) || ((_e760 == 2i) && (_e763 <= 0f))) {
                                continue;
                            }
                            let _e768 = k_10;
                            if (_e768 == 0i) {
                                local_61 = 40f;
                            } else {
                                let _e772 = k_10;
                                if (_e772 == 1i) {
                                    local_60 = 11f;
                                } else {
                                    local_60 = 3f;
                                }
                                let _e778 = local_60;
                                local_61 = _e778;
                            }
                            let _e780 = local_61;
                            size = _e780;
                            let _e782 = world_31;
                            let _e783 = size;
                            let _e786 = salt_23;
                            let _e789 = k_10;
                            let _e792 = mu_surfaceFacets((_e782 / vec2(_e783)), ((_e786 + 71u) + u32(_e789)));
                            f_13 = _e792;
                            let _e794 = f_13;
                            let _e796 = fp_7;
                            let _e797 = size;
                            let _e799 = mu_surfaceJoint(_e794.y, (_e796 / _e797));
                            joint = _e799;
                            let _e801 = k_10;
                            if (_e801 == 0i) {
                                {
                                    let _e805 = k1_;
                                    let _e807 = f_13;
                                    let _e811 = k1_;
                                    let _e813 = joint;
                                    block = (((0.1f * _e805) * _e807.x) - ((0.12f * _e811) * _e813));
                                    let _e817 = k1_;
                                    let _e819 = f_13;
                                    tilt = ((0.5f * _e817) * _e819.zw);
                                }
                            } else {
                                let _e822 = k_10;
                                if (_e822 == 1i) {
                                    {
                                        let _e825 = block;
                                        let _e827 = k2_;
                                        let _e829 = f_13;
                                        let _e833 = k2_;
                                        let _e835 = joint;
                                        block = (_e825 + (((0.08f * _e827) * _e829.x) - ((0.12f * _e833) * _e835)));
                                        let _e839 = tilt;
                                        let _e841 = k2_;
                                        let _e843 = f_13;
                                        tilt = (_e839 + ((0.4f * _e841) * _e843.zw));
                                    }
                                } else {
                                    {
                                        let _e847 = block;
                                        let _e849 = k3_;
                                        let _e851 = f_13;
                                        let _e855 = k3_;
                                        let _e857 = joint;
                                        block = (_e847 + (((0.06f * _e849) * _e851.x) - ((0.1f * _e855) * _e857)));
                                        let _e861 = tilt;
                                        let _e863 = k3_;
                                        let _e865 = f_13;
                                        tilt = (_e861 + ((0.3f * _e863) * _e865.zw));
                                    }
                                }
                            }
                        }
                        continuing {
                            let _e750 = k_10;
                            k_10 = (_e750 + 1i);
                        }
                    }
                    let _e869 = detail_1;
                    let _e870 = blocky;
                    let _e871 = block;
                    detail_1 = (_e869 + (_e870 * _e871));
                    let _e874 = mu_surfaceMicroSlope;
                    let _e875 = rock_3;
                    let _e876 = blocky;
                    let _e878 = tilt;
                    mu_surfaceMicroSlope = (_e874 + ((_e875 * _e876) * _e878));
                }
            }
            let _e881 = tone_2;
            let _e882 = rock_3;
            let _e883 = relief;
            let _e885 = detail_1;
            tone_2 = (_e881 + ((_e882 * _e883) * _e885));
            let _e888 = rock_3;
            let _e891 = slope_4;
            let _e892 = mu_sstep(0.35f, 0.9f, _e891);
            let _e894 = mu_surfaceStrataWeight;
            face = ((_e888 * _e892) * _e894);
            let _e897 = slope_4;
            rate_1 = (_e897 + 0.2f);
            let _e901 = face;
            if (_e901 > 0.01f) {
                {
                    loop {
                        let _e909 = k_11;
                        if !((_e909 < 3i)) {
                            break;
                        }
                        {
                            let _e916 = k_11;
                            if (_e916 == 0i) {
                                local_63 = 18f;
                            } else {
                                let _e920 = k_11;
                                if (_e920 == 1i) {
                                    local_62 = 4.5f;
                                } else {
                                    local_62 = 1.1f;
                                }
                                let _e926 = local_62;
                                local_63 = _e926;
                            }
                            let _e928 = local_63;
                            t_13 = _e928;
                            let _e930 = k_11;
                            let _e932 = mu_surfaceRise;
                            let _e933 = t_13;
                            let _e935 = salt_23;
                            let _e938 = k_11;
                            let _e941 = mu_surfaceBed((_e932 / _e933), ((_e935 + 41u) + u32(_e938)));
                            let _e945 = fp_7;
                            let _e946 = rate_1;
                            let _e948 = t_13;
                            let _e950 = mu_sstep(0.125f, 0.25f, ((_e945 * _e946) / _e948));
                            beds[_e930] = (_e941 * (1f - _e950));
                        }
                        continuing {
                            let _e913 = k_11;
                            k_11 = (_e913 + 1i);
                        }
                    }
                    let _e953 = beds;
                    b1_ = _e953.x;
                    let _e956 = beds;
                    b2_ = _e956.y;
                    let _e959 = beds;
                    b3_ = _e959.z;
                    let _e962 = tone_2;
                    let _e963 = face;
                    let _e964 = relief;
                    let _e967 = b1_;
                    let _e970 = b2_;
                    let _e974 = b3_;
                    tone_2 = (_e962 + ((_e963 * _e964) * (((0.24f * _e967) + (0.18f * _e970)) + (0.12f * _e974))));
                }
            }
        }
    }
    let _e981 = slope_4;
    let _e982 = mu_sstep(0.22f, 0.42f, _e981);
    let _e986 = slope_4;
    let _e987 = mu_sstep(0.8f, 1.2f, _e986);
    let _e992 = grass_3;
    talus = ((_e982 * (1f - _e987)) * (1f - (0.85f * _e992)));
    let _e997 = talus;
    let _e999 = soil;
    clastic = max(_e997, (0.55f * _e999));
    loop {
        let _e1005 = k_12;
        if !((_e1005 < 3i)) {
            break;
        }
        {
            let _e1012 = k_12;
            let _e1015 = talus;
            let _e1018 = fp_7;
            let _e1024 = k_12;
            let _e1027 = talus;
            let _e1030 = fp_7;
            let _e1034 = fp_7;
            let _e1041 = k_12;
            let _e1044 = clastic;
            let _e1047 = fp_7;
            if ((((_e1012 == 0i) && !(((_e1015 > 0.01f) && (_e1018 < 1.3f)))) || ((_e1024 == 1i) && !((((_e1027 > 0.01f) && (_e1030 < 1.3f)) && (_e1034 < 0.34f))))) || ((_e1041 == 2i) && !(((_e1044 > 0.01f) && (_e1047 < 0.09f))))) {
                continue;
            }
            let _e1054 = world_31;
            let _e1055 = k_12;
            if (_e1055 == 0i) {
                local_65 = 2.9f;
            } else {
                let _e1059 = k_12;
                if (_e1059 == 1i) {
                    local_64 = 0.78f;
                } else {
                    local_64 = 0.21f;
                }
                let _e1065 = local_64;
                local_65 = _e1065;
            }
            let _e1067 = local_65;
            let _e1070 = salt_23;
            let _e1073 = k_12;
            let _e1076 = mu_surfaceStones((_e1054 / vec2(_e1067)), ((_e1070 + 51u) + u32(_e1073)));
            st = _e1076;
            let _e1078 = k_12;
            if (_e1078 == 0i) {
                let _e1081 = tone_2;
                let _e1082 = talus;
                let _e1086 = fp_7;
                let _e1087 = mu_sstep(0.6f, 1.3f, _e1086);
                let _e1091 = st;
                let _e1096 = st;
                tone_2 = (_e1081 + ((_e1082 * (1f - _e1087)) * ((0.3f * _e1091.x) - (0.2f * (0.27f - _e1096.y)))));
            } else {
                let _e1103 = k_12;
                if (_e1103 == 1i) {
                    let _e1106 = tone_2;
                    let _e1107 = talus;
                    let _e1111 = fp_7;
                    let _e1112 = mu_sstep(0.16f, 0.34f, _e1111);
                    let _e1116 = st;
                    let _e1121 = st;
                    tone_2 = (_e1106 + ((_e1107 * (1f - _e1112)) * ((0.26f * _e1116.x) - (0.16f * (0.27f - _e1121.y)))));
                } else {
                    let _e1128 = tone_2;
                    let _e1129 = clastic;
                    let _e1133 = fp_7;
                    let _e1134 = mu_sstep(0.045f, 0.09f, _e1133);
                    let _e1138 = st;
                    let _e1143 = st;
                    tone_2 = (_e1128 + ((_e1129 * (1f - _e1134)) * ((0.22f * _e1138.x) - (0.12f * (0.27f - _e1143.y)))));
                }
            }
        }
        continuing {
            let _e1009 = k_12;
            k_12 = (_e1009 + 1i);
        }
    }
    let _e1150 = hue;
    let _e1151 = tone_2;
    let _e1156 = world_31;
    let _e1157 = slope_4;
    let _e1158 = fp_7;
    let _e1159 = rock_3;
    let _e1160 = grass_3;
    let _e1161 = salt_23;
    let _e1162 = n1_;
    let _e1163 = n2_;
    let _e1164 = mu_surfaceGroundCover(_e1156, _e1157, _e1158, _e1159, _e1160, _e1161, _e1162, _e1163);
    mu_surfaceMicroTone = ((_e1150 * clamp(_e1151, 0.55f, 1.45f)) + _e1164);
    let _e1166 = mu_surfaceMicroSlope;
    let _e1167 = mu_surfaceGroundSlope;
    mu_surfaceMicroSlope = (_e1166 + _e1167);
    let _e1169 = mu_surfaceMicroTone;
    return _e1169;
}

fn mu_groundSmooth(a_15: f32, b_15: f32, x_2: f32) -> f32 {
    var a_16: f32;
    var b_16: f32;
    var x_3: f32;
    var t_15: f32;

    a_16 = a_15;
    b_16 = b_15;
    x_3 = x_2;
    let _e297 = x_3;
    let _e298 = a_16;
    let _e300 = b_16;
    let _e301 = a_16;
    t_15 = clamp(((_e297 - _e298) / (_e300 - _e301)), 0f, 1f);
    let _e308 = t_15;
    let _e309 = t_15;
    let _e313 = t_15;
    return ((_e308 * _e309) * (3f - (2f * _e313)));
}

fn mu_groundToLinear(c_14: vec3<f32>) -> vec3<f32> {
    var c_15: vec3<f32>;

    c_15 = c_14;
    let _e293 = c_15;
    return pow(max(_e293, vec3(0f)), vec3(2.2f));
}

fn mu_groundToDisplay(c_16: vec3<f32>) -> vec3<f32> {
    var c_17: vec3<f32>;

    c_17 = c_16;
    if (mu_uRenderV2_ > 0i) {
        let _e295 = c_17;
        return pow(max(_e295, vec3(0f)), vec3(0.45454544f));
    }
    let _e304 = c_17;
    return pow(clamp(_e304, vec3(0f), vec3(1f)), vec3(0.45454544f));
}

fn mu_groundVisibility(wavelength_6: f32, footprint_6: f32) -> f32 {
    var wavelength_7: f32;
    var footprint_7: f32;
    var t_16: f32;

    wavelength_7 = wavelength_6;
    footprint_7 = footprint_6;
    let _e302 = footprint_7;
    let _e303 = wavelength_7;
    t_16 = clamp((((_e302 / _e303) - 0.18f) / 0.24f), 0f, 1f);
    let _e314 = t_16;
    let _e315 = t_16;
    let _e317 = t_16;
    let _e319 = t_16;
    let _e320 = t_16;
    return (1f - (((_e314 * _e315) * _e317) * ((_e319 * ((_e320 * 6f) - 15f)) + 10f)));
}

fn mu_groundSlopeVariance(fp:f32)->f32{return groundSlopeVariance(fp);}

fn mu_groundField(p:vec3<f32>,seed:u32,fp:f32)->vec4<f32>{var clasts=0.;return groundFieldWithClasts(p.xz,seed,fp,&clasts);}

fn mu_groundBrdf(m_2: mu_GroundMaterial, view: vec3<f32>, light_1: vec3<f32>) -> vec3<f32> {
    var m_3: mu_GroundMaterial;
    var view_1: vec3<f32>;
    var light_2: vec3<f32>;
    var n_11: vec3<f32>;
    var v_12: vec3<f32>;
    var l: vec3<f32>;
    var halfVector: vec3<f32>;
    var nv: f32;
    var nl: f32;
    var nh: f32;
    var vh: f32;
    var a_20: f32;
    var a2_: f32;
    var denominator: f32;
    var distribution: f32;
    var visibility: f32;
    var fresnel: f32;

    m_3 = m_2;
    view_1 = view;
    light_2 = light_1;
    let _e304 = m_3;
    n_11 = _e304.normal;
    let _e307 = view_1;
    v_12 = normalize(_e307);
    let _e310 = light_2;
    l = normalize(_e310);
    let _e313 = v_12;
    let _e314 = l;
    halfVector = normalize((_e313 + _e314));
    let _e318 = n_11;
    let _e319 = v_12;
    nv = max(dot(_e318, _e319), 0.001f);
    let _e324 = n_11;
    let _e325 = l;
    nl = max(dot(_e324, _e325), 0f);
    let _e330 = n_11;
    let _e331 = halfVector;
    nh = max(dot(_e330, _e331), 0f);
    let _e336 = v_12;
    let _e337 = halfVector;
    vh = max(dot(_e336, _e337), 0f);
    let _e342 = m_3;
    let _e344 = m_3;
    a_20 = (_e342.roughness * _e344.roughness);
    let _e348 = a_20;
    let _e349 = a_20;
    a2_ = (_e348 * _e349);
    let _e352 = nh;
    let _e353 = nh;
    let _e355 = a2_;
    denominator = (((_e352 * _e353) * (_e355 - 1f)) + 1f);
    let _e362 = a2_;
    let _e364 = denominator;
    let _e366 = denominator;
    distribution = (_e362 / ((3.1415927f * _e364) * _e366));
    let _e371 = nl;
    let _e372 = nv;
    let _e373 = nv;
    let _e376 = a2_;
    let _e379 = a2_;
    let _e383 = nv;
    let _e384 = nl;
    let _e385 = nl;
    let _e388 = a2_;
    let _e391 = a2_;
    visibility = (0.5f / max(((_e371 * sqrt((((_e372 * _e373) * (1f - _e376)) + _e379))) + (_e383 * sqrt((((_e384 * _e385) * (1f - _e388)) + _e391)))), 0.001f));
    let _e403 = vh;
    fresnel = (0.04f + (0.96f * pow((1f - _e403), 5f)));
    let _e411 = fresnel;
    let _e413 = m_3;
    let _e419 = distribution;
    let _e420 = visibility;
    let _e422 = fresnel;
    let _e426 = nl;
    return (((((1f - _e411) * _e413.albedo) / vec3(3.1415927f)) + vec3(((_e419 * _e420) * _e422))) * _e426);
}

fn mu_shadeTerrainGround(m_4: mu_GroundMaterial, view_2: vec3<f32>, light_3: vec3<f32>, light2_: vec3<f32>, strength: f32) -> vec3<f32> {
    var m_5: mu_GroundMaterial;
    var view_3: vec3<f32>;
    var light_4: vec3<f32>;
    var light2_1: vec3<f32>;
    var strength_1: f32;
    var diffuse: f32;
    var amount_3: f32;
    var ambient: vec3<f32>;
    var direct: vec3<f32>;

    m_5 = m_4;
    view_3 = view_2;
    light_4 = light_3;
    light2_1 = light2_;
    strength_1 = strength;
    let _e308 = mu_global.uQuality;
    if (_e308 > 0i) {
        {
            let _e311 = m_5;
            let _e313 = light_4;
            diffuse = max(dot(_e311.normal, normalize(_e313)), 0f);
            let _e319 = m_5;
            let _e323 = m_5;
            let _e327 = diffuse;
            let _e330 = strength_1;
            let _e336 = mu_groundToDisplay((_e319.albedo * mix(1f, ((0.5f * _e323.occlusion) + (0.75f * _e327)), clamp(_e330, 0f, 1f))));
            return _e336;
        }
    }
    let _e337 = strength_1;
    amount_3 = clamp(_e337, 0f, 1f);
    let _e342 = m_5;
    let _e346 = m_5;
    let _e355 = m_5;
    ambient = ((_e342.albedo * (0.34f + (0.16f * clamp(_e346.normal.y, 0f, 1f)))) * _e355.occlusion);
    let _e359 = m_5;
    let _e360 = view_3;
    let _e361 = light_4;
    let _e362 = mu_groundBrdf(_e359, _e360, _e361);
    let _e365 = m_5;
    let _e366 = view_3;
    let _e367 = light2_1;
    let _e368 = mu_groundBrdf(_e365, _e366, _e367);
    direct = ((_e362 * 2.2f) + (_e368 * 0.58f));
    let _e373 = m_5;
    let _e375 = ambient;
    let _e376 = amount_3;
    let _e379 = direct;
    let _e380 = strength_1;
    let _e385 = mu_groundToDisplay((mix(_e373.albedo, _e375, vec3(_e376)) + (_e379 * max(_e380, 0f))));
    return _e385;
}

fn mu_forestField(p:vec2<f32>,lod:f32)->vec4<f32>{let f=r3Aux(p,2,lod);return vec4<f32>(f.rgb,v2CanopyTone);}

fn mu_forestBankCover(p:vec2<f32>,fp:f32)->f32{return 0.;}

fn mu_forestDensity(c_25: f32, kappa: f32) -> f32 {
    var c_26: f32;
    var kappa_1: f32;

    c_26 = c_25;
    kappa_1 = kappa;
    let _e317 = c_26;
    let _e318 = kappa_1;
    let _e320 = c_26;
    let _e324 = c_26;
    return clamp((_e317 + (_e318 * (0.985f - _e320))), _e324, 0.985f);
}

fn mu_forestKind(trop: f32, dry: f32) -> vec3<f32> {
    var trop_1: f32;
    var dry_1: f32;
    var savanna: f32;

    trop_1 = trop;
    dry_1 = dry;
    let _e317 = trop_1;
    let _e320 = dry_1;
    let _e321 = mu_groundSmooth(0.3f, 0.6f, _e320);
    savanna = (_e317 * _e321);
    let _e325 = trop_1;
    let _e329 = mu_global.uMaterialClimate;
    let _e331 = mu_groundSmooth(10f, 14f, _e329.x);
    let _e335 = dry_1;
    let _e336 = mu_groundSmooth(0.35f, 0.7f, _e335);
    let _e338 = trop_1;
    let _e339 = savanna;
    let _e341 = savanna;
    return vec3<f32>((((1f - _e325) * _e331) * _e336), (_e338 - _e339), _e341);
}

fn mu_forestPalette(st_11: mu_ForestStand, needle_7: f32) -> vec3<f32> {
    var st_12: mu_ForestStand;
    var needle_8: f32;
    var broad_3: vec3<f32>;

    st_12 = st_11;
    needle_8 = needle_7;
    let _e325 = st_12;
    broad_3 = mix(vec3<f32>(0.21f, 0.31f, 0.13f), vec3<f32>(0.16f, 0.3f, 0.11f), vec3(_e325.trop));
    let _e330 = broad_3;
    let _e335 = st_12;
    let _e339 = st_12;
    broad_3 = mix(_e330, vec3<f32>(0.31f, 0.32f, 0.19f), vec3((_e335.dry * (1f - (0.5f * _e339.trop)))));
    let _e346 = broad_3;
    let _e355 = st_12;
    let _e359 = needle_8;
    return mix(_e346, mix(vec3<f32>(0.15f, 0.25f, 0.16f), vec3<f32>(0.2f, 0.24f, 0.15f), vec3(_e355.dry)), vec3(_e359));
}

fn mu_forestSeason() -> vec3<f32> {
    var local_154: f32;
    var t_19: f32;
    var s_20: f32;

    if (mu_uRenderV2_ > 0i) {
        let _e316 = mu_global.uV2Season;
        local_154 = _e316;
    } else {
        local_154 = 0.5f;
    }
    let _e319 = local_154;
    t_19 = (6.2831855f * _e319);
    let _e322 = t_19;
    s_20 = sin(_e322);
    let _e325 = s_20;
    let _e329 = s_20;
    let _e334 = t_19;
    return vec3<f32>(max(-(_e325), 0f), max(_e329, 0f), (0.5f - (0.5f * cos(_e334))));
}

fn mu_forestGrade(st_13: mu_ForestStand, needle_9: f32) -> vec3<f32> {
    var st_14: mu_ForestStand;
    var needle_10: f32;
    var kind_1: vec3<f32>;
    var season_1: vec3<f32>;
    var temperate: vec3<f32>;
    var savanna_1: vec3<f32>;
    var broad_4: vec3<f32>;

    st_14 = st_13;
    needle_10 = needle_9;
    let _e317 = st_14;
    let _e319 = st_14;
    let _e321 = mu_forestKind(_e317.trop, _e319.dry);
    kind_1 = _e321;
    let _e323 = mu_forestSeason();
    season_1 = _e323;
    let _e333 = season_1;
    let _e342 = season_1;
    temperate = mix(mix(vec3<f32>(1.02f, 1.04f, 0.92f), vec3<f32>(1.1f, 1.1f, 0.82f), vec3(_e333.y)), vec3<f32>(1.5f, 0.98f, 0.5f), vec3((0.75f * _e342.x)));
    let _e356 = season_1;
    savanna_1 = mix(vec3<f32>(1.12f, 1f, 0.66f), vec3<f32>(0.94f, 1.02f, 0.76f), vec3(_e356.z));
    let _e361 = temperate;
    let _e363 = kind_1;
    let _e366 = kind_1;
    let _e369 = kind_1;
    let _e377 = kind_1;
    let _e385 = kind_1;
    let _e389 = savanna_1;
    let _e390 = kind_1;
    broad_4 = ((((_e361 * (((1f - _e363.x) - _e366.y) - _e369.z)) + (vec3<f32>(0.58f, 0.68f, 0.66f) * _e377.x)) + (vec3<f32>(0.7f, 1.06f, 0.62f) * _e385.y)) + (_e389 * _e390.z));
    let _e395 = broad_4;
    let _e404 = season_1;
    let _e408 = needle_10;
    return mix(_e395, mix(vec3<f32>(0.6f, 0.7f, 0.7f), vec3<f32>(0.62f, 0.76f, 0.74f), vec3(_e404.z)), vec3(_e408));
}

fn mu_forestSnowed(crown: vec3<f32>, load: f32) -> vec3<f32> {
    var crown_1: vec3<f32>;
    var load_1: f32;

    crown_1 = crown;
    load_1 = load;
    let _e317 = crown_1;
    let _e322 = load_1;
    return mix(_e317, vec3<f32>(0.84f, 0.88f, 0.9f), vec3(_e322));
}

fn mu_forestCanopy(st_15: mu_ForestStand, needle_11: f32, jitter_1: vec2<f32>, special: f32) -> vec3<f32> {
    var st_16: mu_ForestStand;
    var needle_12: f32;
    var jitter_2: vec2<f32>;
    var special_1: f32;
    var c_31: vec3<f32>;
    var kind_2: vec3<f32>;
    var temperate_1: f32;
    var t_20: f32;
    var amp_1: f32;
    var local_156: vec3<f32>;
    var local_157: vec3<f32>;
    var local_158: vec3<f32>;
    var accent: vec3<f32>;

    st_16 = st_15;
    needle_12 = needle_11;
    jitter_2 = jitter_1;
    special_1 = special;
    let _e321 = st_16;
    let _e322 = needle_12;
    let _e323 = mu_forestPalette(_e321, _e322);
    c_31 = _e323;
    let _e325 = st_16;
    let _e327 = st_16;
    let _e329 = mu_forestKind(_e325.trop, _e327.dry);
    kind_2 = _e329;
    let _e332 = kind_2;
    let _e335 = kind_2;
    let _e338 = kind_2;
    temperate_1 = (((1f - _e332.x) - _e335.y) - _e338.z);
    let _e342 = st_16;
    t_20 = (_e342.tone - 0.5f);
    let _e347 = c_31;
    let _e350 = t_20;
    c_31 = (_e347 * (1f + (0.34f * _e350)));
    let _e354 = c_31;
    let _e355 = c_31;
    let _e361 = t_20;
    c_31 = mix(_e354, (_e355 * vec3<f32>(1.12f, 1.07f, 0.78f)), vec3((max(_e361, 0f) * 0.9f)));
    let _e368 = c_31;
    let _e369 = c_31;
    let _e375 = t_20;
    c_31 = mix(_e368, (_e369 * vec3<f32>(0.9f, 0.97f, 1.08f)), vec3((max(-(_e375), 0f) * 0.7f)));
    let _e385 = temperate_1;
    let _e388 = st_16;
    let _e393 = needle_12;
    amp_1 = (mix(mix(0.09f, 0.14f, _e385), 0.2f, _e388.trop) * (1f - (0.45f * _e393)));
    let _e398 = c_31;
    let _e400 = amp_1;
    let _e401 = jitter_2;
    c_31 = (_e398 * (1f + (_e400 * _e401.x)));
    let _e406 = c_31;
    let _e407 = c_31;
    let _e413 = jitter_2;
    let _e417 = amp_1;
    c_31 = mix(_e406, (_e407 * vec3<f32>(1.12f, 1.06f, 0.72f)), vec3(((max(_e413.y, 0f) * _e417) * 2.2f)));
    let _e423 = c_31;
    let _e424 = c_31;
    let _e430 = jitter_2;
    let _e435 = amp_1;
    c_31 = mix(_e423, (_e424 * vec3<f32>(0.84f, 0.97f, 1.12f)), vec3(((max(-(_e430.y), 0f) * _e435) * 1.6f)));
    let _e441 = c_31;
    let _e444 = jitter_2;
    if (_e444.y > 0f) {
        local_156 = vec3<f32>(1.3f, 1.04f, 0.62f);
    } else {
        local_156 = vec3<f32>(1.3f, 0.72f, 0.66f);
    }
    let _e457 = local_156;
    let _e458 = jitter_2;
    let _e461 = mu_forestSeason();
    let _e464 = temperate_1;
    let _e467 = needle_12;
    let _e472 = jitter_2;
    c_31 = (_e441 * mix(vec3(1f), _e457, vec3(((((abs(_e458.y) * _e461.x) * _e464) * (1f - _e467)) * (0.6f + (0.4f * _e472.x))))));
    let _e480 = special_1;
    if (_e480 < 0.45f) {
        local_158 = vec3<f32>(0.4f, 0.42f, 0.19f);
    } else {
        let _e487 = special_1;
        if (_e487 < 0.8f) {
            local_157 = vec3<f32>(0.4f, 0.3f, 0.22f);
        } else {
            local_157 = vec3<f32>(0.36f, 0.38f, 0.32f);
        }
        let _e499 = local_157;
        local_158 = _e499;
    }
    let _e501 = local_158;
    accent = _e501;
    let _e503 = c_31;
    let _e504 = accent;
    let _e505 = st_16;
    let _e509 = special_1;
    let _e513 = mu_groundSmooth(0.965f, 0.99f, fract((_e509 * 7.31f)));
    return mix(_e503, _e504, vec3(((_e505.trop * _e513) * 0.6f)));
}

fn mu_forestCrownColor(st_17: mu_ForestStand, guide: vec3<f32>, needle_13: f32, jitter_3: vec2<f32>, special_2: f32) -> vec3<f32> {
    var st_18: mu_ForestStand;
    var guide_1: vec3<f32>;
    var needle_14: f32;
    var jitter_4: vec2<f32>;
    var special_3: f32;

    st_18 = st_17;
    guide_1 = guide;
    needle_14 = needle_13;
    jitter_4 = jitter_3;
    special_3 = special_2;
    let _e323 = st_18;
    if _e323.orchard {
        let _e333 = mu_forestSeason();
        let _e339 = jitter_4;
        return (mix(vec3<f32>(0.25f, 0.34f, 0.15f), vec3<f32>(0.22f, 0.37f, 0.16f), vec3(_e333.z)) * (0.86f + (0.28f * ((_e339.x * 0.5f) + 0.5f))));
    }
    let _e348 = guide_1;
    let _e349 = st_18;
    let _e350 = needle_14;
    let _e351 = mu_forestGrade(_e349, _e350);
    let _e353 = st_18;
    let _e354 = needle_14;
    let _e355 = jitter_4;
    let _e356 = special_3;
    let _e357 = mu_forestCanopy(_e353, _e354, _e355, _e356);
    let _e359 = st_18;
    let _e360 = st_18;
    let _e362 = mu_forestPalette(_e359, _e360.needle);
    return (((_e348 * _e351) * _e357) / max(_e362, vec3(0.02f)));
}

fn mu_forestFloor(ground_1: vec3<f32>, st_19: mu_ForestStand, cold: f32) -> vec3<f32> {
    var ground_2: vec3<f32>;
    var st_20: mu_ForestStand;
    var cold_1: f32;
    var litter: vec3<f32>;
    var kind_3: vec3<f32>;

    ground_2 = ground_1;
    st_20 = st_19;
    cold_1 = cold;
    let _e327 = st_20;
    litter = mix(vec3<f32>(0.26f, 0.21f, 0.15f), vec3<f32>(0.24f, 0.21f, 0.15f), vec3(_e327.needle));
    let _e332 = st_20;
    let _e334 = st_20;
    let _e336 = mu_forestKind(_e332.trop, _e334.dry);
    kind_3 = _e336;
    let _e338 = litter;
    let _e343 = kind_3;
    litter = mix(_e338, vec3<f32>(0.15f, 0.24f, 0.1f), vec3(_e343.y));
    let _e347 = litter;
    let _e352 = kind_3;
    let _e354 = kind_3;
    litter = mix(_e347, vec3<f32>(0.6f, 0.54f, 0.42f), vec3((_e352.x + _e354.z)));
    let _e359 = litter;
    let _e369 = st_20;
    let _e374 = cold_1;
    let _e375 = st_20;
    litter = mix(_e359, mix(vec3<f32>(0.27f, 0.32f, 0.18f), vec3<f32>(0.58f, 0.58f, 0.48f), vec3((0.7f * _e369.dry))), vec3((_e374 * _e375.needle)));
    if (mu_uRenderV2_ > 0i) {
        {
            let _e382 = ground_2;
            let _e385 = mu_v2GroundSnow;
            let _e386 = st_20;
            ground_2 = (_e382 * mix(1f, 0.42f, (_e385 * _e386.needle)));
            let _e391 = litter;
            let _e392 = ground_2;
            let _e393 = mu_v2GroundSnow;
            let _e396 = st_20;
            litter = mix(_e391, _e392, vec3((_e393 * (1f - (0.6f * _e396.needle)))));
        }
    }
    let _e403 = ground_2;
    let _e404 = litter;
    let _e408 = st_20;
    return mix(_e403, _e404, vec3((0.65f * (1f - (0.6f * _e408.dry)))));
}

fn mu_forestFloorAlbedo(color_5: vec3<f32>) -> vec3<f32> {
    var color_6: vec3<f32>;
    var local_159: f32;

    color_6 = color_5;
    let _e315 = color_6;
    if (mu_uRenderV2_ > 0i) {
        let _e318 = mu_v2FloorModulation;
        local_159 = _e318;
    } else {
        local_159 = 1f;
    }
    let _e321 = local_159;
    let _e323 = mu_groundToLinear((_e315 * _e321));
    return _e323;
}

fn mu_forestClosure(cover_5: f32) -> f32 {
    var cover_6: f32;

    cover_6 = cover_5;
    let _e317 = cover_6;
    let _e318 = mu_groundSmooth(0.12f, 0.45f, _e317);
    return _e318;
}

fn mu_v2Pattern(p:vec3<f32>,scale:f32,fp:f32,salt:u32)->f32{return magnaPattern(p,scale,fp,salt);}

fn mu_v2Appearance(e_4: mu_V2Env, fp_24: f32) -> mu_V2Env {
    var e_5: mu_V2Env;
    var fp_25: f32;
    var shown: mu_V2Env;

    e_5 = e_4;
    fp_25 = fp_24;
    let _e319 = e_5;
    shown = _e319;
    let _e323 = e_5;
    let _e328 = fp_25;
    let _e329 = mu_sstep(600f, 2400f, _e328);
    shown.mineral = mix(0.5f, _e323.mineral, (1f - _e329));
    let _e334 = e_5;
    let _e339 = fp_25;
    let _e340 = mu_sstep(120f, 500f, _e339);
    shown.tone = mix(0.5f, _e334.tone, (1f - _e340));
    let _e343 = shown;
    return _e343;
}

fn mu_v2Mineral(e_6: mu_V2Env) -> vec3<f32> {
    var e_7: mu_V2Env;
    var dark: vec3<f32> = vec3<f32>(0.19f, 0.19f, 0.175f);
    var brown: vec3<f32> = vec3<f32>(0.39f, 0.335f, 0.27f);
    var cream: vec3<f32> = vec3<f32>(0.68f, 0.655f, 0.57f);
    var red: vec3<f32> = vec3<f32>(0.49f, 0.37f, 0.285f);

    e_7 = e_6;
    let _e337 = dark;
    let _e338 = brown;
    let _e341 = e_7;
    let _e343 = mu_sstep(0.08f, 0.35f, _e341.mineral);
    let _e346 = cream;
    let _e347 = red;
    let _e350 = e_7;
    let _e352 = mu_sstep(0.68f, 0.94f, _e350.mineral);
    let _e357 = e_7;
    let _e359 = mu_sstep(0.34f, 0.66f, _e357.mineral);
    return mix(mix(_e337, _e338, vec3(_e343)), mix(_e346, _e347, vec3(_e352)), vec3(_e359));
}

fn mu_v2GroundTint(base_3: vec3<f32>, e_8: mu_V2Env) -> vec3<f32> {
    var base_4: vec3<f32>;
    var e_9: mu_V2Env;
    var neutral: mu_V2Env;
    var mineralRatio: vec3<f32>;
    var mineralWeight: f32;

    base_4 = base_3;
    e_9 = e_8;
    let _e319 = e_9;
    neutral = _e319;
    neutral.mineral = 0.5f;
    let _e323 = e_9;
    let _e324 = mu_v2Mineral(_e323);
    let _e325 = neutral;
    let _e326 = mu_v2Mineral(_e325);
    mineralRatio = clamp((_e324 / max(_e326, vec3(0.08f))), vec3(0.6f), vec3(1.4f));
    let _e338 = e_9;
    let _e341 = e_9;
    let _e345 = e_9;
    let _e350 = e_9;
    let _e359 = e_9;
    mineralWeight = ((0.35f * (_e338.rock + ((((1f - _e341.rock) * (1f - _e345.grass)) * (1f - _e350.organic)) * 0.35f))) * (1f - _e359.snow));
    let _e364 = base_4;
    let _e367 = mineralRatio;
    let _e368 = mineralWeight;
    let _e372 = e_9;
    let _e374 = e_9;
    let _e376 = e_9;
    let _e378 = e_9;
    let _e383 = e_9;
    let _e385 = e_9;
    let _e387 = mu_surfaceMicro(_e372.world, _e374.slope, _e376.footprint, clamp(_e378.rock, 0f, 1f), _e383.grass, _e385.seed);
    return ((_e364 * mix(vec3(1f), _e367, vec3(_e368))) * _e387);
}

fn mu_v2DisplayShoulder(rgb: vec3<f32>) -> vec3<f32> {
    var rgb_1: vec3<f32>;
    var peak: f32;
    var mapped: f32;

    rgb_1 = rgb;
    let _e317 = rgb_1;
    let _e319 = rgb_1;
    let _e321 = rgb_1;
    peak = max(_e317.x, max(_e319.y, _e321.z));
    let _e326 = peak;
    if (_e326 <= 0.8f) {
        {
            let _e329 = rgb_1;
            return _e329;
        }
    }
    let _e333 = peak;
    mapped = (0.8f + (0.2f * (1f - exp((-((_e333 - 0.8f)) / 0.2f)))));
    let _e344 = rgb_1;
    let _e345 = mapped;
    let _e346 = peak;
    return (_e344 * (_e345 / _e346));
}

fn mu_v2Canopy(e_12: mu_V2Env) -> vec3<f32> {
    var e_13: mu_V2Env;
    var evergreen: vec3<f32>;
    var senescent: vec3<f32>;
    var c_45: vec3<f32>;

    e_13 = e_12;
    let _e325 = e_13;
    evergreen = mix(vec3<f32>(0.16f, 0.235f, 0.14f), vec3<f32>(0.12f, 0.205f, 0.15f), vec3(_e325.needle));
    let _e340 = e_13;
    let _e342 = mu_sstep(4f, 12f, _e340.temperature);
    senescent = mix(vec3<f32>(0.34f, 0.29f, 0.215f), vec3<f32>(0.43f, 0.335f, 0.205f), vec3(_e342));
    let _e346 = senescent;
    let _e347 = evergreen;
    let _e350 = e_13;
    let _e352 = mu_sstep(0.3f, 0.9f, _e350.leaf);
    let _e357 = e_13;
    c_45 = (mix(_e346, _e347, vec3(_e352)) * (0.85f + (0.3f * _e357.tone)));
    let _e363 = c_45;
    let _e368 = e_13;
    let _e372 = e_13;
    return mix(_e363, vec3<f32>(0.72f, 0.79f, 0.77f), vec3((_e368.snow * mix(0.62f, 0.08f, _e372.needle))));
}

fn mu_v2CrownResponse(light_11: vec3<f32>, needle_15: f32) -> vec2<f32> {
    var light_12: vec3<f32>;
    var needle_16: f32;
    var level_9: f32;
    var i_14: i32;
    var t_28: f32;
    var local_260: array<vec4<f32>, 19> = mu_V2_CROWN_RESPONSE;
    var local_261: array<vec4<f32>, 19> = mu_V2_CROWN_RESPONSE;
    var q_21: vec4<f32>;

    light_12 = light_11;
    needle_16 = needle_15;
    let _e322 = light_12;
    level_9 = (asin(clamp(normalize(_e322).y, 0f, 1f)) * 11.459156f);
    let _e332 = level_9;
    i_14 = min(i32(floor(_e332)), 17i);
    let _e338 = level_9;
    let _e339 = i_14;
    t_28 = clamp((_e338 - f32(_e339)), 0f, 1f);
    let _e346 = i_14;
    let _e350 = local_260[_e346];
    let _e351 = i_14;
    let _e357 = local_261[(_e351 + 1i)];
    let _e358 = t_28;
    q_21 = mix(_e350, _e357, vec4(_e358));
    let _e362 = q_21;
    let _e364 = q_21;
    let _e366 = needle_16;
    return mix(_e362.xy, _e364.zw, vec2(_e366));
}

fn mu_v2CanopyMeanRatio(rgb_2: vec3<f32>, needle_17: f32, light_13: vec3<f32>, light2_2: vec3<f32>, strength_2: f32) -> vec3<f32> {
    var rgb_3: vec3<f32>;
    var needle_18: f32;
    var light_14: vec3<f32>;
    var light2_3: vec3<f32>;
    var strength_3: f32;
    var amount_4: f32;
    var albedo_1: vec3<f32>;
    var response: vec2<f32>;
    var ambient_1: f32;
    var mean_3: vec3<f32>;
    var referenceMaterial: mu_GroundMaterial;
    var base_5: vec3<f32>;

    rgb_3 = rgb_2;
    needle_18 = needle_17;
    light_14 = light_13;
    light2_3 = light2_2;
    strength_3 = strength_2;
    let _e328 = strength_3;
    amount_4 = clamp(_e328, 0f, 1f);
    let _e333 = rgb_3;
    let _e334 = mu_groundToLinear(_e333);
    albedo_1 = _e334;
    let _e337 = light_14;
    let _e338 = needle_18;
    let _e339 = mu_v2CrownResponse(_e337, _e338);
    let _e342 = light2_3;
    let _e343 = needle_18;
    let _e344 = mu_v2CrownResponse(_e342, _e343);
    response = ((2.2f * _e339) + (0.58f * _e344));
    let _e354 = needle_18;
    ambient_1 = mix(0.3766294f, 0.34189487f, _e354);
    let _e357 = albedo_1;
    let _e359 = amount_4;
    let _e361 = amount_4;
    let _e362 = ambient_1;
    let _e366 = strength_3;
    let _e369 = albedo_1;
    let _e370 = response;
    let _e373 = response;
    mean_3 = ((_e357 * ((1f - _e359) + (_e361 * _e362))) + (max(_e366, 0f) * ((_e369 * _e370.x) + vec3(_e373.y))));
    let _e382 = albedo_1;
    referenceMaterial.albedo = _e382;
    referenceMaterial.normal = vec3<f32>(0f, 1f, 0f);
    referenceMaterial.roughness = 0.9f;
    referenceMaterial.occlusion = 1f;
    let _e392 = referenceMaterial;
    let _e397 = light_14;
    let _e398 = light2_3;
    let _e399 = strength_3;
    let _e400 = mu_shadeTerrainGround(_e392, vec3<f32>(0f, 1f, 0f), _e397, _e398, _e399);
    base_5 = _e400;
    let _e402 = mean_3;
    let _e411 = base_5;
    return (pow(max(_e402, vec3(0f)), vec3(0.45454544f)) / max(_e411, vec3(0.0001f)));
}

fn mu_v2FlatSedimentExposure(herbs_1: f32, shore_15: f32, openness_12: f32) -> f32 {
    var herbs_2: f32;
    var shore_16: f32;
    var openness_13: f32;
    var coastal: f32;

    herbs_2 = herbs_1;
    shore_16 = shore_15;
    openness_13 = openness_12;
    let _e331 = shore_16;
    let _e332 = mu_groundSmooth(160f, 400f, _e331);
    let _e336 = openness_13;
    let _e337 = mu_groundSmooth(0f, 0.15f, _e336);
    coastal = ((1f - _e332) * _e337);
    let _e341 = herbs_2;
    let _e343 = coastal;
    return (1f - (_e341 * (1f - _e343)));
}

fn mu_v2ZoneOpen(c:vec3<f32>,e:mu_V2Env,p:vec2<f32>,fp:f32)->vec3<f32>{return c;}

fn mu_v2GroundEnv(local_274: mu_V2Env, s_31: vec3<f32>, world_100: vec2<f32>, footprint_54: f32, light_15: vec3<f32>, light2_4: vec3<f32>, strength_4: f32, macroOpen: vec3<f32>) -> mu_GroundMaterial {
    var local_275: mu_V2Env;
    var s_32: vec3<f32>;
    var world_101: vec2<f32>;
    var footprint_55: f32;
    var light_16: vec3<f32>;
    var light2_5: vec3<f32>;
    var strength_5: f32;
    var macroOpen_1: vec3<f32>;
    var macroWoods: f32;
    var baseClosed: vec3<f32>;
    var local_276: array<vec4<f32>, 11> = mu_V2_ZONE_FOREST;
    var local_277: array<vec4<f32>, 11> = mu_V2_ZONE_CANOPY;
    var woodsUv: vec2<f32>;
    var snowStands: f32 = 0f;
    var cell_20: f32;
    var stands_1: f32;
    var e_28: mu_V2Env;
    var local_278: f32;
    var woods_2: f32;
    var opened: vec3<f32>;
    var snowDrift: f32 = 0f;
    var snowGrain_2: f32 = 0f;
    var snowColor: vec3<f32>;
    var canopyPhase: mu_V2Env;
    var leafOn: mu_V2Env;
    var closed: vec3<f32>;
    var frost_1: f32;
    var meanClosed: vec3<f32>;
    var m_13: mu_GroundMaterial;
    var detail_5: vec4<f32>;
    var gain_1: f32;
    var sward_1: f32;
    var form_1: f32;
    var standFloor: vec3<f32>;
    var crownGrain: f32 = 1f;
    var forestUv: vec2<f32>;
    var cell_21: f32;
    var f_24: vec4<f32>;
    var st_27: mu_ForestStand;
    var T_1: f32;
    var humid_4: f32;
    var kind_14: vec3<f32>;
    var d_24: f32;
    var share_3: f32;
    var available_1: f32;
    var canopy_2: f32;
    var bareGuide: vec3<f32>;
    var crownBare: vec3<f32>;
    var crownMean: vec3<f32>;

    local_275 = local_274;
    s_32 = s_31;
    world_101 = world_100;
    footprint_55 = footprint_54;
    light_16 = light_15;
    light2_5 = light2_4;
    strength_5 = strength_4;
    macroOpen_1 = macroOpen;
    let _e341 = mu_renderGuideWoods;
    let _e343 = mu_renderGuideBeach;
    macroWoods = (_e341 * (1f - clamp((_e343 / 0.8f), 0f, 1f)));
    let _e352 = mu_renderGuideBaseClosed;
    baseClosed = _e352;
    let _e354 = mu_v2Zoned();
    if _e354 {
        {
            let _e355 = macroWoods;
            let _e356 = mu_v2Zone();
            let _e360 = local_276[_e356];
            macroWoods = min((_e355 * _e360.x), 1f);
            let _e365 = baseClosed;
            let _e366 = mu_v2Zone();
            let _e370 = local_277[_e366];
            baseClosed = (_e365 * _e370.xyz);
        }
    }
    let _e373 = world_101;
    let _e374 = mu_global.uForestRect;
    let _e377 = mu_global.uForestRect;
    woodsUv = ((_e373 - _e374.xy) / vec2(_e377.z));
    let _e386 = local_275;
    let _e391 = woodsUv;
    let _e397 = woodsUv;
    if ((((mu_uForest == 1i) && (_e386.snow > 0f)) && all((_e391 >= vec2(0f)))) && all((_e397 <= vec2(1f)))) {
        {
            let _e403 = mu_global.uForestRect;
            let _e406 = textureDimensions(aux, 0);
            cell_20 = (_e403.z / f32(vec2<i32>(_e406).x));
            let _e412 = world_101;
            let _e413 = footprint_55;
            let _e416 = cell_20;
            let _e421 = mu_forestField(_e412, max(log2((max(_e413, 0.02f) / _e416)), 0f));
            let _e424 = mu_renderGuideBeach;
            stands_1 = (_e421.x * (1f - clamp((_e424 / 0.8f), 0f, 1f)));
            let _e433 = macroWoods;
            let _e434 = macroWoods;
            let _e435 = stands_1;
            let _e437 = local_275;
            macroWoods = mix(_e433, max(_e434, _e435), _e437.snow);
            let _e440 = stands_1;
            let _e441 = local_275;
            snowStands = (_e440 * _e441.snow);
        }
    }
    let _e444 = local_275;
    let _e445 = footprint_55;
    let _e446 = mu_v2Appearance(_e444, _e445);
    e_28 = _e446;
    let _e448 = e_28;
    mu_v2Foliage = _e448.leaf;
    let _e450 = e_28;
    mu_v2GroundSnow = _e450.snow;
    if (mu_uForest == 1i) {
        let _e454 = macroWoods;
        local_278 = _e454;
    } else {
        let _e455 = e_28;
        local_278 = _e455.woody;
    }
    let _e458 = local_278;
    let _e459 = e_28;
    let _e463 = mu_v2ZoneBare;
    woods_2 = ((_e458 * _e459.leaf) * (1f - _e463));
    let _e467 = woods_2;
    let _e468 = snowStands;
    let _e469 = e_28;
    woods_2 = max(_e467, (_e468 * _e469.leaf));
    let _e473 = macroOpen_1;
    let _e474 = e_28;
    let _e475 = mu_v2GroundTint(_e473, _e474);
    opened = _e475;
    let _e477 = mu_v2Zoned();
    if _e477 {
        {
            let _e478 = opened;
            let _e479 = e_28;
            let _e480 = world_101;
            let _e481 = footprint_55;
            let _e482 = mu_v2ZoneOpen(_e478, _e479, _e480, _e481);
            opened = _e482;
            let _e483 = opened;
            let _e486 = mu_v2SeasonHerb();
            let _e487 = e_28;
            let _e492 = e_28;
            opened = (_e483 * mix(vec3(1f), _e486, vec3((max(_e487.grass, 0.3f) * (1f - _e492.rock)))));
        }
    }
    let _e499 = opened;
    let _e500 = mu_renderGuideSediment;
    let _e503 = mu_renderGuideFlats;
    let _e504 = e_28;
    let _e506 = mu_renderGuideShore;
    let _e507 = mu_renderGuideOpenness;
    let _e508 = mu_v2FlatSedimentExposure(_e504.grass, _e506, _e507);
    opened = mix(_e499, (_e500 * 0.9f), vec3((_e503 * _e508)));
    let _e516 = mu_global.uQuality;
    let _e519 = e_28;
    if ((_e516 == 0i) && (_e519.snow > 0f)) {
        {
            let _e532 = world_101;
            let _e533 = (mat2x2<f32>(vec2<f32>(0.8f, 0.6f), vec2<f32>(-0.6f, 0.8f)) * _e532);
            let _e539 = footprint_55;
            let _e540 = mu_global.uMaterialSeed;
            let _e543 = mu_v2Pattern(vec3<f32>(_e533.x, _e533.y, 0f), 11f, _e539, (_e540 + 726u));
            snowDrift = _e543;
            let _e544 = world_101;
            let _e550 = footprint_55;
            let _e551 = mu_global.uMaterialSeed;
            let _e554 = mu_v2Pattern(vec3<f32>(_e544.x, _e544.y, 0f), 3.2f, _e550, (_e551 + 727u));
            snowGrain_2 = _e554;
        }
    }
    let _e561 = snowDrift;
    let _e571 = snowGrain_2;
    snowColor = (vec3<f32>(0.88f, 0.91f, 0.91f) * ((1f + (0.035f * clamp((_e561 * 1.7f), -1f, 1f))) + (0.018f * _e571)));
    let _e576 = opened;
    let _e577 = snowColor;
    let _e578 = e_28;
    opened = mix(_e576, _e577, vec3(_e578.snow));
    let _e582 = opened;
    let _e583 = mu_renderGuideSediment;
    let _e584 = mu_renderGuideBeach;
    let _e591 = e_28;
    opened = mix(_e582, _e583, vec3((clamp((_e584 / 0.8f), 0f, 1f) * (1f - _e591.snow))));
    let _e597 = e_28;
    canopyPhase = _e597;
    canopyPhase.snow = 0f;
    let _e601 = canopyPhase;
    leafOn = _e601;
    leafOn.leaf = 1f;
    let _e605 = baseClosed;
    let _e606 = canopyPhase;
    let _e607 = mu_v2Canopy(_e606);
    let _e609 = leafOn;
    let _e610 = mu_v2Canopy(_e609);
    closed = ((_e605 * _e607) / max(_e610, vec3(0.02f)));
    let _e616 = closed;
    mu_v2CrownGuide = _e616;
    let _e617 = e_28;
    let _e621 = e_28;
    frost_1 = (_e617.snow * mix(0.62f, 0.22f, _e621.needle));
    let _e626 = closed;
    meanClosed = _e626;
    let _e628 = mu_global.uQuality;
    if (_e628 == 0i) {
        let _e631 = closed;
        let _e634 = closed;
        let _e635 = e_28;
        let _e637 = light_16;
        let _e638 = light2_5;
        let _e639 = strength_5;
        let _e640 = mu_v2CanopyMeanRatio(_e634, _e635.needle, _e637, _e638, _e639);
        let _e642 = e_28;
        let _e645 = e_28;
        meanClosed = (_e631 * mix(vec3(1f), _e640, vec3((1f - (_e642.snow * (1f - _e645.needle))))));
    }
    let _e653 = closed;
    let _e658 = frost_1;
    closed = mix(_e653, vec3<f32>(0.72f, 0.79f, 0.77f), vec3(_e658));
    let _e661 = meanClosed;
    let _e666 = frost_1;
    meanClosed = mix(_e661, vec3<f32>(0.72f, 0.79f, 0.77f), vec3(_e666));
    let _e669 = meanClosed;
    let _e670 = mu_renderGuideSediment;
    let _e673 = mu_renderGuideFlats;
    let _e675 = e_28;
    mu_v2MacroCanopy = mix(_e669, (_e670 * 0.9f), vec3((_e673 * (1f - _e675.snow))));
    let _e682 = world_101;
    let _e685 = world_101;
    let _e688 = mu_global.uMaterialSeed;
    let _e689 = footprint_55;
    let _e690 = mu_groundField(vec3<f32>(_e682.x, 0f, _e685.y), _e688, _e689);
    detail_5 = _e690;
    let _e694 = e_28;
    let _e700 = e_28;
    let _e706 = e_28;
    gain_1 = mix(((0.02f + (0.43f * _e694.rock)) + (0.07f * (1f - _e700.grass))), 0.008f, _e706.snow);
    let _e711 = s_32;
    let _e714 = detail_5;
    let _e716 = gain_1;
    let _e720 = s_32;
    let _e723 = detail_5;
    let _e725 = gain_1;
    m_13.normal = normalize(vec3<f32>((-(_e711.y) - (_e714.y * _e716)), 1f, (-(_e720.z) - (_e723.w * _e725))));
    let _e730 = detail_5;
    let _e740 = e_28;
    let _e744 = e_28;
    let _e749 = footprint_55;
    let _e750 = mu_groundVisibility(0.32f, _e749);
    sward_1 = ((((clamp((_e730.x * 4f), -1f, 1f) * 0.025f) * _e740.grass) * (1f - _e744.snow)) * _e750);
    let _e753 = detail_5;
    form_1 = clamp((_e753.x * 10f), -1f, 1f);
    let _e763 = form_1;
    let _e766 = e_28;
    let _e770 = e_28;
    let _e775 = sward_1;
    mu_v2FloorModulation = ((1f + (_e763 * mix(mix(0.04f, 0.06f, _e766.rock), 0.006f, _e770.snow))) + _e775);
    let _e777 = s_32;
    let _e781 = s_32;
    mu_v2MacroNormal = normalize(vec3<f32>(-(_e777.y), 1f, -(_e781.z)));
    let _e786 = opened;
    let _e787 = opened;
    let _e798 = snowStands;
    let _e799 = e_28;
    standFloor = mix(_e786, ((_e787 * vec3<f32>(0.45f, 0.48f, 0.5f)) + vec3<f32>(0.03f, 0.04f, 0.03f)), vec3((_e798 * _e799.needle)));
    let _e807 = mu_global.uQuality;
    if (_e807 == 0i) {
        let _e811 = snowStands;
        let _e813 = world_101;
        let _e819 = footprint_55;
        let _e820 = mu_global.uMaterialSeed;
        let _e823 = mu_v2Pattern(vec3<f32>(_e813.x, _e813.y, 0f), 45f, _e819, (_e820 + 761u));
        let _e826 = world_101;
        let _e832 = footprint_55;
        let _e833 = mu_global.uMaterialSeed;
        let _e836 = mu_v2Pattern(vec3<f32>(_e826.x, _e826.y, 0f), 14f, _e832, (_e833 + 762u));
        crownGrain = (1f + (_e811 * ((0.22f * _e823) + (0.14f * _e836))));
    }
    let _e842 = standFloor;
    let _e843 = mu_forestFloorAlbedo(_e842);
    let _e844 = mu_v2MacroCanopy;
    let _e845 = mu_groundToLinear(_e844);
    let _e846 = crownGrain;
    let _e848 = woods_2;
    m_13.albedo = mix(_e843, (_e845 * _e846), vec3(_e848));
    let _e854 = e_28;
    let _e859 = gain_1;
    let _e860 = gain_1;
    let _e862 = footprint_55;
    let _e863 = mu_groundSlopeVariance(_e862);
    m_13.roughness = clamp(pow((pow(mix(0.88f, 0.54f, _e854.rock), 4f) + ((_e859 * _e860) * _e863)), 0.25f), 0.4f, 0.98f);
    let _e874 = detail_5;
    let _e882 = e_28;
    m_13.occlusion = clamp((1f - ((0.1f * max((-(_e874.x) * 12f), 0f)) * _e882.rock)), 0.8f, 1f);
    mu_v2MeanEnabled = 0f;
    mu_v2ForestReveal = 0f;
    let _e891 = m_13;
    mu_v2MeanOpen = _e891;
    let _e893 = opened;
    let _e894 = mu_forestFloorAlbedo(_e893);
    mu_v2MeanOpen.albedo = _e894;
    let _e895 = mu_v2MeanOpen;
    mu_v2MeanLitter = _e895;
    let _e896 = mu_v2MeanOpen;
    mu_v2MeanCrown = _e896;
    mu_v2MeanWeights = vec3<f32>(1f, 0f, 0f);
    let _e901 = world_101;
    let _e902 = mu_global.uForestRect;
    let _e905 = mu_global.uForestRect;
    forestUv = ((_e901 - _e902.xy) / vec2(_e905.z));
    let _e910 = mu_global.uQuality;
    let _e916 = mu_global.uHasOccupation;
    let _e920 = world_101;
    let _e922 = footprint_55;
    let _e924 = mu_forestBankCover(_e920, max(1f, _e922));
    let _e928 = forestUv;
    let _e934 = forestUv;
    if ((((((_e910 == 0i) && (mu_uForest == 1i)) && (_e916 == 0i)) && (_e924 < 0.001f)) && all((_e928 >= vec2(0f)))) && all((_e934 <= vec2(1f)))) {
        {
            let _e940 = mu_global.uForestRect;
            let _e943 = textureDimensions(aux, 0);
            cell_21 = (_e940.z / f32(vec2<i32>(_e943).x));
            let _e949 = world_101;
            let _e950 = footprint_55;
            let _e953 = cell_21;
            let _e958 = mu_forestField(_e949, max(log2((max(_e950, 0.02f) / _e953)), 0f));
            f_24 = _e958;
            st_27.orchard = false;
            st_27.rows = vec2<f32>(1f, 0f);
            let _e968 = world_101;
            st_27.p = _e968;
            let _e970 = f_24;
            st_27.cover = _e970.x;
            let _e973 = f_24;
            st_27.height = (_e973.y * 50f);
            let _e978 = f_24;
            st_27.needle = _e978.z;
            let _e981 = f_24;
            st_27.tone = _e981.w;
            let _e983 = mu_global.uMaterialClimate;
            T_1 = _e983.x;
            let _e986 = mu_global.uMaterialClimate;
            let _e989 = T_1;
            humid_4 = (_e986.y / max(((20f * _e989) + 280f), 120f));
            let _e1000 = T_1;
            let _e1003 = mu_groundSmooth(13f, 19f, (_e1000 - 6f));
            st_27.trop = _e1003;
            let _e1008 = humid_4;
            let _e1009 = mu_groundSmooth(0.8f, 2f, _e1008);
            st_27.dry = (1f - _e1009);
            let _e1011 = st_27;
            let _e1013 = st_27;
            let _e1015 = mu_forestKind(_e1011.trop, _e1013.dry);
            kind_14 = _e1015;
            let _e1022 = humid_4;
            let _e1023 = mu_groundSmooth(0.9f, 1.8f, _e1022);
            let _e1029 = st_27;
            let _e1031 = mu_groundSmooth(3f, 10f, _e1029.height);
            let _e1036 = st_27;
            let _e1042 = st_27;
            let _e1049 = kind_14;
            let _e1056 = kind_14;
            st_27.kappa = (((((mix(0.25f, 0.78f, _e1023) * mix(0.55f, 1f, _e1031)) * mix(1f, 0.85f, _e1036.needle)) * (1f - (0.6f * _e1042.trop))) * (1f - (0.5f * _e1049.x))) * (1f - (0.85f * _e1056.z)));
            st_27.open = -1000f;
            st_27.forced = -1000f;
            st_27.allow = 0f;
            st_27.hedge = 0f;
            let _e1071 = st_27;
            let _e1073 = st_27;
            let _e1075 = mu_forestDensity(_e1071.cover, _e1073.kappa);
            d_24 = _e1075;
            let _e1077 = st_27;
            let _e1079 = d_24;
            share_3 = (_e1077.cover / max(_e1079, 0.001f));
            let _e1085 = mu_renderGuideBeach;
            let _e1093 = world_101;
            let _e1094 = mu_forestWaterCoverage(_e1093);
            let _e1098 = mu_v2ZoneBare;
            available_1 = (((1f - clamp((_e1085 / 0.8f), 0f, 1f)) * (1f - _e1094)) * (1f - _e1098));
            let _e1102 = st_27;
            let _e1104 = e_28;
            let _e1107 = available_1;
            canopy_2 = ((_e1102.cover * _e1104.leaf) * _e1107);
            let _e1110 = share_3;
            let _e1111 = available_1;
            let _e1113 = st_27;
            let _e1115 = mu_forestClosure(_e1113.cover);
            let _e1117 = canopy_2;
            share_3 = max(((_e1110 * _e1111) * _e1115), _e1117);
            let _e1120 = share_3;
            let _e1122 = share_3;
            let _e1123 = canopy_2;
            let _e1127 = canopy_2;
            mu_v2MeanWeights = vec3<f32>((1f - _e1120), max((_e1122 - _e1123), 0f), _e1127);
            let _e1130 = opened;
            let _e1131 = st_27;
            let _e1135 = T_1;
            let _e1138 = mu_groundSmooth(8f, 16f, (_e1135 + 6f));
            let _e1140 = mu_forestFloor(_e1130, _e1131, (1f - _e1138));
            let _e1141 = mu_forestFloorAlbedo(_e1140);
            mu_v2MeanLitter.albedo = _e1141;
            let _e1143 = mu_v2MeanLitter;
            mu_v2MeanLitter.occlusion = (_e1143.occlusion * 0.72f);
            let _e1147 = baseClosed;
            let _e1148 = canopyPhase;
            let _e1149 = mu_v2Canopy(_e1148);
            let _e1151 = leafOn;
            let _e1152 = mu_v2Canopy(_e1151);
            bareGuide = ((_e1147 * _e1149) / max(_e1152, vec3(0.02f)));
            let _e1158 = st_27;
            let _e1159 = bareGuide;
            let _e1160 = st_27;
            let _e1165 = mu_forestCrownColor(_e1158, _e1159, _e1160.needle, vec2(0f), 0f);
            crownBare = _e1165;
            let _e1167 = crownBare;
            let _e1170 = crownBare;
            let _e1171 = st_27;
            let _e1173 = light_16;
            let _e1174 = light2_5;
            let _e1175 = strength_5;
            let _e1176 = mu_v2CanopyMeanRatio(_e1170, _e1171.needle, _e1173, _e1174, _e1175);
            let _e1178 = e_28;
            let _e1181 = st_27;
            crownMean = (_e1167 * mix(vec3(1f), _e1176, vec3((1f - (_e1178.snow * (1f - _e1181.needle))))));
            let _e1191 = crownMean;
            let _e1192 = e_28;
            let _e1196 = st_27;
            let _e1200 = mu_forestSnowed(_e1191, (_e1192.snow * mix(0.62f, 0.22f, _e1196.needle)));
            let _e1201 = mu_groundToLinear(_e1200);
            mu_v2MeanCrown.albedo = _e1201;
            let _e1203 = mu_v2MacroNormal;
            mu_v2MeanCrown.normal = _e1203;
            mu_v2MeanCrown.roughness = 0.9f;
            mu_v2MeanCrown.occlusion = 1f;
            mu_v2MeanEnabled = 1f;
        }
    }
    let _e1209 = m_13;
    let _e1211 = mu_groundToDisplay(_e1209.albedo);
    mu_renderGuideColor = _e1211;
    let _e1212 = opened;
    mu_renderGuideOpen = _e1212;
    let _e1213 = closed;
    mu_renderGuideClosed = _e1213;
    let _e1214 = woods_2;
    mu_renderGuideWoods = _e1214;
    let _e1215 = m_13;
    return _e1215;
}

fn mu_v2ShadeGround(resolved_2: mu_GroundMaterial, view_4: vec3<f32>, light_17: vec3<f32>, light2_6: vec3<f32>, strength_6: f32) -> vec3<f32> {
    var resolved_3: mu_GroundMaterial;
    var view_5: vec3<f32>;
    var light_18: vec3<f32>;
    var light2_7: vec3<f32>;
    var strength_7: f32;
    var nearColor: vec3<f32>;
    var openResponse: vec3<f32>;
    var litterResponse: vec3<f32>;
    var crownResponse: vec3<f32>;
    var meanResponse: vec3<f32>;

    resolved_3 = resolved_2;
    view_5 = view_4;
    light_18 = light_17;
    light2_7 = light2_6;
    strength_7 = strength_6;
    let _e335 = resolved_3;
    let _e336 = view_5;
    let _e337 = light_18;
    let _e338 = light2_7;
    let _e339 = strength_7;
    let _e340 = mu_shadeTerrainGround(_e335, _e336, _e337, _e338, _e339);
    nearColor = _e340;
    let _e342 = mu_v2MeanEnabled;
    if (_e342 < 0.5f) {
        let _e345 = nearColor;
        return _e345;
    }
    let _e346 = mu_v2MeanOpen;
    let _e347 = view_5;
    let _e348 = light_18;
    let _e349 = light2_7;
    let _e350 = strength_7;
    let _e351 = mu_shadeTerrainGround(_e346, _e347, _e348, _e349, _e350);
    let _e352 = mu_groundToLinear(_e351);
    openResponse = _e352;
    let _e354 = mu_v2MeanLitter;
    let _e355 = view_5;
    let _e356 = light_18;
    let _e357 = light2_7;
    let _e358 = strength_7;
    let _e359 = mu_shadeTerrainGround(_e354, _e355, _e356, _e357, _e358);
    let _e360 = mu_groundToLinear(_e359);
    litterResponse = _e360;
    let _e362 = mu_v2MeanCrown;
    let _e363 = view_5;
    let _e364 = light_18;
    let _e365 = light2_7;
    let _e366 = strength_7;
    let _e367 = mu_shadeTerrainGround(_e362, _e363, _e364, _e365, _e366);
    let _e368 = mu_groundToLinear(_e367);
    crownResponse = _e368;
    let _e370 = openResponse;
    let _e371 = mu_v2MeanWeights;
    let _e374 = litterResponse;
    let _e375 = mu_v2MeanWeights;
    let _e379 = crownResponse;
    let _e380 = mu_v2MeanWeights;
    meanResponse = (((_e370 * _e371.x) + (_e374 * _e375.y)) + (_e379 * _e380.z));
    let _e385 = meanResponse;
    let _e386 = nearColor;
    let _e387 = mu_groundToLinear(_e386);
    let _e388 = mu_v2ForestReveal;
    let _e391 = mu_groundToDisplay(mix(_e385, _e387, vec3(_e388)));
    return _e391;
}
fn mu_surfaceMicro(p:vec2<f32>,s:f32,fp:f32,r:f32,g:f32,seed:u32)->vec3<f32>{return magnaMicro(p,s,fp,r,g,seed);}
