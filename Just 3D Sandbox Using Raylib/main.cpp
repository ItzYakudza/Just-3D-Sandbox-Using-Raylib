#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <array>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <shared_mutex>
#include <chrono>
#include <filesystem>
#include <climits>

#include "raylib.h"
#include "rlImGui.h"
#include "raymath.h"
#include "imgui.h"
#include "rlgl.h"
#include "nlohmann/json.hpp"
#include "FastNoiseLite.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOUSER
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#define CHUNK_SIZE 16
#define WORLD_HEIGHT 256
#define REGION_SIZE 4
#define SEA_LEVEL 36

static constexpr float kPlayerWidth = 0.60f;
static constexpr float kPlayerHalfW = 0.30f;
static constexpr float kBodyHeightStand = 1.80f;
static constexpr float kBodyHeightCrouch = 1.25f;
static constexpr float kEyeHeightStand = 1.62f;
static constexpr float kEyeHeightCrouch = 1.12f;

int g_renderDistance = 8;

Shader g_worldShader;
int g_dayFactorLoc = -1;
float g_dayFactor = 1.0f;
Vector3 g_sunDirection = { -0.4f, -1.0f, -0.2f };

std::shared_mutex g_chunksMutex;

const char* kWorldVS = R"(
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;

out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
out vec3 fragWorldPos;

void main() {
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = vertexNormal;
    fragWorldPos = vertexPosition;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)";

const char* kWorldFS = R"(
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
in vec3 fragWorldPos;

uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec3 sunDirection;
uniform float dayFactor;

out vec4 finalColor;

float lightCurve(float l) {
    l = clamp(l, 0.0, 1.0);
    return l * (2.0 - l);
}

void main() {
    vec4 texel = texture(texture0, fragTexCoord) * colDiffuse;
    if (texel.a < 0.5) discard;

    float shade = max(fragColor.r, 0.40);
    float skyL  = clamp(fragColor.g, 0.0, 1.0);
    float blkL  = clamp(fragColor.b, 0.0, 1.0);

    float skyB = lightCurve(skyL);
    float blkB = lightCurve(blkL);

    vec3 nightTint = vec3(0.58, 0.66, 0.90);
    vec3 dayTint   = vec3(1.00, 0.99, 0.95);
    vec3 skyTint   = mix(nightTint, dayTint, dayFactor);
    float skyAmp   = mix(0.28, 1.00, dayFactor);
    vec3 skyLight  = skyTint * skyAmp * skyB;

    vec3 torchCol   = vec3(1.00, 0.70, 0.35);
    vec3 torchLight = torchCol * blkB * mix(1.20, 0.80, dayFactor);

    vec3 mixed = max(skyLight, torchLight);

    const float ambient = 0.20;
    vec3 lit = texel.rgb * shade * (mixed + ambient);
    finalColor = vec4(lit, texel.a);
}
)";

using BlockId = uint16_t;
constexpr BlockId INVALID_BLOCK_ID = 0xFFFF;

class BlockRegistry;
extern BlockRegistry g_blocks;
extern BlockId g_AirId, g_WaterId, g_LavaId, g_BedrockId;

Texture2D g_waterTexture{};
Texture2D g_lavaTexture{};

int64_t ChunkKey(int cx, int cz) {
	return ((int64_t)(uint32_t)cx << 32) | (uint32_t)cz;
}

enum class ChunkState {
	NotLoaded,
	Loading,
	Loaded,
	Unloaded
};

struct Plane {
	Vector3 normal;
	float d;
};

struct Frustum {
	Plane planes[6];
};

struct LocalAABB {
	Vector3 min;
	Vector3 max;
};

struct VoxelShape {
	std::vector<LocalAABB> boxes;
	bool isFullCube = false;
};

static LocalAABB BoxPx(float x0, float y0, float z0, float x1, float y1, float z1) {
	return { { x0 / 16.0f, y0 / 16.0f, z0 / 16.0f }, { x1 / 16.0f, y1 / 16.0f, z1 / 16.0f } };
}

bool PointInLocalAABB(Vector3 p, const LocalAABB& b, float eps) {
	return p.x >= b.min.x - eps && p.x <= b.max.x + eps &&
		p.y >= b.min.y - eps && p.y <= b.max.y + eps &&
		p.z >= b.min.z - eps && p.z <= b.max.z + eps;
}

bool RayIntersectsAABB(Vector3 orig, Vector3 dir, Vector3 bmin, Vector3 bmax,
	float& outT, Vector3& outNormal)
{
	const float EPS = 1e-8f;
	float tmin = 0.0f;
	float tmax = FLT_MAX;
	Vector3 hitN{ 0, 1, 0 };

	for (int i = 0; i < 3; i++) {
		float o = (&orig.x)[i];
		float d = (&dir.x)[i];
		float mn = (&bmin.x)[i];
		float mx = (&bmax.x)[i];
		if (fabsf(d) < EPS) {
			if (o < mn || o > mx) return false;
			continue;
		}
		float inv = 1.0f / d;
		float t0 = (mn - o) * inv;
		float t1 = (mx - o) * inv;
		float n0 = -1.0f;
		if (t0 > t1) { std::swap(t0, t1); n0 = 1.0f; }
		if (t0 > tmin) {
			tmin = t0;
			hitN = { 0, 0, 0 };
			(&hitN.x)[i] = n0;
		}
		if (t1 < tmax) tmax = t1;
		if (tmax < tmin) return false;
	}
	if (tmin < 1e-5f) return false;
	outT = tmin;
	outNormal = hitN;
	return true;
}

bool RaycastVoxelShape(const VoxelShape& shape, Vector3 blockOrigin,
	Vector3 rayOrig, Vector3 rayDir, float maxDist, float& outT, Vector3& outN)
{
	bool any = false;
	float best = maxDist;
	Vector3 bestN{};
	for (const auto& b : shape.boxes) {
		Vector3 mn = Vector3Add(blockOrigin, b.min);
		Vector3 mx = Vector3Add(blockOrigin, b.max);
		float t; Vector3 n;
		if (RayIntersectsAABB(rayOrig, rayDir, mn, mx, t, n) && t <= best) {
			best = t; bestN = n; any = true;
		}
	}
	if (any) { outT = best; outN = bestN; }
	return any;
}

void ExtractFrustum(Frustum& frustum, const Matrix& clip) {
	frustum.planes[0].normal = { clip.m3 + clip.m0, clip.m7 + clip.m4, clip.m11 + clip.m8 };
	frustum.planes[0].d = clip.m15 + clip.m12;
	frustum.planes[1].normal = { clip.m3 - clip.m0, clip.m7 - clip.m4, clip.m11 - clip.m8 };
	frustum.planes[1].d = clip.m15 - clip.m12;
	frustum.planes[2].normal = { clip.m3 + clip.m1, clip.m7 + clip.m5, clip.m11 + clip.m9 };
	frustum.planes[2].d = clip.m15 + clip.m13;
	frustum.planes[3].normal = { clip.m3 - clip.m1, clip.m7 - clip.m5, clip.m11 - clip.m9 };
	frustum.planes[3].d = clip.m15 - clip.m13;
	frustum.planes[4].normal = { clip.m3 + clip.m2, clip.m7 + clip.m6, clip.m11 + clip.m10 };
	frustum.planes[4].d = clip.m15 + clip.m14;
	frustum.planes[5].normal = { clip.m3 - clip.m2, clip.m7 - clip.m6, clip.m11 - clip.m10 };
	frustum.planes[5].d = clip.m15 - clip.m14;

	for (int i = 0; i < 6; i++) {
		float len = Vector3Length(frustum.planes[i].normal);
		if (len > 0.0001f) {
			frustum.planes[i].normal = Vector3Scale(frustum.planes[i].normal, 1.0f / len);
			frustum.planes[i].d /= len;
		}
	}
}

bool IsAABBInFrustum(const Frustum& f, Vector3 min, Vector3 max) {
	for (int i = 0; i < 6; i++) {
		const Plane& p = f.planes[i];
		Vector3 positive = {
			(p.normal.x >= 0) ? max.x : min.x,
			(p.normal.y >= 0) ? max.y : min.y,
			(p.normal.z >= 0) ? max.z : min.z
		};
		if (p.normal.x * positive.x + p.normal.y * positive.y + p.normal.z * positive.z + p.d < 0)
			return false;
	}
	return true;
}

void GetChunkAABB(int chunkX, int chunkZ, Vector3& outMin, Vector3& outMax) {
	outMin = { (float)(chunkX * CHUNK_SIZE), 0.0f, (float)(chunkZ * CHUNK_SIZE) };
	outMax = { outMin.x + CHUNK_SIZE, (float)WORLD_HEIGHT, outMin.z + CHUNK_SIZE };
}

int64_t RegionKey(int rx, int rz) {
	return ((int64_t)(uint32_t)rx << 32) | (uint32_t)rz;
}

void GetRegionAABB(int rx, int rz, Vector3& outMin, Vector3& outMax) {
	outMin = { (float)(rx * REGION_SIZE * CHUNK_SIZE), 0.0f, (float)(rz * REGION_SIZE * CHUNK_SIZE) };
	outMax = { outMin.x + REGION_SIZE * CHUNK_SIZE, (float)WORLD_HEIGHT, outMin.z + REGION_SIZE * CHUNK_SIZE };
}

uint8_t GetBlockLightEmission(BlockId id);

struct Chunk {
	int chunkX = 0;
	int chunkZ = 0;
	Model model{};
	Model waterModel{};
	bool dirtyGeometry = true;
	bool dirtyLight = true;
	bool hasMesh = false;
	bool hasWaterMesh = false;
	std::vector<Vector3> lightSourcePositions;

	bool NeedsRebuild() const { return dirtyGeometry || dirtyLight; }
	void MarkClean() { dirtyGeometry = false; dirtyLight = false; }
	void MarkGeometryDirty() { dirtyGeometry = true; dirtyLight = true; }
	void MarkLightDirty() { dirtyLight = true; }

	uint8_t* skyLight = nullptr;
	uint8_t* blockLight = nullptr;

	enum class Mode : uint8_t { Single, Layered, Dense };
	Mode mode = Mode::Single;
	BlockId singleBlock = 0;

	struct Layer {
		bool uniform = true;
		BlockId id = 0;
		BlockId* data = nullptr;
	};
	Layer* layers = nullptr;
	BlockId* dense = nullptr;
	std::unordered_map<int, uint8_t> fluidLevels;

	static int LightIndex(int x, int y, int z) { return (x * WORLD_HEIGHT + y) * CHUNK_SIZE + z; }

	void EnsureLightAllocated() {
		const int N = CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE;
		if (!skyLight) { skyLight = new uint8_t[N]; memset(skyLight, 0, N); }
		if (!blockLight) { blockLight = new uint8_t[N]; memset(blockLight, 0, N); }
	}
	void ClearLight() { delete[] skyLight; skyLight = nullptr; delete[] blockLight; blockLight = nullptr; }

	uint8_t GetSkyLight(int x, int y, int z) const {
		if (!skyLight || x < 0 || x >= CHUNK_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= CHUNK_SIZE) return 0;
		return skyLight[LightIndex(x, y, z)];
	}
	uint8_t GetBlockLight(int x, int y, int z) const {
		if (!blockLight || x < 0 || x >= CHUNK_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= CHUNK_SIZE) return 0;
		return blockLight[LightIndex(x, y, z)];
	}

	BlockId Get(int x, int y, int z) const {
		if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= CHUNK_SIZE) return g_AirId;
		switch (mode) {
		case Mode::Single: return singleBlock;
		case Mode::Layered: {
			if (!layers) return g_AirId;
			const Layer& L = layers[y];
			if (L.uniform) return L.id;
			if (!L.data) return g_AirId;
			return L.data[x * CHUNK_SIZE + z];
		}
		case Mode::Dense:
			if (!dense) return g_AirId;
			return dense[(x * WORLD_HEIGHT + y) * CHUNK_SIZE + z];
		}
		return g_AirId;
	}

	void Set(int x, int y, int z, BlockId type) {
		if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= CHUNK_SIZE) return;
		BlockId old = Get(x, y, z);

		if (mode != Mode::Dense) {
			BlockId* newData = new BlockId[CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE];
			for (int xx = 0; xx < CHUNK_SIZE; xx++)
				for (int yy = 0; yy < WORLD_HEIGHT; yy++)
					for (int zz = 0; zz < CHUNK_SIZE; zz++)
						newData[(xx * WORLD_HEIGHT + yy) * CHUNK_SIZE + zz] = Get(xx, yy, zz);
			ClearStorage();
			dense = newData;
			mode = Mode::Dense;
		}
		dense[(x * WORLD_HEIGHT + y) * CHUNK_SIZE + z] = type;

		if (old != type) {
			Vector3 wp{
				(float)(chunkX * CHUNK_SIZE + x) + 0.5f,
				(float)y + 0.5f,
				(float)(chunkZ * CHUNK_SIZE + z) + 0.5f
			};
			uint8_t oldLight = GetBlockLightEmission(old);
			uint8_t newLight = GetBlockLightEmission(type);
			if (oldLight > 0) {
				lightSourcePositions.erase(std::remove_if(lightSourcePositions.begin(), lightSourcePositions.end(),
					[&](const Vector3& p) {
						return fabsf(p.x - wp.x) < 0.01f && fabsf(p.y - wp.y) < 0.01f && fabsf(p.z - wp.z) < 0.01f;
					}), lightSourcePositions.end());
			}
			if (newLight > 0) lightSourcePositions.push_back(wp);
		}
	}

	void Optimize() {
		if (mode != Mode::Dense) return;
		const int N = CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE;
		BlockId first = dense[0];
		bool allSame = true;
		for (int i = 1; i < N; i++) if (dense[i] != first) { allSame = false; break; }
		if (allSame) { ClearStorage(); mode = Mode::Single; singleBlock = first; return; }

		Layer* newLayers = new Layer[WORLD_HEIGHT];
		for (int y = 0; y < WORLD_HEIGHT; y++) {
			BlockId firstInLayer = dense[(0 * WORLD_HEIGHT + y) * CHUNK_SIZE + 0];
			bool uniform = true;
			for (int x = 0; x < CHUNK_SIZE && uniform; x++)
				for (int z = 0; z < CHUNK_SIZE; z++)
					if (dense[(x * WORLD_HEIGHT + y) * CHUNK_SIZE + z] != firstInLayer) { uniform = false; break; }
			newLayers[y].uniform = uniform;
			if (uniform) { newLayers[y].id = firstInLayer; newLayers[y].data = nullptr; }
			else {
				newLayers[y].data = new BlockId[CHUNK_SIZE * CHUNK_SIZE];
				for (int x = 0; x < CHUNK_SIZE; x++)
					for (int z = 0; z < CHUNK_SIZE; z++)
						newLayers[y].data[x * CHUNK_SIZE + z] = dense[(x * WORLD_HEIGHT + y) * CHUNK_SIZE + z];
			}
		}
		ClearStorage();
		layers = newLayers;
		mode = Mode::Layered;
	}

	size_t GetStorageBytes() const {
		switch (mode) {
		case Mode::Single: return sizeof(BlockId);
		case Mode::Layered: {
			size_t total = WORLD_HEIGHT * sizeof(Layer);
			for (int y = 0; y < WORLD_HEIGHT; y++)
				if (!layers[y].uniform && layers[y].data) total += (size_t)CHUNK_SIZE * CHUNK_SIZE * sizeof(BlockId);
			return total;
		}
		case Mode::Dense: return (size_t)CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE * sizeof(BlockId);
		}
		return 0;
	}

	void ClearStorage() {
		if (layers) { for (int i = 0; i < WORLD_HEIGHT; i++) delete[] layers[i].data; delete[] layers; layers = nullptr; }
		delete[] dense; dense = nullptr;
		ClearLight();
	}

	Chunk() = default;
	~Chunk() { ClearStorage(); if (hasMesh) UnloadModel(model); if (hasWaterMesh) UnloadModel(waterModel); }
	Chunk(const Chunk&) = delete;
	Chunk& operator=(const Chunk&) = delete;

	Chunk(Chunk&& other) noexcept {
		chunkX = other.chunkX; chunkZ = other.chunkZ;
		model = other.model; waterModel = other.waterModel;
		dirtyGeometry = other.dirtyGeometry; dirtyLight = other.dirtyLight;
		hasMesh = other.hasMesh; hasWaterMesh = other.hasWaterMesh;
		lightSourcePositions = std::move(other.lightSourcePositions);
		fluidLevels = std::move(other.fluidLevels);
		skyLight = other.skyLight; blockLight = other.blockLight;
		mode = other.mode; singleBlock = other.singleBlock;
		layers = other.layers; dense = other.dense;
		other.layers = nullptr; other.dense = nullptr;
		other.skyLight = nullptr; other.blockLight = nullptr;
		other.hasMesh = false; other.hasWaterMesh = false;
	}
	Chunk& operator=(Chunk&& other) noexcept {
		if (this != &other) {
			ClearStorage();
			if (hasMesh) UnloadModel(model);
			if (hasWaterMesh) UnloadModel(waterModel);
			chunkX = other.chunkX; chunkZ = other.chunkZ;
			model = other.model; waterModel = other.waterModel;
			dirtyGeometry = other.dirtyGeometry; dirtyLight = other.dirtyLight;
			hasMesh = other.hasMesh; hasWaterMesh = other.hasWaterMesh;
			lightSourcePositions = std::move(other.lightSourcePositions);
			fluidLevels = std::move(other.fluidLevels);
			skyLight = other.skyLight; blockLight = other.blockLight;
			mode = other.mode; singleBlock = other.singleBlock;
			layers = other.layers; dense = other.dense;
			other.layers = nullptr; other.dense = nullptr;
			other.skyLight = nullptr; other.blockLight = nullptr;
			other.hasMesh = false; other.hasWaterMesh = false;
		}
		return *this;
	}
};

struct MeshVertex {
	Vector3 pos;
	Vector2 uv;
	Vector3 normal;
	Color color;
};

struct FaceData {
	Texture2D texture;
	std::vector<MeshVertex> vertices;
	std::vector<unsigned short> indices;
};

struct FaceInfo {
	Texture2D texture{};
	bool tinted = false;
};

BlockId GetWorldBlock(const std::unordered_map<int64_t, Chunk>& chunks, int chunkX, int chunkZ,
	int localX, int localY, int localZ)
{
	if (localX < 0) { chunkX -= 1; localX = CHUNK_SIZE - 1; }
	if (localX >= CHUNK_SIZE) { chunkX += 1; localX = 0; }
	if (localZ < 0) { chunkZ -= 1; localZ = CHUNK_SIZE - 1; }
	if (localZ >= CHUNK_SIZE) { chunkZ += 1; localZ = 0; }
	if (localY < 0 || localY >= WORLD_HEIGHT) return g_AirId;

	auto it = chunks.find(ChunkKey(chunkX, chunkZ));
	if (it == chunks.end()) return g_AirId;

	return it->second.Get(localX, localY, localZ);
}

struct BlockHit {
	bool hit = false;
	int x = 0, y = 0, z = 0;
	int nx = 0, ny = 0, nz = 0;
	BlockId type = INVALID_BLOCK_ID;
};

Texture2D GetOrLoadTexture(std::map<std::string, Texture2D>& cache, const std::string& textureId) {
	auto it = cache.find(textureId);
	if (it != cache.end()) return it->second;

	std::string id = textureId;
	const std::string prefix = "minecraft:";
	if (id.substr(0, prefix.size()) == prefix) id = id.substr(prefix.size());

	std::string path = "resources/textures/" + id + ".png";
	Texture2D tex = LoadTexture(path.c_str());
	SetTextureFilter(tex, TEXTURE_FILTER_POINT);
	SetTextureWrap(tex, TEXTURE_WRAP_CLAMP);

	cache[textureId] = tex;
	return tex;
}

nlohmann::json LoadRawJson(const std::string& path) {
	std::ifstream file(path);
	nlohmann::json data;
	file >> data;
	return data;
}

std::string ResolveModelPath(const std::string& minecraftId) {
	std::string name = minecraftId;
	const std::string prefix = "minecraft:";
	if (name.substr(0, prefix.size()) == prefix) {
		name = name.substr(prefix.size());
	}
	return "resources/models/" + name + ".json";
}

std::map<std::string, std::string> ResolveTextures(const std::string& modelPath) {
	nlohmann::json model = LoadRawJson(modelPath);
	std::map<std::string, std::string> result;

	if (model.contains("parent")) {
		std::string parentName = model["parent"];
		std::string parentPath = ResolveModelPath(parentName);
		result = ResolveTextures(parentPath);
	}

	if (model.contains("textures")) {
		for (auto& [key, value] : model["textures"].items()) {
			result[key] = value.get<std::string>();
		}
	}

	return result;
}

nlohmann::json FindElement(const std::string& modelPath) {
	nlohmann::json model = LoadRawJson(modelPath);
	if (model.contains("elements")) {
		return model["elements"][0];
	}
	if (model.contains("parent")) {
		return FindElement(ResolveModelPath(model["parent"]));
	}
	return nlohmann::json();
}

std::string ResolveTextureReference(const std::string& ref, const std::map<std::string, std::string>& textures) {
	std::string current = ref;
	while (!current.empty() && current[0] == '#') {
		std::string key = current.substr(1);
		auto it = textures.find(key);
		if (it == textures.end()) break;
		current = it->second;
	}
	return current;
}

struct LocalQuad {
	Vector3 v[4]{};
	Vector3 normal{};
	Vector2 uv[4] = { {0,1},{0,0},{1,0},{1,1} };
	Texture2D texture{};
	bool hasOwnTexture = false;
	bool tinted = false;
};

struct LocalTri {
	Vector3 a, b, c;
	Vector3 normal;
};

struct BlockModel {
	std::vector<LocalQuad> quads;
	std::vector<LocalTri> triangles;
	std::vector<std::pair<Vector3, Vector3>> outlineEdges;
	bool isFullCube = false;
};

void AddBoxQuads(std::vector<LocalQuad>& out, Vector3 lo, Vector3 hi) {
	Vector3 p000{ lo.x,lo.y,lo.z }, p100{ hi.x,lo.y,lo.z };
	Vector3 p010{ lo.x,hi.y,lo.z }, p110{ hi.x,hi.y,lo.z };
	Vector3 p001{ lo.x,lo.y,hi.z }, p101{ hi.x,lo.y,hi.z };
	Vector3 p011{ lo.x,hi.y,hi.z }, p111{ hi.x,hi.y,hi.z };
	out.push_back({ {p000,p010,p110,p100}, {0,0,-1} });
	out.push_back({ {p001,p101,p111,p011}, {0,0, 1} });
	out.push_back({ {p000,p001,p011,p010}, {-1,0,0} });
	out.push_back({ {p100,p110,p111,p101}, {1,0,0} });
	out.push_back({ {p010,p011,p111,p110}, {0,1,0} });
	out.push_back({ {p000,p100,p101,p001}, {0,-1,0} });
}

void AddCrossQuads(std::vector<LocalQuad>& out) {
	Vector3 n{ 0,1,0 }, nn{ 0,-1,0 };
	auto plane = [&](Vector3 bl, Vector3 br, Vector3 tr, Vector3 tl) {
		out.push_back({ {bl,tl,tr,br}, n });
		out.push_back({ {br,tr,tl,bl}, nn });
		};
	plane({ 0.24f,0.0f,0.24f }, { 0.76f,0.0f,0.76f }, { 0.76f,0.78f,0.76f }, { 0.24f,0.78f,0.24f });
	plane({ 0.76f,0.0f,0.24f }, { 0.24f,0.0f,0.76f }, { 0.24f,0.78f,0.76f }, { 0.76f,0.78f,0.24f });
}

struct JsonFaceDef {
	bool present = false;
	std::string textureRef;
	float uv[4] = { 0,0,16,16 };
	int rotation = 0;
	bool hasTint = false;
};

struct JsonElementDef {
	Vector3 from{ 0,0,0 }, to{ 16,16,16 };
	JsonFaceDef down, up, north, south, west, east;
};

std::vector<JsonElementDef> LoadModelElementsRecursive(const std::string& modelPath) {
	nlohmann::json model = LoadRawJson(modelPath);

	if (model.contains("elements")) {
		std::vector<JsonElementDef> result;
		for (auto& elJson : model["elements"]) {
			JsonElementDef e;
			auto fromArr = elJson["from"];
			auto toArr = elJson["to"];
			e.from = { fromArr[0].get<float>(), fromArr[1].get<float>(), fromArr[2].get<float>() };
			e.to = { toArr[0].get<float>(),   toArr[1].get<float>(),   toArr[2].get<float>() };

			if (elJson.contains("faces")) {
				auto& facesJson = elJson["faces"];
				auto parseFace = [&](const char* name, JsonFaceDef& out) {
					if (!facesJson.contains(name)) return;
					out.present = true;
					auto& fj = facesJson[name];
					out.textureRef = fj.value("texture", std::string());
					if (fj.contains("uv")) {
						auto uvArr = fj["uv"];
						out.uv[0] = uvArr[0].get<float>(); out.uv[1] = uvArr[1].get<float>();
						out.uv[2] = uvArr[2].get<float>(); out.uv[3] = uvArr[3].get<float>();
					}
					out.rotation = fj.value("rotation", 0);
					out.hasTint = fj.contains("tintindex");
					};
				parseFace("down", e.down);  parseFace("up", e.up);
				parseFace("north", e.north); parseFace("south", e.south);
				parseFace("west", e.west);   parseFace("east", e.east);
			}
			result.push_back(e);
		}
		return result;
	}
	if (model.contains("parent")) {
		return LoadModelElementsRecursive(ResolveModelPath(model["parent"].get<std::string>()));
	}
	return {};
}

struct ElementFaceGeom {
	Vector3 v[4];
	Vector3 normal;
	Vector2 uvUnit[4];
};

bool GetElementFaceGeom(const std::string& faceName, Vector3 lo, Vector3 hi, ElementFaceGeom& out) {
	Vector3 p000{ lo.x,lo.y,lo.z }, p100{ hi.x,lo.y,lo.z };
	Vector3 p010{ lo.x,hi.y,lo.z }, p110{ hi.x,hi.y,lo.z };
	Vector3 p001{ lo.x,lo.y,hi.z }, p101{ hi.x,lo.y,hi.z };
	Vector3 p011{ lo.x,hi.y,hi.z }, p111{ hi.x,hi.y,hi.z };
	if (faceName == "north") { out.v[0] = p000; out.v[1] = p010; out.v[2] = p110; out.v[3] = p100; out.normal = { 0,0,-1 }; out.uvUnit[0] = { 0,1 }; out.uvUnit[1] = { 0,0 }; out.uvUnit[2] = { 1,0 }; out.uvUnit[3] = { 1,1 }; return true; }
	if (faceName == "south") { out.v[0] = p001; out.v[1] = p101; out.v[2] = p111; out.v[3] = p011; out.normal = { 0,0,1 };  out.uvUnit[0] = { 0,1 }; out.uvUnit[1] = { 1,1 }; out.uvUnit[2] = { 1,0 }; out.uvUnit[3] = { 0,0 }; return true; }
	if (faceName == "west") { out.v[0] = p000; out.v[1] = p001; out.v[2] = p011; out.v[3] = p010; out.normal = { -1,0,0 }; out.uvUnit[0] = { 0,1 }; out.uvUnit[1] = { 1,1 }; out.uvUnit[2] = { 1,0 }; out.uvUnit[3] = { 0,0 }; return true; }
	if (faceName == "east") { out.v[0] = p100; out.v[1] = p110; out.v[2] = p111; out.v[3] = p101; out.normal = { 1,0,0 };  out.uvUnit[0] = { 0,1 }; out.uvUnit[1] = { 0,0 }; out.uvUnit[2] = { 1,0 }; out.uvUnit[3] = { 1,1 }; return true; }
	if (faceName == "up") { out.v[0] = p010; out.v[1] = p011; out.v[2] = p111; out.v[3] = p110; out.normal = { 0,1,0 };  out.uvUnit[0] = { 0,0 }; out.uvUnit[1] = { 0,1 }; out.uvUnit[2] = { 1,1 }; out.uvUnit[3] = { 1,0 }; return true; }
	if (faceName == "down") { out.v[0] = p000; out.v[1] = p100; out.v[2] = p101; out.v[3] = p001; out.normal = { 0,-1,0 }; out.uvUnit[0] = { 0,0 }; out.uvUnit[1] = { 1,0 }; out.uvUnit[2] = { 1,1 }; out.uvUnit[3] = { 0,1 }; return true; }
	return false;
}

void AddElementFaceIfPresent(std::vector<LocalQuad>& quads, const char* name, const JsonFaceDef& fd,
	Vector3 lo, Vector3 hi, const std::map<std::string, std::string>& textures,
	std::map<std::string, Texture2D>& textureCache)
{
	if (!fd.present) return;
	ElementFaceGeom geom;
	if (!GetElementFaceGeom(name, lo, hi, geom)) return;

	float u0 = fd.uv[0] / 16.0f, v0 = fd.uv[1] / 16.0f, u1 = fd.uv[2] / 16.0f, v1 = fd.uv[3] / 16.0f;
	Vector2 corners[4];
	for (int i = 0; i < 4; i++) {
		corners[i].x = (geom.uvUnit[i].x < 0.5f) ? u0 : u1;
		corners[i].y = (geom.uvUnit[i].y < 0.5f) ? v0 : v1;
	}
	int k = ((fd.rotation / 90) % 4 + 4) % 4;
	Vector2 rotated[4];
	for (int i = 0; i < 4; i++) rotated[i] = corners[(i + k) % 4];

	std::string texId = ResolveTextureReference(fd.textureRef, textures);
	Texture2D tex = GetOrLoadTexture(textureCache, texId);

	LocalQuad q;
	for (int i = 0; i < 4; i++) { q.v[i] = geom.v[i]; q.uv[i] = rotated[i]; }
	q.normal = geom.normal;
	q.texture = tex;
	q.hasOwnTexture = true;
	q.tinted = fd.hasTint;
	quads.push_back(q);
}

static int64_t QuantizeCoord(float f) { return (int64_t)llroundf(f * 1000.0f); }
static uint64_t VertKey(const Vector3& v) {
	int64_t x = QuantizeCoord(v.x), y = QuantizeCoord(v.y), z = QuantizeCoord(v.z);
	uint64_t h = 1469598103934665603ULL;
	for (int64_t c : { x, y, z }) { h ^= (uint64_t)c; h *= 1099511628211ULL; }
	return h;
}

// ============================================================================
//  BLOCK REGISTRY — data-driven система блоков.
//  Грузит ВСЕ блоки из resources/blockstates/*.json (variants + multipart,
//  включая OR-условия), резолвит модели/текстуры/коллизии генерически.
//  Заменяет собой жёсткий enum BlockType — следующим шагом Chunk/рендер/
//  плейсмент будут переведены на BlockId.
// ============================================================================
namespace fs = std::filesystem;

enum class SupportType : uint8_t {
	None,     // не требует опоры (булыжник, доски, лестницы, брёвна...)
	Floor,    // требует твёрдый блок СНИЗУ (трава, цветы, факел на полу, ковёр, рельсы, саженцы...)
	WallSide, // требует твёрдый блок со стороны facing (факел/знак/баннер на стене, лестницы, лианы)
	Ceiling   // зарезервировано на будущее (растяжки/шулкеры вверх ногами и т.п.)
};

struct BlockPropertyDef {
	std::string name;
	std::vector<std::string> values;
};

// Ссылка на модель в одном конкретном apply/variant блока (model + поворот)
struct StateModelRef {
	std::string modelId; // "minecraft:block/oak_stairs"
	int rotX = 0, rotY = 0;
	bool uvlock = false;
};

struct MultipartRule {
	bool always = true;   // when отсутствует → применяется всегда
	bool isOr = false;    // when: { "OR": [...] }
	std::vector<std::map<std::string, std::string>> conditions;
	StateModelRef apply;
};

struct BlockStateData {
	bool isMultipart = false;
	std::vector<std::pair<std::map<std::string, std::string>, StateModelRef>> variants;
	std::vector<MultipartRule> multipart;
};

struct BlockDef {
	BlockId id = INVALID_BLOCK_ID;
	std::string name;
	std::vector<BlockPropertyDef> properties;
	std::map<std::string, std::string> defaultState;
	BlockStateData stateData;

	SupportType support = SupportType::None;
	FaceInfo cubeFaces[6];   // 0=down,1=up,2=north,3=south,4=west,5=east — для быстрого пути полных кубов
	uint8_t lightEmission = 0; // 0..15, см. KnownLightEmission()

	bool isFullCube = false;
	bool isOpaque = false;   // полный непрозрачный куб → соседние грани культятся
	bool isCross = false;    // крестовая модель (трава/цветы/саженцы)

	// геометрия/коллизия кэшируются по сериализованному состоянию ("" = дефолт)
	std::unordered_map<std::string, BlockModel> modelCache;
	std::unordered_map<std::string, VoxelShape> shapeCache;
};

static Vector3 RotateAroundY90(Vector3 p, int steps, Vector3 center) {
	p = Vector3Subtract(p, center);
	steps = ((steps % 4) + 4) % 4;
	for (int i = 0; i < steps; i++) { float x = p.x; p.x = p.z; p.z = -x; }
	return Vector3Add(p, center);
}
static Vector3 RotateAroundX90(Vector3 p, int steps, Vector3 center) {
	p = Vector3Subtract(p, center);
	steps = ((steps % 4) + 4) % 4;
	for (int i = 0; i < steps; i++) { float y = p.y; p.y = -p.z; p.z = y; }
	return Vector3Add(p, center);
}
// NB: знак поворота не проверен в движке (нет возможности прособрать проект здесь).
// Если после вставки лестницы/брёвна повёрнуты зеркально — поменять знак в обоих хелперах.

std::map<std::string, std::string> JsonCondToMap(const nlohmann::json& j)
{
	std::map<std::string, std::string> result;

	if (!j.is_object())
		return result;

	for (auto& [key, value] : j.items())
	{
		if (value.is_string())
		{
			result[key] = value.get<std::string>();
		}
		else if (value.is_boolean())
		{
			result[key] = value.get<bool>() ? "true" : "false";
		}
		else if (value.is_number_integer())
		{
			result[key] = std::to_string(value.get<int>());
		}
		else if (value.is_number_float())
		{
			result[key] = value.dump();
		}
		else if (value.is_array())
		{
			std::string combined;

			for (const auto& item : value)
			{
				if (!item.is_string())
					continue;

				if (!combined.empty())
					combined += ",";

				combined += item.get<std::string>();
			}

			result[key] = combined;
		}
		else
		{
			result[key] = value.dump();
		}
	}

	return result;
}

static bool MatchesConditions(const std::map<std::string, std::string>& state,
	const std::map<std::string, std::string>& cond)
{
	for (auto& [k, v] : cond) {
		auto it = state.find(k);
		if (it == state.end()) continue; // свойство не относится к этому блоку — мягкое совпадение
		if (v.find('|') != std::string::npos) { // синтаксис "north|south"
			bool any = false; size_t start = 0;
			while (start <= v.size()) {
				size_t p = v.find('|', start);
				std::string part = v.substr(start, p == std::string::npos ? std::string::npos : p - start);
				if (part == it->second) { any = true; break; }
				if (p == std::string::npos) break;
				start = p + 1;
			}
			if (!any) return false;
		}
		else if (it->second != v) return false;
	}
	return true;
}

static std::string PreferredDefaultFor(const std::string& prop, const std::vector<std::string>& values) {
	auto has = [&](const char* v) { return std::find(values.begin(), values.end(), std::string(v)) != values.end(); };
	if (prop == "half" && has("bottom")) return "bottom";
	if (prop == "facing" && has("north")) return "north";
	if (prop == "axis" && has("y")) return "y";
	if (prop == "shape" && has("straight")) return "straight";
	static const char* kFalseDefaults[] = { "waterlogged","open","powered","lit","triggered","extended",
		"disarmed","attached","hanging","occupied","persistent","snowy","short","drag" };
	for (const char* fp : kFalseDefaults) if (prop == fp && has("false")) return "false";
	return values.empty() ? "" : values.front();
}

static SupportType ClassifySupport(const std::string& name, bool isCross) {
	auto has = [&](const char* s) { return name.find(s) != std::string::npos; };
	if (isCross) return SupportType::Floor;
	if (has("wall_torch") || has("wall_sign") || has("wall_banner") || has("wall_head") ||
		has("wall_skull") || has("wall_hanging_sign") || has("ladder") || has("vine"))
		return SupportType::WallSide;
	if (has("torch") || has("sign") || has("banner") || has("carpet") || has("rail") ||
		has("pressure_plate") || has("sapling") || has("flower_pot") || has("campfire") ||
		has("button") || has("lever"))
		return SupportType::Floor; // грубое приближение для button/lever — уточним в шаге "крепление" через facing
	return SupportType::None;
}

static std::unordered_map<unsigned int, bool> g_textureAlphaCache;
static bool TextureHasAlpha(Texture2D tex) {
	auto it = g_textureAlphaCache.find(tex.id);
	if (it != g_textureAlphaCache.end()) return it->second;
	Image img = LoadImageFromTexture(tex);
	bool has = false;
	Color* px = LoadImageColors(img);
	for (int i = 0; i < img.width * img.height; i++) if (px[i].a < 250) { has = true; break; }
	UnloadImageColors(px);
	UnloadImage(img);
	g_textureAlphaCache[tex.id] = has;
	return has;
}

static bool ModelHasCrossParent(const std::string& modelPath) {
	nlohmann::json model = LoadRawJson(modelPath);
	if (model.contains("parent")) {
		std::string p = model["parent"].get<std::string>();
		if (p.find("cross") != std::string::npos) return true;
		return ModelHasCrossParent(ResolveModelPath(p));
	}
	return false;
}

// Строит геометрию ОДНОГО apply/variant с учётом поворота x/y блокстейта
BlockModel BuildGenericBlockModel(const StateModelRef& ref, std::map<std::string, Texture2D>& textureCache) {
	BlockModel m;
	if (ref.modelId.empty()) return m;
	std::string modelPath = ResolveModelPath(ref.modelId);
	auto textures = ResolveTextures(modelPath);
	auto elements = LoadModelElementsRecursive(modelPath);

	for (auto& el : elements) {
		Vector3 lo{ el.from.x / 16.0f, el.from.y / 16.0f, el.from.z / 16.0f };
		Vector3 hi{ el.to.x / 16.0f,   el.to.y / 16.0f,   el.to.z / 16.0f };
		size_t before = m.quads.size();
		AddElementFaceIfPresent(m.quads, "down", el.down, lo, hi, textures, textureCache);
		AddElementFaceIfPresent(m.quads, "up", el.up, lo, hi, textures, textureCache);
		AddElementFaceIfPresent(m.quads, "north", el.north, lo, hi, textures, textureCache);
		AddElementFaceIfPresent(m.quads, "south", el.south, lo, hi, textures, textureCache);
		AddElementFaceIfPresent(m.quads, "west", el.west, lo, hi, textures, textureCache);
		AddElementFaceIfPresent(m.quads, "east", el.east, lo, hi, textures, textureCache);

		if (ref.rotX != 0 || ref.rotY != 0) {
			const Vector3 center{ 0.5f, 0.5f, 0.5f };
			for (size_t qi = before; qi < m.quads.size(); ++qi) {
				LocalQuad& q = m.quads[qi];
				for (int vi = 0; vi < 4; ++vi) {
					Vector3 p = q.v[vi];
					if (ref.rotX) p = RotateAroundX90(p, ref.rotX / 90, center);
					if (ref.rotY) p = RotateAroundY90(p, ref.rotY / 90, center);
					q.v[vi] = p;
				}
				Vector3 n = q.normal;
				if (ref.rotX) n = RotateAroundX90(n, ref.rotX / 90, { 0,0,0 });
				if (ref.rotY) n = RotateAroundY90(n, ref.rotY / 90, { 0,0,0 });
				q.normal = n;
			}
		}
	}
	for (auto& q : m.quads) {
		m.triangles.push_back({ q.v[0], q.v[1], q.v[2], q.normal });
		m.triangles.push_back({ q.v[0], q.v[2], q.v[3], q.normal });
	}
	m.isFullCube = (elements.size() == 1 &&
		elements[0].from.x <= 0.01f && elements[0].from.y <= 0.01f && elements[0].from.z <= 0.01f &&
		elements[0].to.x >= 15.99f && elements[0].to.y >= 15.99f && elements[0].to.z >= 15.99f &&
		elements[0].down.present && elements[0].up.present && elements[0].north.present &&
		elements[0].south.present && elements[0].west.present && elements[0].east.present &&
		ref.rotX == 0 && ref.rotY == 0);
	return m;
}

// Собирает полную геометрию блока для конкретного состояния (variants ИЛИ multipart)
BlockModel BuildStateModel(const BlockDef& def, const std::map<std::string, std::string>& state,
	std::map<std::string, Texture2D>& textureCache)
{
	BlockModel result;
	if (!def.stateData.isMultipart) {
		const StateModelRef* ref = nullptr;
		for (auto& [cond, r] : def.stateData.variants) if (cond == state) { ref = &r; break; }
		if (!ref && !def.stateData.variants.empty()) ref = &def.stateData.variants.front().second;
		if (ref) result = BuildGenericBlockModel(*ref, textureCache);
	}
	else {
		for (auto& rule : def.stateData.multipart) {
			bool match = rule.always;
			if (!match) {
				if (rule.isOr) { for (auto& c : rule.conditions) if (MatchesConditions(state, c)) { match = true; break; } }
				else match = rule.conditions.empty() || MatchesConditions(state, rule.conditions[0]);
			}
			if (!match) continue;
			BlockModel part = BuildGenericBlockModel(rule.apply, textureCache);
			result.quads.insert(result.quads.end(), part.quads.begin(), part.quads.end());
			result.triangles.insert(result.triangles.end(), part.triangles.begin(), part.triangles.end());
			result.isFullCube = result.isFullCube || part.isFullCube;
		}
	}
	return result;
}

VoxelShape BuildStateVoxelShape(const BlockDef& def, const std::map<std::string, std::string>& state) {
	VoxelShape s;
	if (def.isCross) { s.boxes.push_back(BoxPx(5, 0, 5, 11, 10, 11)); return s; }
	if (def.isFullCube) { s.boxes.push_back({ {0,0,0},{1,1,1} }); s.isFullCube = true; return s; }

	std::vector<JsonElementDef> elements;
	if (!def.stateData.isMultipart) {
		const StateModelRef* ref = nullptr;
		for (auto& [cond, r] : def.stateData.variants) if (cond == state) { ref = &r; break; }
		if (!ref && !def.stateData.variants.empty()) ref = &def.stateData.variants.front().second;
		if (ref && !ref->modelId.empty()) elements = LoadModelElementsRecursive(ResolveModelPath(ref->modelId));
	}
	else {
		for (auto& rule : def.stateData.multipart) {
			bool match = rule.always;
			if (!match) {
				if (rule.isOr) { for (auto& c : rule.conditions) if (MatchesConditions(state, c)) { match = true; break; } }
				else match = rule.conditions.empty() || MatchesConditions(state, rule.conditions[0]);
			}
			if (!match || rule.apply.modelId.empty()) continue;
			auto els = LoadModelElementsRecursive(ResolveModelPath(rule.apply.modelId));
			elements.insert(elements.end(), els.begin(), els.end());
		}
	}

	for (auto& el : elements) {
		Vector3 lo{ el.from.x / 16.0f, el.from.y / 16.0f, el.from.z / 16.0f };
		Vector3 hi{ el.to.x / 16.0f,   el.to.y / 16.0f,   el.to.z / 16.0f };
		if (hi.x < lo.x) std::swap(hi.x, lo.x);
		if (hi.y < lo.y) std::swap(hi.y, lo.y);
		if (hi.z < lo.z) std::swap(hi.z, lo.z);
		if ((hi.x - lo.x) < 1e-4f || (hi.y - lo.y) < 1e-4f || (hi.z - lo.z) < 1e-4f) continue;
		s.boxes.push_back({ lo, hi });
	}
	if (s.boxes.empty() && def.support == SupportType::None) {
		s.boxes.push_back({ {0,0,0},{1,1,1} });
		s.isFullCube = true;
	}
	return s;
}

static uint8_t KnownLightEmission(const std::string& name) {
	static const std::unordered_map<std::string, uint8_t> table = {
		{"torch",14}, {"wall_torch",14}, {"soul_torch",10}, {"soul_wall_torch",10},
		{"redstone_torch",7}, {"redstone_wall_torch",7},
		{"lava",15}, {"glowstone",15}, {"sea_lantern",15}, {"shroomlight",15},
		{"jack_o_lantern",15}, {"beacon",15}, {"end_rod",14}, {"lantern",15},
		{"soul_lantern",10}, {"campfire",15}, {"soul_campfire",10},
		{"redstone_lamp",15}, {"magma_block",3}, {"crying_obsidian",10},
		{"ochre_froglight",15}, {"verdant_froglight",15}, {"pearlescent_froglight",15},
		{"amethyst_cluster",5}, {"respawn_anchor",15}, {"candle",3}, {"sculk_catalyst",6},
		{"glow_lichen",7}, {"brewing_stand",1}, {"nether_portal",11}
	};
	auto it = table.find(name);
	return it == table.end() ? 0 : it->second;
}

class BlockRegistry {
public:
	std::vector<BlockDef> defs; // index == BlockId (индекс 0 зарезервирован/пуст)
	std::unordered_map<std::string, BlockId> nameToId;

	BlockId Resolve(const std::string& name) const {
		auto it = nameToId.find(name);
		return it == nameToId.end() ? INVALID_BLOCK_ID : it->second;
	}
	BlockDef& Get(BlockId id) { return defs[id]; }
	const BlockDef& Get(BlockId id) const { return defs[id]; }
	size_t Count() const { return defs.size(); }

	void LoadAll(const std::string& blockstatesDir, std::map<std::string, Texture2D>& textureCache) {
		defs.clear(); nameToId.clear();
		defs.push_back(BlockDef{}); // id 0 — пустой слот (настоящий "air" получит свой реальный id ниже)

		for (auto& entry : fs::directory_iterator(blockstatesDir)) {
			if (entry.path().extension() != ".json") continue;
			std::string name = entry.path().stem().string();

			std::ifstream f(entry.path());
			nlohmann::json j; f >> j;

			BlockDef def;
			def.name = name;

			auto registerProp = [&](const std::string& pk, const std::string& pv) {
				auto pit = std::find_if(def.properties.begin(), def.properties.end(),
					[&](const BlockPropertyDef& p) { return p.name == pk; });
				if (pit == def.properties.end()) def.properties.push_back({ pk, { pv } });
				else if (std::find(pit->values.begin(), pit->values.end(), pv) == pit->values.end())
					pit->values.push_back(pv);
				};

			if (j.contains("variants")) {
				for (auto& [key, val] : j["variants"].items()) {
					std::map<std::string, std::string> cond;
					if (!key.empty()) {
						size_t start = 0;
						while (start < key.size()) {
							size_t comma = key.find(',', start);
							std::string tok = key.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
							size_t eq = tok.find('=');
							if (eq != std::string::npos) cond[tok.substr(0, eq)] = tok.substr(eq + 1);
							if (comma == std::string::npos) break;
							start = comma + 1;
						}
					}
					const nlohmann::json& vv = val.is_array() ? val[0] : val;
					StateModelRef ref;
					ref.modelId = vv.value("model", std::string());
					ref.rotX = vv.value("x", 0);
					ref.rotY = vv.value("y", 0);
					ref.uvlock = vv.value("uvlock", false);
					for (auto& [pk, pv] : cond) registerProp(pk, pv);
					def.stateData.variants.push_back({ cond, ref });
				}
			}
			else if (j.contains("multipart")) {
				def.stateData.isMultipart = true;
				for (auto& caseJson : j["multipart"]) {
					MultipartRule rule;
					if (caseJson.contains("when")) {
						const auto& when = caseJson["when"];
						rule.always = false;
						if (when.contains("OR")) {
							rule.isOr = true;
							for (auto& c : when["OR"]) rule.conditions.push_back(JsonCondToMap(c));
						}
						else {
							rule.isOr = false;
							rule.conditions.push_back(JsonCondToMap(when));
						}
						for (auto& condMap : rule.conditions)
							for (auto& [pk, pv] : condMap) registerProp(pk, pv);
					}
					const nlohmann::json& applyJson = caseJson["apply"];
					const nlohmann::json& av = applyJson.is_array() ? applyJson[0] : applyJson;
					rule.apply.modelId = av.value("model", std::string());
					rule.apply.rotX = av.value("x", 0);
					rule.apply.rotY = av.value("y", 0);
					rule.apply.uvlock = av.value("uvlock", false);
					def.stateData.multipart.push_back(rule);
				}
			}
			else continue;

			for (auto& p : def.properties) def.defaultState[p.name] = PreferredDefaultFor(p.name, p.values);

			BlockId id = (BlockId)defs.size();
			def.id = id;
			nameToId[def.name] = id;
			nameToId["minecraft:" + def.name] = id;
			defs.push_back(std::move(def));
		}

		for (BlockDef& def : defs) {
			if (def.name.empty()) continue;
			BlockModel model = BuildStateModel(def, def.defaultState, textureCache);
			def.isFullCube = model.isFullCube;

			bool transparent = false;
			for (auto& q : model.quads) if (TextureHasAlpha(q.texture)) { transparent = true; break; }
			def.isOpaque = def.isFullCube && !transparent;

			std::string modelPath;
			if (!def.stateData.isMultipart && !def.stateData.variants.empty())
				modelPath = ResolveModelPath(def.stateData.variants.front().second.modelId);
			def.isCross = !modelPath.empty() && ModelHasCrossParent(modelPath);

			def.support = ClassifySupport(def.name, def.isCross);

			def.lightEmission = KnownLightEmission(def.name);

			if (def.isFullCube) {
				for (auto& q : model.quads) {
					int idx = -1;
					if (q.normal.y < -0.5f) idx = 0;
					else if (q.normal.y > 0.5f) idx = 1;
					else if (q.normal.z < -0.5f) idx = 2;
					else if (q.normal.z > 0.5f) idx = 3;
					else if (q.normal.x < -0.5f) idx = 4;
					else if (q.normal.x > 0.5f) idx = 5;
					if (idx >= 0) {
						def.cubeFaces[idx].texture = q.texture;
						def.cubeFaces[idx].tinted = q.tinted;
						if (def.name == "grass_block" && idx == 1) def.cubeFaces[idx].tinted = true; // верх grass_block красится вручную, как в ваниле
					}
				}
			}

			def.modelCache[""] = std::move(model);
			def.shapeCache[""] = BuildStateVoxelShape(def, def.defaultState);
		}

		g_AirId = Resolve("air");
		g_WaterId = Resolve("water");
		g_LavaId = Resolve("lava");
		g_BedrockId = Resolve("bedrock");

		TraceLog(LOG_INFO, "BlockRegistry: loaded %d block definitions", (int)defs.size() - 1);
	}
};

BlockRegistry g_blocks;
BlockId g_AirId = INVALID_BLOCK_ID, g_WaterId = INVALID_BLOCK_ID,
g_LavaId = INVALID_BLOCK_ID, g_BedrockId = INVALID_BLOCK_ID;

uint8_t GetBlockLightEmission(BlockId id)
{
	if (id == INVALID_BLOCK_ID || id >= g_blocks.Count())
		return 0;

	return g_blocks.Get(id).lightEmission;
}

inline bool IsOpaqueCube(BlockId b) {
	if (b == INVALID_BLOCK_ID || b >= g_blocks.Count()) return false;
	return g_blocks.Get(b).isOpaque;
}
inline bool IsCross(BlockId b) {
	if (b == INVALID_BLOCK_ID || b >= g_blocks.Count()) return false;
	return g_blocks.Get(b).isCross;
}
inline bool FaceVisible(BlockId self, BlockId neighbor) {
	return !IsOpaqueCube(neighbor) && neighbor != self;
}
inline SupportType GetSupportType(BlockId b) {
	if (b == INVALID_BLOCK_ID || b >= g_blocks.Count()) return SupportType::None;
	return g_blocks.Get(b).support;
}

inline bool IsFluidBlock(BlockId b) { return b == g_WaterId || b == g_LavaId; }

inline bool IsReplaceableByFluid(BlockId b) {
	if (b == INVALID_BLOCK_ID || b >= g_blocks.Count() || b == g_AirId)
		return false;
	if (IsFluidBlock(b))
		return false;
	if (IsCross(b))
		return true;
	SupportType s = GetSupportType(b);
	return s == SupportType::Floor || s == SupportType::WallSide || s == SupportType::Ceiling;
}

struct KnownBlockIds {
	BlockId stone, dirt, grass, sand, shortGrass, dandelion, poppy, anvil, oakLog, oakLeaves, torch;
} g_known;

void ResolveKnownBlockIds() {
	g_known.stone = g_blocks.Resolve("stone");
	g_known.dirt = g_blocks.Resolve("dirt");
	g_known.grass = g_blocks.Resolve("grass_block");
	g_known.sand = g_blocks.Resolve("sand");
	g_known.shortGrass = g_blocks.Resolve("short_grass");
	g_known.dandelion = g_blocks.Resolve("dandelion");
	g_known.poppy = g_blocks.Resolve("poppy");
	g_known.anvil = g_blocks.Resolve("anvil");
	g_known.oakLog = g_blocks.Resolve("oak_log");
	g_known.oakLeaves = g_blocks.Resolve("oak_leaves");
	g_known.torch = g_blocks.Resolve("torch");

	BlockId required[] = { g_AirId, g_WaterId, g_LavaId, g_BedrockId,
		g_known.stone, g_known.dirt, g_known.grass, g_known.sand, g_known.shortGrass,
		g_known.dandelion, g_known.poppy, g_known.anvil, g_known.oakLog, g_known.oakLeaves, g_known.torch };
	for (BlockId id : required)
		if (id == INVALID_BLOCK_ID) TraceLog(LOG_ERROR, "ResolveKnownBlockIds: обязательный блок не найден в resources/blockstates!");
}


const VoxelShape& GetBlockShape(BlockId id);
void DirtyChunksAroundBlock(std::unordered_map<int64_t, Chunk>& chunks, int bx, int by, int bz,
	int radius, bool geometryChange);

// Assign attachment rules from vanilla block identifiers. The registry remains data-driven:
// names only describe how blocks attach, not how their models are rendered.
void ConfigureBlockSupportTypes()
{
	for (BlockId id = 1; id < g_blocks.Count(); ++id) {
		BlockDef& def = g_blocks.Get(id);
		const std::string& n = def.name;
		def.support = SupportType::None;

		if (n == "air" || n == "cave_air" || n == "void_air" || n == "water" || n == "lava")
			continue;

		const bool floorAttached =
			n.find("flower") != std::string::npos || n.find("tulip") != std::string::npos ||
			n.find("torch") != std::string::npos || n.find("sapling") != std::string::npos ||
			n.find("mushroom") != std::string::npos || n.find("fungus") != std::string::npos ||
			n.find("roots") != std::string::npos || n.find("lichen") != std::string::npos ||
			n.find("rail") != std::string::npos || n.find("carpet") != std::string::npos ||
			n.find("pressure_plate") != std::string::npos || n.find("button") != std::string::npos ||
			n.find("dead_bush") != std::string::npos || n.find("sugar_cane") != std::string::npos ||
			n.find("wheat") != std::string::npos || n.find("carrots") != std::string::npos ||
			n.find("potatoes") != std::string::npos || n.find("beetroots") != std::string::npos ||
			n.find("torchflower") != std::string::npos || n.find("pitcher_plant") != std::string::npos ||
			n.find("azalea") != std::string::npos || n.find("bamboo_sapling") != std::string::npos ||
			n.find("lily_pad") != std::string::npos || n.find("small_dripleaf") != std::string::npos ||
			n.find("big_dripleaf") != std::string::npos || n.find("candle") != std::string::npos ||
			n.find("flower_pot") != std::string::npos || n.find("decorated_pot") != std::string::npos ||
			n.find("rail") != std::string::npos;

		const bool wallAttached =
			n.find("wall_torch") != std::string::npos || n.find("wall_sign") != std::string::npos ||
			n.find("wall_banner") != std::string::npos || n.find("wall_hanging_sign") != std::string::npos ||
			n.find("wall_skull") != std::string::npos || n.find("wall_head") != std::string::npos ||
			n.find("wall_") == 0 || n.find("ladder") != std::string::npos ||
			n.find("lever") != std::string::npos || n.find("tripwire_hook") != std::string::npos ||
			n.find("vine") != std::string::npos;

		const bool ceilingAttached =
			n.find("hanging_sign") != std::string::npos || n.find("lantern") != std::string::npos ||
			n.find("chain") != std::string::npos || n.find("pointed_dripstone") != std::string::npos;

		if (ceilingAttached) def.support = SupportType::Ceiling;
		else if (wallAttached) def.support = SupportType::WallSide;
		else if (floorAttached) def.support = SupportType::Floor;
		else if (def.isCross) def.support = SupportType::Floor;   // ← добавлено
	}
}

static bool IsAttachmentSupportBlock(BlockId id)
{
	if (id == INVALID_BLOCK_ID || id >= g_blocks.Count() || id == g_AirId || id == g_WaterId || id == g_LavaId)
		return false;
	if (IsCross(id)) return false;
	const VoxelShape& shape = GetBlockShape(id);
	return shape.isFullCube || !shape.boxes.empty() || IsOpaqueCube(id);
}

static bool HasSupportOnFace(const std::unordered_map<int64_t, Chunk>& chunks,
	int x, int y, int z, int dx, int dy, int dz)
{
	int cx = (int)floorf((float)x / CHUNK_SIZE);
	int cz = (int)floorf((float)z / CHUNK_SIZE);
	int lx = x - cx * CHUNK_SIZE;
	int lz = z - cz * CHUNK_SIZE;
	BlockId neighbor = GetWorldBlock(chunks, cx, cz, lx + dx, y + dy, lz + dz);
	if (!IsAttachmentSupportBlock(neighbor)) return false;

	const VoxelShape& shape = GetBlockShape(neighbor);
	if (shape.isFullCube || IsOpaqueCube(neighbor)) return true;
	for (const LocalAABB& box : shape.boxes) {
		if (dy == -1 && box.max.y >= 0.999f) return true;
		if (dy == 1 && box.min.y <= 0.001f) return true;
		if (dx == -1 && box.max.x >= 0.999f) return true;
		if (dx == 1 && box.min.x <= 0.001f) return true;
		if (dz == -1 && box.max.z >= 0.999f) return true;
		if (dz == 1 && box.min.z <= 0.001f) return true;
	}
	return false;
}

static bool BlockHasRequiredSupport(const std::unordered_map<int64_t, Chunk>& chunks,
	int x, int y, int z, BlockId id)
{
	if (id == INVALID_BLOCK_ID || id >= g_blocks.Count()) return true;
	const BlockDef& def = g_blocks.Get(id);
	if (def.support == SupportType::None) return true;
	if (def.support == SupportType::Floor)
		return HasSupportOnFace(chunks, x, y, z, 0, -1, 0);
	if (def.support == SupportType::Ceiling)
		return HasSupportOnFace(chunks, x, y, z, 0, 1, 0);

	// A wall-mounted block's facing points away from its support.
	auto it = def.defaultState.find("facing");
	if (it != def.defaultState.end()) {
		const std::string& facing = it->second;
		if (facing == "north") return HasSupportOnFace(chunks, x, y, z, 0, 0, 1);
		if (facing == "south") return HasSupportOnFace(chunks, x, y, z, 0, 0, -1);
		if (facing == "east") return HasSupportOnFace(chunks, x, y, z, -1, 0, 0);
		if (facing == "west") return HasSupportOnFace(chunks, x, y, z, 1, 0, 0);
	}
	return HasSupportOnFace(chunks, x, y, z, 0, -1, 0) ||
		HasSupportOnFace(chunks, x, y, z, 0, 1, 0) ||
		HasSupportOnFace(chunks, x, y, z, 1, 0, 0) ||
		HasSupportOnFace(chunks, x, y, z, -1, 0, 0) ||
		HasSupportOnFace(chunks, x, y, z, 0, 0, 1) ||
		HasSupportOnFace(chunks, x, y, z, 0, 0, -1);
}

// Called while the world-chunk mutex is held. Unsupported blocks disappear, and
// the queue naturally handles chains of attached blocks losing their support.
static void RemoveUnsupportedBlocks(std::unordered_map<int64_t, Chunk>& chunks,
	int startX, int startY, int startZ)
{
	static constexpr int dirs[6][3] = {
		{ 0,-1,0 }, { 0,1,0 }, { -1,0,0 }, { 1,0,0 }, { 0,0,-1 }, { 0,0,1 }
	};
	std::queue<std::array<int, 3>> pending;
	std::unordered_set<int64_t> visited;
	for (const auto& d : dirs) pending.push({ startX + d[0], startY + d[1], startZ + d[2] });

	while (!pending.empty()) {
		auto pos = pending.front();
		pending.pop();
		const int x = pos[0], y = pos[1], z = pos[2];
		if (y < 0 || y >= WORLD_HEIGHT) continue;
		const int cx = (int)floorf((float)x / CHUNK_SIZE);
		const int cz = (int)floorf((float)z / CHUNK_SIZE);
		const int lx = x - cx * CHUNK_SIZE;
		const int lz = z - cz * CHUNK_SIZE;
		const int64_t key = ChunkKey(cx, cz);
		if (!visited.insert((((int64_t)x & 0x1fffff) << 42) ^ (((int64_t)y & 0x3ff) << 32) ^ ((int64_t)z & 0xffffffff)).second)
			continue;
		auto chunkIt = chunks.find(key);
		if (chunkIt == chunks.end()) continue;
		BlockId current = chunkIt->second.Get(lx, y, lz);
		if (current == g_AirId || GetSupportType(current) == SupportType::None) continue;
		if (BlockHasRequiredSupport(chunks, x, y, z, current)) continue;

		chunkIt->second.Set(lx, y, lz, g_AirId);
		DirtyChunksAroundBlock(chunks, x, y, z, 14, true);
		for (const auto& d : dirs) pending.push({ x + d[0], y + d[1], z + d[2] });
	}
}

const VoxelShape& GetBlockShape(BlockId id) {
	static VoxelShape empty;
	if (id == INVALID_BLOCK_ID || id >= g_blocks.Count()) return empty;
	auto it = g_blocks.Get(id).shapeCache.find("");
	return it == g_blocks.Get(id).shapeCache.end() ? empty : it->second;
}

Color TintColorFor(BlockId id) {
	if (id >= g_blocks.Count()) return { 110,190,70,255 };
	const std::string& n = g_blocks.Get(id).name;
	if (n.find("leaves") != std::string::npos) return { 90,160,50,255 };
	if (id == g_WaterId) return { 63,118,228,255 };
	return { 110,190,70,255 };
}

// Список блоков для размещения (всё, кроме воздуха) — используется UI-комбобоксом
std::vector<BlockId> g_placeableBlocks;
void BuildPlaceableBlockList() {
	g_placeableBlocks.clear();
	static const char* kSkipNames[] = { "air", "cave_air", "void_air" };
	for (BlockId id = 1; id < g_blocks.Count(); id++) {
		const std::string& n = g_blocks.Get(id).name;
		bool skip = false;
		for (const char* s : kSkipNames) if (n == s) { skip = true; break; }
		if (!skip) g_placeableBlocks.push_back(id);
	}
	std::sort(g_placeableBlocks.begin(), g_placeableBlocks.end(),
		[](BlockId a, BlockId b) { return g_blocks.Get(a).name < g_blocks.Get(b).name; });
}

void DrawBlockOutline(const VoxelShape& shape, Vector3 blockOrigin, Vector3 cameraPos,
	const std::unordered_map<int64_t, Chunk>& chunks)
{
	const float inflate = 0.004f;
	const Color outlineColor = { 0, 0, 0, 180 };

	int bx = (int)floorf(blockOrigin.x);
	int by = (int)floorf(blockOrigin.y);
	int bz = (int)floorf(blockOrigin.z);
	int cx = (int)floorf(blockOrigin.x / (float)CHUNK_SIZE);
	int cz = (int)floorf(blockOrigin.z / (float)CHUNK_SIZE);
	int lx = bx - cx * CHUNK_SIZE;
	int lz = bz - cz * CHUNK_SIZE;

	struct Edge { int v0, v1; int faceA, faceB; };
	const Edge cubeEdges[12] = {
		{ 0, 1, 0, 2 }, { 1, 5, 0, 5 }, { 5, 4, 0, 3 }, { 4, 0, 0, 4 },
		{ 3, 2, 1, 2 }, { 2, 6, 1, 5 }, { 6, 7, 1, 3 }, { 7, 3, 1, 4 },
		{ 0, 3, 2, 4 }, { 1, 2, 2, 5 }, { 5, 6, 3, 5 }, { 4, 7, 3, 4 }
	};

	const int dx[6] = { 0, 0, 0, 0, -1, 1 };
	const int dy[6] = { -1, 1, 0, 0, 0, 0 };
	const int dz[6] = { 0, 0, -1, 1, 0, 0 };

	for (const auto& box : shape.boxes) {
		Vector3 lo = { box.min.x - inflate, box.min.y - inflate, box.min.z - inflate };
		Vector3 hi = { box.max.x + inflate, box.max.y + inflate, box.max.z + inflate };

		Vector3 v[8] = {
			{ lo.x, lo.y, lo.z }, { hi.x, lo.y, lo.z }, { hi.x, hi.y, lo.z }, { lo.x, hi.y, lo.z },
			{ lo.x, lo.y, hi.z }, { hi.x, lo.y, hi.z }, { hi.x, hi.y, hi.z }, { lo.x, hi.y, hi.z }
		};

		for (int i = 0; i < 8; i++) {
			v[i] = Vector3Add(blockOrigin, v[i]);
		}

		bool faceVisible[6] = { false };
		Vector3 faceNormals[6] = { {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}, {-1,0,0}, {1,0,0} };

		for (int f = 0; f < 6; f++) {
			Vector3 edgeInBox = (f % 2 == 0) ? lo : hi;
			Vector3 localCenter = {
				(f < 4) ? (lo.x + hi.x) * 0.5f : edgeInBox.x,
				(f < 2) ? edgeInBox.y : (lo.y + hi.y) * 0.5f,
				(f >= 2 && f < 4) ? edgeInBox.z : (lo.z + hi.z) * 0.5f
			};
			Vector3 worldFaceCenter = Vector3Add(blockOrigin, localCenter);
			Vector3 toCamera = Vector3Subtract(cameraPos, worldFaceCenter);

			if (Vector3DotProduct(faceNormals[f], toCamera) > 0.0f) {
				if (shape.isFullCube) {
					BlockId neighbor = GetWorldBlock(chunks, cx, cz, lx + dx[f], by + dy[f], lz + dz[f]);
					if (!IsOpaqueCube(neighbor)) {
						faceVisible[f] = true;
					}
				}
				else {
					faceVisible[f] = true;
				}
			}
		}

		for (int i = 0; i < 12; i++) {
			bool fA = faceVisible[cubeEdges[i].faceA];
			bool fB = faceVisible[cubeEdges[i].faceB];

			if (fA || fB) {
				DrawLine3D(v[cubeEdges[i].v0], v[cubeEdges[i].v1], outlineColor);
			}
		}
	}
}

BlockHit RaycastBlock(const std::unordered_map<int64_t, Chunk>& chunks,
	Vector3 origin, Vector3 direction, float maxDistance)
{
	BlockHit result;
	direction = Vector3Normalize(direction);

	int x = (int)floorf(origin.x), y = (int)floorf(origin.y), z = (int)floorf(origin.z);
	int stepX = direction.x > 0 ? 1 : -1;
	int stepY = direction.y > 0 ? 1 : -1;
	int stepZ = direction.z > 0 ? 1 : -1;

	auto safeDiv = [](float num, float den) { return fabsf(den) < 1e-9f ? FLT_MAX : num / den; };
	auto tMax0 = [](float o, int voxel, int step) { return (float)(step > 0 ? (voxel + 1) : voxel) - o; };

	float tMaxX = safeDiv(tMax0(origin.x, x, stepX), direction.x);
	float tMaxY = safeDiv(tMax0(origin.y, y, stepY), direction.y);
	float tMaxZ = safeDiv(tMax0(origin.z, z, stepZ), direction.z);
	float tDeltaX = safeDiv(1.0f, fabsf(direction.x));
	float tDeltaY = safeDiv(1.0f, fabsf(direction.y));
	float tDeltaZ = safeDiv(1.0f, fabsf(direction.z));

	float traveled = 0.0f;
	while (traveled <= maxDistance) {
		if (y >= 0 && y < WORLD_HEIGHT) {
			int cx = (int)floorf((float)x / CHUNK_SIZE);
			int cz = (int)floorf((float)z / CHUNK_SIZE);
			int lx = x - cx * CHUNK_SIZE;
			int lz = z - cz * CHUNK_SIZE;
			BlockId type = GetWorldBlock(chunks, cx, cz, lx, y, lz);
			if (type != g_AirId && type != g_WaterId && type != g_LavaId) {
				const VoxelShape& shape = GetBlockShape(type);
				if (!shape.boxes.empty()) {
					Vector3 blockOrigin{ (float)x, (float)y, (float)z };
					float t; Vector3 n;
					if (RaycastVoxelShape(shape, blockOrigin, origin, direction, maxDistance, t, n)) {
						result.hit = true;
						result.x = x; result.y = y; result.z = z; result.type = type;
						result.nx = (int)roundf(n.x);
						result.ny = (int)roundf(n.y);
						result.nz = (int)roundf(n.z);
						return result;
					}
				}
			}
		}
		if (tMaxX < tMaxY) {
			if (tMaxX < tMaxZ) { x += stepX; traveled = tMaxX; tMaxX += tDeltaX; }
			else { z += stepZ; traveled = tMaxZ; tMaxZ += tDeltaZ; }
		}
		else {
			if (tMaxY < tMaxZ) { y += stepY; traveled = tMaxY; tMaxY += tDeltaY; }
			else { z += stepZ; traveled = tMaxZ; tMaxZ += tDeltaZ; }
		}
	}
	return result;
}

FastNoiseLite g_terrainNoise;

void InitTerrainNoise(int seed) {
	g_terrainNoise.SetSeed(seed);
	g_terrainNoise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
	g_terrainNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
	g_terrainNoise.SetFractalOctaves(4);
	g_terrainNoise.SetFractalLacunarity(2.0f);
	g_terrainNoise.SetFractalGain(0.5f);
	g_terrainNoise.SetFrequency(0.015f);
}

int g_worldSeed = 0;

inline uint32_t HashU32(uint32_t h) {
	h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
	return h;
}
float Hash01(int x, int y, int z, int salt) {
	uint32_t h = HashU32((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^
		(uint32_t)z * 83492791u ^ (uint32_t)(g_worldSeed + salt) * 2654435761u);
	return (h & 0xFFFFFF) / 16777216.0f;
}

int SurfaceHeightAt(int wx, int wz) {
	float raw = g_terrainNoise.GetNoise((float)wx, (float)wz);
	return 20 + (int)(((raw + 1.0f) * 0.5f) * 40);
}

FastNoiseLite g_caveA, g_caveB, g_cheese;

void InitCaveNoise(int seed) {
	for (FastNoiseLite* n : { &g_caveA, &g_caveB, &g_cheese }) {
		n->SetNoiseType(FastNoiseLite::NoiseType_Perlin);
		n->SetFractalType(FastNoiseLite::FractalType_None);
	}
	g_caveA.SetSeed(seed + 1000);  g_caveA.SetFrequency(0.018f);
	g_caveB.SetSeed(seed + 2000);  g_caveB.SetFrequency(0.018f);
	g_cheese.SetSeed(seed + 3000); g_cheese.SetFrequency(0.012f);
}

bool IsCaveAt(int wx, int wy, int wz, int surface) {
	if (wy <= 5) return false;
	int depth = surface - wy;
	if (depth < 0) return false;
	if (surface <= SEA_LEVEL + 1 && depth < 6) return false;

	float x = (float)wx, y = (float)wy, z = (float)wz;

	float a = g_caveA.GetNoise(x, y * 1.4f, z);
	float b = g_caveB.GetNoise(x, y * 1.4f, z);
	float t = 0.06f + 0.04f * std::min(depth / 40.0f, 1.0f);
	if (a * a + b * b < t * t) return true;

	if (depth > 10) {
		if (g_cheese.GetNoise(x, y * 2.0f, z) > 0.45f) return true;
	}
	return false;
}

int GenerateRandomSeed() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(1, 1000000);
	return dist(gen);
}

struct MeshData {
	struct Part {
		Texture2D texture{};
		bool tinted = false;
		Color tintColor = WHITE;
		std::vector<MeshVertex> vertices;
		std::vector<unsigned short> indices;
	};
	std::vector<Part> parts;
	std::vector<Part> waterParts;
};


static int ChunkMaxNonAirY(const Chunk& c)
{
	if (c.mode == Chunk::Mode::Single)
		return c.singleBlock == g_AirId ? -1 : WORLD_HEIGHT - 1;

	if (g_AirId == INVALID_BLOCK_ID)
		return -1;

	for (int y = WORLD_HEIGHT - 1; y >= 0; --y)
	{
		for (int x = 0; x < CHUNK_SIZE; ++x)
		{
			for (int z = 0; z < CHUNK_SIZE; ++z)
			{
				if (c.Get(x, y, z) != g_AirId)
					return y;
			}
		}
	}

	return -1;
}

static float ShadeForNormal(Vector3 n) {
	if (n.y > 0.5f) return 1.0f;
	if (n.y < -0.5f) return 0.5f;
	if (n.z < -0.5f) return 0.8f;
	if (n.z > 0.5f) return 0.8f;
	return 0.6f;
}

inline int MaxFluidLevel(BlockId fluidType) { return (fluidType == g_LavaId) ? 4 : 8; }

MeshData BuildChunkMeshData(const Chunk& chunk, const std::unordered_map<int64_t, Chunk>& allChunks)
{
	using PartMap = std::map<uint64_t, MeshData::Part>;
	PartMap solidParts, waterParts;
	std::map<unsigned int, unsigned int> splitIndex;

	auto addQuad = [&](PartMap& dst, const FaceInfo& face, Color tintColor,
		Vector3 v1, Vector2 uv1, Vector3 v2, Vector2 uv2,
		Vector3 v3, Vector2 uv3, Vector3 v4, Vector2 uv4,
		Vector3 normal, float shade, float sky, float blk)
		{
			unsigned int& split = splitIndex[face.texture.id];
			MeshData::Part* part = &dst[(uint64_t)face.texture.id | ((uint64_t)split << 32)];
			if (part->vertices.size() + 4 > 65000) {
				split++;
				part = &dst[(uint64_t)face.texture.id | ((uint64_t)split << 32)];
			}
			part->texture = face.texture;

			if (face.tinted) {
				part->tinted = true;
				part->tintColor = tintColor;
			}

			float vs = (float)face.texture.width / (float)face.texture.height;
			uv1.y *= vs; uv2.y *= vs; uv3.y *= vs; uv4.y *= vs;

			Color col = {
				(unsigned char)(255.0f * shade),
				(unsigned char)(255.0f * sky),
				(unsigned char)(255.0f * blk),
				255
			};

			unsigned short base = (unsigned short)part->vertices.size();
			part->vertices.push_back({ v1, uv1, normal, col });
			part->vertices.push_back({ v2, uv2, normal, col });
			part->vertices.push_back({ v3, uv3, normal, col });
			part->vertices.push_back({ v4, uv4, normal, col });
			part->indices.insert(part->indices.end(),
				{ base, (unsigned short)(base + 1), (unsigned short)(base + 2),
				  base, (unsigned short)(base + 2), (unsigned short)(base + 3) });
		};

	auto addQuadSmooth = [&](PartMap& dst, const FaceInfo& face, Color tintColor,
		Vector3 v1, Vector2 uv1, Vector2 l1,
		Vector3 v2, Vector2 uv2, Vector2 l2,
		Vector3 v3, Vector2 uv3, Vector2 l3,
		Vector3 v4, Vector2 uv4, Vector2 l4,
		Vector3 normal, float shade)
		{
			unsigned int& split = splitIndex[face.texture.id];
			MeshData::Part* part = &dst[(uint64_t)face.texture.id | ((uint64_t)split << 32)];
			if (part->vertices.size() + 4 > 65000) {
				split++;
				part = &dst[(uint64_t)face.texture.id | ((uint64_t)split << 32)];
			}
			part->texture = face.texture;

			if (face.tinted) {
				part->tinted = true;
				part->tintColor = tintColor;
			}

			float vs = (float)face.texture.width / (float)face.texture.height;
			uv1.y *= vs; uv2.y *= vs; uv3.y *= vs; uv4.y *= vs;

			auto colFor = [&](Vector2 l) -> Color {
				return {
					(unsigned char)(255.0f * shade),
					(unsigned char)(255.0f * l.x),
					(unsigned char)(255.0f * l.y),
					255
				};
				};

			unsigned short base = (unsigned short)part->vertices.size();
			part->vertices.push_back({ v1, uv1, normal, colFor(l1) });
			part->vertices.push_back({ v2, uv2, normal, colFor(l2) });
			part->vertices.push_back({ v3, uv3, normal, colFor(l3) });
			part->vertices.push_back({ v4, uv4, normal, colFor(l4) });
			part->indices.insert(part->indices.end(),
				{ base, (unsigned short)(base + 1), (unsigned short)(base + 2),
				  base, (unsigned short)(base + 2), (unsigned short)(base + 3) });
		};

	// Универсальный путь: рисует ПОЛНУЮ geometry блока из реестра (без culling граней,
	// подходит для всего, что не является быстрым полным кубом: лестницы, плиты, заборы,
	// двери, факелы, наковальня и т.д.)
	auto addGenericModel = [&](BlockId block, float wx, float wy, float wz, float sky, float blk) {
		if (block >= g_blocks.Count()) return;
		const BlockDef& def = g_blocks.Get(block);
		auto it = def.modelCache.find("");
		if (it == def.modelCache.end()) return;
		Vector3 origin{ wx, wy, wz };
		Color tint = TintColorFor(block);
		float constShade = (def.support != SupportType::None && !def.isCross) ? 1.0f : -1.0f; // факелы/баннеры и т.п. светятся ровно
		for (const auto& q : it->second.quads) {
			FaceInfo face{ q.texture, q.tinted };
			float shade = (constShade > 0.0f) ? constShade : ShadeForNormal(q.normal);
			addQuad(solidParts, face, tint,
				Vector3Add(origin, q.v[0]), q.uv[0],
				Vector3Add(origin, q.v[1]), q.uv[1],
				Vector3Add(origin, q.v[2]), q.uv[2],
				Vector3Add(origin, q.v[3]), q.uv[3],
				q.normal, shade, sky, blk);
		}
		};

	auto addCross = [&](BlockId block, const FaceInfo& face, float wx, float wy, float wz, float sky, float blk) {
		Vector3 n = { 0, 1, 0 };
		Color tint = TintColorFor(block);
		auto plane = [&](Vector3 bl, Vector3 br, Vector3 tr, Vector3 tl) {
			addQuad(solidParts, face, tint, bl, { 0,1 }, tl, { 0,0 }, tr, { 1,0 }, br, { 1,1 }, n, 1.0f, sky, blk);
			addQuad(solidParts, face, tint, br, { 1,1 }, tr, { 1,0 }, tl, { 0,0 }, bl, { 0,1 }, n, 1.0f, sky, blk);
			};
		plane({ wx, wy, wz }, { wx + 1, wy, wz + 1 }, { wx + 1, wy + 1, wz + 1 }, { wx, wy + 1, wz });
		plane({ wx + 1, wy, wz }, { wx, wy, wz + 1 }, { wx, wy + 1, wz + 1 }, { wx + 1, wy + 1, wz });
		};

	const Chunk* nbs[3][3];
	for (int iz = 0; iz < 3; ++iz) {
		for (int ix = 0; ix < 3; ++ix) {
			if (ix == 1 && iz == 1) {
				nbs[ix][iz] = &chunk;
				continue;
			}
			auto it = allChunks.find(ChunkKey(chunk.chunkX + ix - 1, chunk.chunkZ + iz - 1));
			nbs[ix][iz] = (it == allChunks.end()) ? nullptr : &it->second;
		}
	}

	auto blockAt = [&](int x, int y, int z) -> BlockId {
		if (y < 0) return g_BedrockId;
		if (y >= WORLD_HEIGHT) return g_AirId;
		int ix = 1, iz = 1, lx = x, lz = z;
		if (x < 0) { ix = 0; lx = x + CHUNK_SIZE; }
		else if (x >= CHUNK_SIZE) { ix = 2; lx = x - CHUNK_SIZE; }
		if (z < 0) { iz = 0; lz = z + CHUNK_SIZE; }
		else if (z >= CHUNK_SIZE) { iz = 2; lz = z - CHUNK_SIZE; }
		const Chunk* c = nbs[ix][iz];
		if (!c) return g_AirId;
		return c->Get(lx, y, lz);
		};

	int maxY = 0;
	for (int iz = 0; iz < 3; ++iz) {
		for (int ix = 0; ix < 3; ++ix) {
			if (!nbs[ix][iz]) continue;
			int y = ChunkMaxNonAirY(*nbs[ix][iz]);
			if (y > maxY) maxY = y;
		}
	}
	int LY = maxY + 2;
	if (LY < 2) LY = 2;
	if (LY > WORLD_HEIGHT) LY = WORLD_HEIGHT;

	// Padding must be >= max block-light level (14) so that light from neighbouring
	// chunks is fully accounted for and can propagate into this chunk.
	const int LP = 16;
	const int LX = CHUNK_SIZE + LP * 2;
	const int LZ = CHUNK_SIZE + LP * 2;
	const int GRID = LX * LY * LZ;

	thread_local static std::vector<uint8_t> isOpaque, skyL, blkL;
	thread_local static std::vector<int> lightQ;
	isOpaque.assign(GRID, 1);
	skyL.assign(GRID, 0);
	blkL.assign(GRID, 0);

	auto idl = [&](int x, int y, int z) { return (x * LY + y) * LZ + z; };

	std::vector<int> torchSeeds;
	torchSeeds.reserve(32);
	std::vector<int> skySeeds;
	skySeeds.reserve(LX * LZ);

	for (int x = 0; x < LX; x++) {
		for (int z = 0; z < LZ; z++) {
			int bx = x - LP, bz = z - LP;
			uint8_t s = 15;
			for (int y = LY - 1; y >= 0; y--) {
				BlockId t = blockAt(bx, y, bz);
				bool op = IsOpaqueCube(t);
				int i = idl(x, y, z);
				isOpaque[i] = op ? 1 : 0;
				if (op) s = 0;
				skyL[i] = op ? 0 : s;
				if (!op && s == 15) skySeeds.push_back(i);
				if (t < g_blocks.Count() && g_blocks.Get(t).lightEmission > 0) {
					blkL[i] = g_blocks.Get(t).lightEmission;
					torchSeeds.push_back(i);
				}
			}
		}
	}

	const int dx6[6] = { -1, 1, 0, 0, 0, 0 };
	const int dy6[6] = { 0, 0, -1, 1, 0, 0 };
	const int dz6[6] = { 0, 0, 0, 0, -1, 1 };

	auto flood = [&](std::vector<uint8_t>& grid, const std::vector<int>& seeds) {
		lightQ.clear();
		if (lightQ.capacity() < 4096) lightQ.reserve(4096);
		lightQ.insert(lightQ.end(), seeds.begin(), seeds.end());

		size_t head = 0;
		while (head < lightQ.size()) {
			int i = lightQ[head++];
			uint8_t lv = grid[i];
			if (lv <= 1) continue;
			uint8_t next = (uint8_t)(lv - 1);
			int z = i % LZ;
			int tmp = i / LZ;
			int y = tmp % LY;
			int x = tmp / LY;
			for (int k = 0; k < 6; ++k) {
				int nx = x + dx6[k], ny = y + dy6[k], nz = z + dz6[k];
				if ((unsigned)nx >= (unsigned)LX || (unsigned)ny >= (unsigned)LY || (unsigned)nz >= (unsigned)LZ) continue;
				int ni = (nx * LY + ny) * LZ + nz;
				if (isOpaque[ni]) continue;
				if (grid[ni] < next) {
					grid[ni] = next;
					if (next > 1) lightQ.push_back(ni);
				}
			}
		}
		};

	flood(skyL, skySeeds);
	if (!torchSeeds.empty()) flood(blkL, torchSeeds);

	// Store the computed lightmaps into the chunk so they persist and can be
	// reused / inspected. This is the key architectural separation: light is
	// computed once and written into Chunk, mesh generation only reads it.
	{
		// chunk is const in the signature, but we need to write light. The caller
		// (worker) holds a mutable local copy, so we const_cast here intentionally.
		Chunk& mutableChunk = const_cast<Chunk&>(chunk);
		mutableChunk.EnsureLightAllocated();
		for (int x = 0; x < CHUNK_SIZE; ++x) {
			for (int y = 0; y < LY && y < WORLD_HEIGHT; ++y) {
				for (int z = 0; z < CHUNK_SIZE; ++z) {
					int i = idl(x + LP, y, z + LP);
					int li = Chunk::LightIndex(x, y, z);
					mutableChunk.skyLight[li] = skyL[i];
					mutableChunk.blockLight[li] = blkL[i];
				}
			}
		}
		// Higher Y not covered by LY stay 0 (already zeroed by EnsureLightAllocated)
		// lightmaps stored; dirty is cleared by the streaming system after upload
	}

	auto sampleLight = [&](int x, int y, int z) -> Vector2 {
		if (y < 0) return { 0.0f, 0.0f };
		if (y >= LY) return { 1.0f, 0.0f };
		int px = x + LP, pz = z + LP;
		if (px < 0 || px >= LX || pz < 0 || pz >= LZ) return { 1.0f, 0.0f };
		int i = idl(px, y, pz);
		return { skyL[i] / 15.0f, blkL[i] / 15.0f };
		};

	for (int y = 0; y < WORLD_HEIGHT; y++) {
		if (chunk.mode == Chunk::Mode::Layered && chunk.layers && chunk.layers[y].uniform &&
			chunk.layers[y].id == g_AirId) continue;

		for (int x = 0; x < CHUNK_SIZE; x++) {
			for (int z = 0; z < CHUNK_SIZE; z++) {
				BlockId block = chunk.Get(x, y, z);
				if (block == g_AirId || block == INVALID_BLOCK_ID || block >= g_blocks.Count()) continue;
				const BlockDef& def = g_blocks.Get(block);

				float wx = (float)(chunk.chunkX * CHUNK_SIZE + x);
				float wy = (float)y;
				float wz = (float)(chunk.chunkZ * CHUNK_SIZE + z);

				if (block == g_WaterId || block == g_LavaId) {
					PartMap& dst = waterParts;
					bool isLava = (block == g_LavaId);

					auto fluidLevelAt = [&](int ax, int ay, int az) -> int {
						if (ay < 0 || ay >= WORLD_HEIGHT) return 0;
						int ix = 1, iz = 1, lx = ax, lz = az;
						if (ax < 0) { ix = 0; lx = ax + CHUNK_SIZE; }
						else if (ax >= CHUNK_SIZE) { ix = 2; lx = ax - CHUNK_SIZE; }
						if (az < 0) { iz = 0; lz = az + CHUNK_SIZE; }
						else if (az >= CHUNK_SIZE) { iz = 2; lz = az - CHUNK_SIZE; }
						const Chunk* c = nbs[ix][iz];
						if (!c) return 0;
						if (c->Get(lx, ay, lz) != block) return 0;
						auto fit = c->fluidLevels.find(Chunk::LightIndex(lx, ay, lz));
						if (fit == c->fluidLevels.end()) return MaxFluidLevel(block);
						return fit->second & 0x0F;
						};

					int selfLevel = fluidLevelAt(x, y, z);
					if (selfLevel <= 0) selfLevel = MaxFluidLevel(block);

					BlockId nN = blockAt(x, y, z - 1), nS = blockAt(x, y, z + 1);
					BlockId nW = blockAt(x - 1, y, z), nE = blockAt(x + 1, y, z);
					BlockId nU = blockAt(x, y + 1, z), nD = blockAt(x, y - 1, z);

					float h = (nU == block) ? 1.0f : (std::max)(0.11f, 0.9f * ((float)selfLevel / (float)MaxFluidLevel(block)));

					Color tint = isLava ? WHITE : TintColorFor(block);
					FaceInfo wf{ isLava ? g_lavaTexture : g_waterTexture, !isLava };

					auto sideVisible = [&](BlockId neighbor, int nx, int ny, int nz) {
						if (neighbor == block) {
							int nLevel = fluidLevelAt(nx, ny, nz);
							float nh = (blockAt(nx, ny + 1, nz) == block) ? 1.0f : (std::max)(0.11f, 0.9f * ((float)nLevel / (float)MaxFluidLevel(block)));
							return nh < h - 0.001f;
						}
						return !IsOpaqueCube(neighbor);
						};

					Vector3 p000 = { wx, wy, wz }, p100 = { wx + 1, wy, wz };
					Vector3 p010 = { wx, wy + h, wz }, p110 = { wx + 1, wy + h, wz };
					Vector3 p001 = { wx, wy, wz + 1 }, p101 = { wx + 1, wy, wz + 1 };
					Vector3 p011 = { wx, wy + h, wz + 1 }, p111 = { wx + 1, wy + h, wz + 1 };

					if (sideVisible(nN, x, y, z - 1)) { Vector2 L = sampleLight(x, y, z - 1); addQuad(dst, wf, tint, p000, { 0,1 }, p010, { 0,0 }, p110, { 1,0 }, p100, { 1,1 }, { 0,0,-1 }, 0.8f, L.x, L.y); }
					if (sideVisible(nS, x, y, z + 1)) { Vector2 L = sampleLight(x, y, z + 1); addQuad(dst, wf, tint, p001, { 0,1 }, p101, { 1,1 }, p111, { 1,0 }, p011, { 0,0 }, { 0,0, 1 }, 0.8f, L.x, L.y); }
					if (sideVisible(nW, x - 1, y, z)) { Vector2 L = sampleLight(x - 1, y, z); addQuad(dst, wf, tint, p000, { 0,1 }, p001, { 1,1 }, p011, { 1,0 }, p010, { 0,0 }, { -1,0,0 }, 0.6f, L.x, L.y); }
					if (sideVisible(nE, x + 1, y, z)) { Vector2 L = sampleLight(x + 1, y, z); addQuad(dst, wf, tint, p100, { 0,1 }, p110, { 0,0 }, p111, { 1,0 }, p101, { 1,1 }, { 1,0,0 }, 0.6f, L.x, L.y); }
					if (FaceVisible(block, nU)) { Vector2 L = sampleLight(x, y + 1, z); addQuad(dst, wf, tint, p010, { 0,0 }, p011, { 0,1 }, p111, { 1,1 }, p110, { 1,0 }, { 0,1,0 }, 1.0f, L.x, L.y); }
					if (FaceVisible(block, nD)) { Vector2 L = sampleLight(x, y - 1, z); addQuad(dst, wf, tint, p000, { 0,0 }, p100, { 1,0 }, p101, { 1,1 }, p001, { 0,1 }, { 0,-1,0 }, 0.5f, L.x, L.y); }
					continue;
				}

				if (def.isCross) {
					Vector2 L = sampleLight(x, y, z);
					const auto& mq = def.modelCache.at("").quads;
					FaceInfo face{ mq.empty() ? Texture2D{} : mq[0].texture, mq.empty() ? false : mq[0].tinted };
					addCross(block, face, wx, wy, wz, L.x, L.y);
					continue;
				}

				if (!(def.isFullCube && def.isOpaque)) {
					// Универсальный путь: любые нестандартные формы (лестницы, плиты, заборы,
					// двери, факелы, наковальня, знаки и т.д.) — без culling граней.
					Vector2 L = sampleLight(x, y, z);
					addGenericModel(block, wx, wy, wz, L.x, L.y);
					continue;
				}

				// --- быстрый путь для непрозрачных полных кубов ---
				BlockId nN = blockAt(x, y, z - 1), nS = blockAt(x, y, z + 1);
				BlockId nW = blockAt(x - 1, y, z), nE = blockAt(x + 1, y, z);
				BlockId nU = blockAt(x, y + 1, z), nD = blockAt(x, y - 1, z);

				int solidNeighbors = (int)IsOpaqueCube(nN) + (int)IsOpaqueCube(nS) +
					(int)IsOpaqueCube(nW) + (int)IsOpaqueCube(nE) +
					(int)IsOpaqueCube(nU) + (int)IsOpaqueCube(nD);
				float ao = 1.0f - solidNeighbors * 0.035f;
				ao = (std::max)(0.78f, ao);
				Color tint = TintColorFor(block);

				Vector3 p000 = { wx, wy, wz }, p100 = { wx + 1, wy, wz };
				Vector3 p010 = { wx, wy + 1, wz }, p110 = { wx + 1, wy + 1, wz };
				Vector3 p001 = { wx, wy, wz + 1 }, p101 = { wx + 1, wy, wz + 1 };
				Vector3 p011 = { wx, wy + 1, wz + 1 }, p111 = { wx + 1, wy + 1, wz + 1 };

				if (FaceVisible(block, nN)) { Vector2 L = sampleLight(x, y, z - 1); addQuad(solidParts, def.cubeFaces[2], tint, p000, { 0,1 }, p010, { 0,0 }, p110, { 1,0 }, p100, { 1,1 }, { 0,0,-1 }, 0.8f * ao, L.x, L.y); }
				if (FaceVisible(block, nS)) { Vector2 L = sampleLight(x, y, z + 1); addQuad(solidParts, def.cubeFaces[3], tint, p001, { 0,1 }, p101, { 1,1 }, p111, { 1,0 }, p011, { 0,0 }, { 0,0, 1 }, 0.8f * ao, L.x, L.y); }
				if (FaceVisible(block, nW)) { Vector2 L = sampleLight(x - 1, y, z); addQuad(solidParts, def.cubeFaces[4], tint, p000, { 0,1 }, p001, { 1,1 }, p011, { 1,0 }, p010, { 0,0 }, { -1,0,0 }, 0.6f * ao, L.x, L.y); }
				if (FaceVisible(block, nE)) { Vector2 L = sampleLight(x + 1, y, z); addQuad(solidParts, def.cubeFaces[5], tint, p100, { 0,1 }, p110, { 0,0 }, p111, { 1,0 }, p101, { 1,1 }, { 1,0,0 }, 0.6f * ao, L.x, L.y); }
				if (FaceVisible(block, nU)) {
					auto cornerLight = [&](int ox, int oz) -> Vector2 {
						Vector2 a = sampleLight(x, y + 1, z);
						Vector2 b = sampleLight(x + ox, y + 1, z);
						Vector2 c = sampleLight(x, y + 1, z + oz);
						Vector2 d = sampleLight(x + ox, y + 1, z + oz);
						return { (a.x + b.x + c.x + d.x) * 0.25f, (a.y + b.y + c.y + d.y) * 0.25f };
						};
					Vector2 L00 = cornerLight(-1, -1), L01 = cornerLight(-1, 1);
					Vector2 L11 = cornerLight(1, 1), L10 = cornerLight(1, -1);
					addQuadSmooth(solidParts, def.cubeFaces[1], tint, p010, { 0,0 }, L00, p011, { 0,1 }, L01,
						p111, { 1,1 }, L11, p110, { 1,0 }, L10, { 0,1,0 }, 1.0f * ao);
				}
				if (FaceVisible(block, nD)) { Vector2 L = sampleLight(x, y - 1, z); addQuad(solidParts, def.cubeFaces[0], tint, p000, { 0,0 }, p100, { 1,0 }, p101, { 1,1 }, p001, { 0,1 }, { 0,-1,0 }, 0.5f * ao, L.x, L.y); }
			}
		}
	}

	MeshData result;
	for (auto& [k, part] : solidParts) if (!part.vertices.empty()) result.parts.push_back(std::move(part));
	for (auto& [k, part] : waterParts) if (!part.vertices.empty()) result.waterParts.push_back(std::move(part));
	return result;
}

static Model BuildModelFromParts(std::vector<MeshData::Part>& parts, bool /*cutout*/) {
	Model model = { 0 };
	model.meshCount = (int)parts.size();
	model.materialCount = model.meshCount;
	model.meshes = (Mesh*)MemAlloc(model.meshCount * sizeof(Mesh));
	model.materials = (Material*)MemAlloc(model.materialCount * sizeof(Material));
	model.meshMaterial = (int*)MemAlloc(model.meshCount * sizeof(int));

	for (int i = 0; i < model.meshCount; i++) {
		auto& part = parts[i];
		Mesh mesh = { 0 };
		mesh.vertexCount = (int)part.vertices.size();
		mesh.triangleCount = (int)part.indices.size() / 3;

		const size_t vertexBytes = static_cast<size_t>(mesh.vertexCount) * 3 * sizeof(float);
		if (vertexBytes > UINT_MAX) continue;

		mesh.vertices = static_cast<float*>(MemAlloc(
			static_cast<unsigned int>(vertexBytes)
		));
		mesh.texcoords = (float*)MemAlloc(mesh.vertexCount * 2 * sizeof(float));
		mesh.normals = (float*)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
		mesh.colors = (unsigned char*)MemAlloc(mesh.vertexCount * 4);
		mesh.indices = (unsigned short*)MemAlloc(part.indices.size() * sizeof(unsigned short));

		for (int v = 0; v < mesh.vertexCount; v++) {
			const MeshVertex& mv = part.vertices[v];
			mesh.vertices[v * 3 + 0] = mv.pos.x; mesh.vertices[v * 3 + 1] = mv.pos.y; mesh.vertices[v * 3 + 2] = mv.pos.z;
			mesh.texcoords[v * 2 + 0] = mv.uv.x; mesh.texcoords[v * 2 + 1] = mv.uv.y;
			mesh.normals[v * 3 + 0] = mv.normal.x; mesh.normals[v * 3 + 1] = mv.normal.y; mesh.normals[v * 3 + 2] = mv.normal.z;
			mesh.colors[v * 4 + 0] = mv.color.r; mesh.colors[v * 4 + 1] = mv.color.g;
			mesh.colors[v * 4 + 2] = mv.color.b; mesh.colors[v * 4 + 3] = mv.color.a;
		}
		memcpy(mesh.indices, part.indices.data(), part.indices.size() * sizeof(unsigned short));

		UploadMesh(&mesh, false);
		MemFree(mesh.vertices);  mesh.vertices = nullptr;
		MemFree(mesh.texcoords); mesh.texcoords = nullptr;
		MemFree(mesh.normals);   mesh.normals = nullptr;
		MemFree(mesh.colors);    mesh.colors = nullptr;
		model.meshes[i] = mesh;

		model.materials[i] = LoadMaterialDefault();
		SetMaterialTexture(&model.materials[i], MATERIAL_MAP_DIFFUSE, part.texture);
		model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color = part.tinted ? part.tintColor : WHITE;
		model.materials[i].shader = g_worldShader;
		model.meshMaterial[i] = i;
	}
	model.transform = MatrixIdentity();
	return model;
}

void UploadChunkMesh(Chunk& chunk, MeshData&& md) {
	if (chunk.hasMesh) { UnloadModel(chunk.model);
		chunk.hasMesh = false; }

	if (chunk.hasWaterMesh) {
		UnloadModel(chunk.waterModel);
		chunk.hasWaterMesh = false;
	}

	if (!md.parts.empty()) {
		chunk.model = BuildModelFromParts(md.parts, true);
		chunk.hasMesh = true;
	}

	if (!md.waterParts.empty()) {
		chunk.waterModel = BuildModelFromParts(md.waterParts, false);
		chunk.hasWaterMesh = true;
	}
}

struct ChunkBuildResult {
	int64_t key = 0;
	Chunk chunk;
	MeshData meshData;
	bool hasMeshData = false;
};

Chunk GenerateTerrainChunk(int chunkX, int chunkZ) {
	Chunk chunk;
	chunk.chunkX = chunkX;
	chunk.chunkZ = chunkZ;
	chunk.mode = Chunk::Mode::Dense;
	chunk.dense = new BlockId[CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE];

	auto idx = [](int x, int y, int z) { return (x * WORLD_HEIGHT + y) * CHUNK_SIZE + z; };
	auto getB = [&](int x, int y, int z) { return chunk.dense[idx(x, y, z)]; };
	auto setB = [&](int x, int y, int z, BlockId t) { chunk.dense[idx(x, y, z)] = t; };
	for (int i = 0; i < CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE; i++) chunk.dense[i] = g_AirId;

	int heights[CHUNK_SIZE][CHUNK_SIZE];

	for (int x = 0; x < CHUNK_SIZE; x++) {
		for (int z = 0; z < CHUNK_SIZE; z++) {
			int wx = chunkX * CHUNK_SIZE + x;
			int wz = chunkZ * CHUNK_SIZE + z;
			int surface = SurfaceHeightAt(wx, wz);
			heights[x][z] = surface;
			bool beach = surface <= SEA_LEVEL + 1;

			int top = std::max(surface, SEA_LEVEL);
			for (int y = 0; y <= top; y++) {
				BlockId t = g_AirId;
				if (y == 0) {
					t = g_BedrockId;
				}
				else if (y <= 4) {
					t = (Hash01(wx, y, wz, 11) < (5 - y) / 5.0f) ? g_BedrockId : g_known.stone;
				}
				else if (y <= surface) {
					if (IsCaveAt(wx, y, wz, surface))   t = g_AirId;
					else if (y < surface - 3)           t = g_known.stone;
					else if (y < surface)               t = beach ? g_known.sand : g_known.dirt;
					else                                t = beach ? g_known.sand : g_known.grass;
				}
				else {
					t = g_WaterId;
				}
				setB(x, y, z, t);
			}
		}
	}

	for (int x = 0; x < CHUNK_SIZE; x++) {
		for (int z = 0; z < CHUNK_SIZE; z++) {
			int s = heights[x][z];
			if (s + 1 >= WORLD_HEIGHT) continue;
			if (getB(x, s, z) != g_known.grass) continue;
			if (getB(x, s + 1, z) != g_AirId) continue;

			int wx = chunkX * CHUNK_SIZE + x, wz = chunkZ * CHUNK_SIZE + z;
			float r = Hash01(wx, s, wz, 77);
			if (r < 0.15f) setB(x, s + 1, z, g_known.shortGrass);
			else if (r < 0.16f) setB(x, s + 1, z, g_known.dandelion);
			else if (r < 0.17f) setB(x, s + 1, z, g_known.poppy);
		}
	}

	for (int x = 0; x < CHUNK_SIZE; x++) {
		for (int z = 0; z < CHUNK_SIZE; z++) {
			int surface = heights[x][z];
			if (surface + 1 >= WORLD_HEIGHT || getB(x, surface, z) != g_known.grass ||
				getB(x, surface + 1, z) != g_AirId) continue;
			int wx = chunkX * CHUNK_SIZE + x;
			int wz = chunkZ * CHUNK_SIZE + z;
			if (Hash01(wx, surface, wz, 333) < 0.0012f)
				setB(x, surface + 1, z, g_known.anvil);
		}
	}

	for (int wx : { 27, 30, 33, 36, 39 }) {
		const int wz = 32;
		if (wx < chunkX * CHUNK_SIZE || wx >= (chunkX + 1) * CHUNK_SIZE ||
			wz < chunkZ * CHUNK_SIZE || wz >= (chunkZ + 1) * CHUNK_SIZE) continue;
		int lx = wx - chunkX * CHUNK_SIZE;
		int lz = wz - chunkZ * CHUNK_SIZE;
		int surface = heights[lx][lz];
		if (surface + 1 < WORLD_HEIGHT && getB(lx, surface, lz) == g_known.grass)
			setB(lx, surface + 1, lz, g_known.anvil);
	}

	const int R = 2;
	for (int tx = -R; tx < CHUNK_SIZE + R; tx++) {
		for (int tz = -R; tz < CHUNK_SIZE + R; tz++) {
			int wx = chunkX * CHUNK_SIZE + tx;
			int wz = chunkZ * CHUNK_SIZE + tz;
			if (Hash01(wx, 0, wz, 500) > 0.012f) continue;

			int surface = SurfaceHeightAt(wx, wz);
			if (surface <= SEA_LEVEL + 1) continue;
			if (IsCaveAt(wx, surface, wz, surface)) continue;
			int h = 4 + (int)(Hash01(wx, 1, wz, 501) * 3.0f);
			if (surface + h + 2 >= WORLD_HEIGHT) continue;

			int top = surface + h;
			for (int dy = -2; dy <= 1; dy++) {
				int yy = top + dy;
				int r = (dy <= -1) ? 2 : 1;
				for (int dx = -r; dx <= r; dx++) {
					for (int dz = -r; dz <= r; dz++) {
						bool corner = (abs(dx) == r && abs(dz) == r);
						if (corner && (dy == 1 || Hash01(wx + dx, yy, wz + dz, 502) < 0.5f)) continue;
						int lx = tx + dx, lz = tz + dz;
						if (lx < 0 || lx >= CHUNK_SIZE || lz < 0 || lz >= CHUNK_SIZE) continue;
						if (getB(lx, yy, lz) == g_AirId)
							setB(lx, yy, lz, g_known.oakLeaves);
					}
				}
			}
			if (tx >= 0 && tx < CHUNK_SIZE && tz >= 0 && tz < CHUNK_SIZE) {
				for (int i = 1; i <= h; i++) setB(tx, surface + i, tz, g_known.oakLog);
			}
		}
	}

	chunk.Optimize();
	return chunk;
}

// Lower = higher priority. Square rings (Chebyshev), then angle for stable order.
float ChunkPriority(int dx, int dz) {
	int ring = (std::max)(abs(dx), abs(dz));
	float a = atan2f((float)dx, (float)-dz);
	if (a < 0.0f) a += 2.0f * PI;
	return ring * 1000.0f + a;
}

struct MeshJob { int64_t key; float priority; };
struct MeshJobCmp { bool operator()(const MeshJob& a, const MeshJob& b) const { return a.priority > b.priority; } };

std::priority_queue<MeshJob, std::vector<MeshJob>, MeshJobCmp> meshQueue;
std::unordered_set<int64_t> g_meshInFlight;

struct MeshBuildResult {
	int64_t key = 0;
	MeshData data;
};
std::queue<MeshBuildResult> finishedMeshes;

struct GenJob { int x, z; float priority; };
struct GenJobCmp { bool operator()(const GenJob& a, const GenJob& b) const { return a.priority > b.priority; } };

std::priority_queue<GenJob, std::vector<GenJob>, GenJobCmp> generationQueue;

std::unordered_set<int64_t> g_inFlight;
std::queue<ChunkBuildResult> finishedChunks;
std::mutex queueMutex;
std::condition_variable cv;
std::atomic<bool> workerRunning{ true };
std::vector<std::thread> workerThreads;

std::unordered_map<int64_t, Chunk>* g_worldChunksPtr = nullptr;


static void FlattenInto(const Chunk& src, Chunk& dst)
{
	if (&src == &dst)
		return;

	dst.ClearStorage();

	dst.chunkX = src.chunkX;
	dst.chunkZ = src.chunkZ;
	dst.dirtyGeometry = src.dirtyGeometry;
	dst.dirtyLight = src.dirtyLight;
	dst.mode = Chunk::Mode::Dense;

	const int N = CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE;

	dst.dense = new BlockId[N];

	for (int x = 0; x < CHUNK_SIZE; ++x)
	{
		for (int y = 0; y < WORLD_HEIGHT; ++y)
		{
			for (int z = 0; z < CHUNK_SIZE; ++z)
			{
				const int index = (x * WORLD_HEIGHT + y) * CHUNK_SIZE + z;
				dst.dense[index] = src.Get(x, y, z);
			}
		}
	}

	dst.lightSourcePositions = src.lightSourcePositions;
	dst.fluidLevels = src.fluidLevels;

	if (src.skyLight && src.blockLight)
	{
		dst.EnsureLightAllocated();

		memcpy(dst.skyLight, src.skyLight, N * sizeof(uint8_t));
		memcpy(dst.blockLight, src.blockLight, N * sizeof(uint8_t));
	}
}

void WorkerThreadFunc() {
	while (workerRunning) {
		bool didWork = false;

		// ---- Generation (preferred while the world is filling) ----
		{
			GenJob job;
			bool hasGenJob = false;
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				if (!generationQueue.empty() && finishedChunks.size() < 64) {
					job = generationQueue.top();
					generationQueue.pop();
					g_inFlight.insert(ChunkKey(job.x, job.z));
					hasGenJob = true;
				}
			}
			if (hasGenJob) {
				ChunkBuildResult result;
				result.key = ChunkKey(job.x, job.z);
				try {
					result.chunk = GenerateTerrainChunk(job.x, job.z);
				}
				catch (const std::exception& e) {
					TraceLog(LOG_ERROR, "Chunk gen failed: %s", e.what());
					std::lock_guard<std::mutex> lock(queueMutex);
					g_inFlight.erase(result.key);
					continue;
				}
				std::lock_guard<std::mutex> lock(queueMutex);
				finishedChunks.push(std::move(result));
				didWork = true;
				// After generating, immediately loop again to drain generation
				// queue before taking expensive mesh jobs.
				continue;
			}
		}

		// ---- Meshing ----
		{
			MeshJob mjob;
			bool hasMeshJob = false;
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				if (!meshQueue.empty() && finishedMeshes.size() < 48) {
					mjob = meshQueue.top();
					meshQueue.pop();
					g_meshInFlight.insert(mjob.key);
					hasMeshJob = true;
				}
			}
			if (hasMeshJob) {
				MeshData data;
				bool ok = false;
				std::unordered_map<int64_t, Chunk> localChunks;

				if (!g_worldChunksPtr) {
					std::lock_guard<std::mutex> lock(queueMutex);
					g_meshInFlight.erase(mjob.key);
					continue;
				}

				{
					std::shared_lock<std::shared_mutex> chunksLock(g_chunksMutex);
					auto it = g_worldChunksPtr->find(mjob.key);
					if (it != g_worldChunksPtr->end()) {
						int cx = it->second.chunkX;
						int cz = it->second.chunkZ;
						for (int dz = -1; dz <= 1; dz++) {
							for (int dx = -1; dx <= 1; dx++) {
								auto n = g_worldChunksPtr->find(ChunkKey(cx + dx, cz + dz));
								if (n == g_worldChunksPtr->end()) continue;
								Chunk copy;
								FlattenInto(n->second, copy);
								localChunks[n->first] = std::move(copy);
							}
						}
					}
				}

				auto cit = localChunks.find(mjob.key);
				if (cit != localChunks.end()) {
					try {
						data = BuildChunkMeshData(cit->second, localChunks);
						ok = true;
					}
					catch (const std::exception& e) {
						TraceLog(LOG_ERROR, "Mesh build failed: %s", e.what());
						ok = false;
					}
				}

				// Persist lightmaps / re-dirty on failure. Always chunksMutex first.
				{
					std::unique_lock<std::shared_mutex> chunksLock(g_chunksMutex);
					auto real = g_worldChunksPtr->find(mjob.key);
					if (real != g_worldChunksPtr->end()) {
						if (ok && cit != localChunks.end() && cit->second.skyLight) {
							real->second.EnsureLightAllocated();
							const int N = CHUNK_SIZE * WORLD_HEIGHT * CHUNK_SIZE;
							memcpy(real->second.skyLight, cit->second.skyLight, N);
							memcpy(real->second.blockLight, cit->second.blockLight, N);
						}
						if (!ok)
							real->second.MarkGeometryDirty();
					}
				}

				{
					std::lock_guard<std::mutex> lock(queueMutex);
					if (ok)
						finishedMeshes.push({ mjob.key, std::move(data) });
					g_meshInFlight.erase(mjob.key);
				}
				didWork = true;
			}
		}

		if (!didWork) {
			std::unique_lock<std::mutex> lock(queueMutex);
			cv.wait_for(lock, std::chrono::milliseconds(5), [] {
				return !generationQueue.empty() || !meshQueue.empty() || !workerRunning;
				});
		}
	}
}

// Mark chunks affected by a light/opacity change. radius=14 is max block-light level.
// geometryChange=true marks both geometry+light; false marks only light.
void DirtyChunksAroundBlock(std::unordered_map<int64_t, Chunk>& chunks, int bx, int by, int bz,
	int radius = 14, bool geometryChange = true)
{
	(void)by;
	int minCx = (int)floorf((float)(bx - radius) / (float)CHUNK_SIZE);
	int maxCx = (int)floorf((float)(bx + radius) / (float)CHUNK_SIZE);
	int minCz = (int)floorf((float)(bz - radius) / (float)CHUNK_SIZE);
	int maxCz = (int)floorf((float)(bz + radius) / (float)CHUNK_SIZE);

	for (int cx = minCx; cx <= maxCx; ++cx) {
		for (int cz = minCz; cz <= maxCz; ++cz) {
			auto it = chunks.find(ChunkKey(cx, cz));
			if (it == chunks.end()) continue;
			if (geometryChange)
				it->second.MarkGeometryDirty();
			else
				it->second.MarkLightDirty();
		}
	}
}

// ============================================================================
//  FLUID SIMULATION — классическая "flood fill" симуляция без давления.
//  Клетки без явной записи в Chunk::fluidLevels, содержащие воду/лаву —
//  неисчерпаемые источники (океаны/озёра из генерации мира), они никогда
//  не угасают. Явные записи (1..8) — "текущая" жидкость, которая угасает,
//  если теряет питание (соседей/клетку сверху того же типа).
// ============================================================================

struct FluidPos { int x, y, z; };
struct FluidPosHash {
	size_t operator()(const FluidPos& p) const {
		int64_t h = (int64_t)p.x * 73856093LL ^ (int64_t)p.y * 19349663LL ^ (int64_t)p.z * 83492791LL;
		return std::hash<int64_t>()(h);
	}
};
struct FluidPosEq {
	bool operator()(const FluidPos& a, const FluidPos& b) const {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};

std::queue<FluidPos> g_waterQueue, g_lavaQueue;
std::unordered_set<FluidPos, FluidPosHash, FluidPosEq> g_waterQueuedSet, g_lavaQueuedSet;
std::mutex g_fluidMutex;

void EnqueueFluidUpdate(int x, int y, int z, BlockId fluidType) {
	if (fluidType != g_WaterId && fluidType != g_LavaId) return;
	std::lock_guard<std::mutex> lock(g_fluidMutex);
	FluidPos p{ x, y, z };
	if (fluidType == g_WaterId) { if (g_waterQueuedSet.insert(p).second) g_waterQueue.push(p); }
	else { if (g_lavaQueuedSet.insert(p).second) g_lavaQueue.push(p); }
}

int GetFluidLevel(const std::unordered_map<int64_t, Chunk>& chunks, int x, int y, int z, BlockId fluidType) {
	int cx = (int)floorf((float)x / CHUNK_SIZE), cz = (int)floorf((float)z / CHUNK_SIZE);
	int lx = x - cx * CHUNK_SIZE, lz = z - cz * CHUNK_SIZE;
	auto it = chunks.find(ChunkKey(cx, cz));
	if (it == chunks.end()) return 0;
	if (it->second.Get(lx, y, lz) != fluidType) return 0;
	auto fit = it->second.fluidLevels.find(Chunk::LightIndex(lx, y, lz));
	if (fit == it->second.fluidLevels.end()) return MaxFluidLevel(fluidType); // неявный бесконечный источник
	return fit->second & 0x0F;
}

void ProcessFluidCell(std::unordered_map<int64_t, Chunk>& chunks, int x, int y, int z, BlockId fluidType) {
	if (y < 0 || y >= WORLD_HEIGHT) return;
	int cx = (int)floorf((float)x / CHUNK_SIZE), cz = (int)floorf((float)z / CHUNK_SIZE);
	int lx = x - cx * CHUNK_SIZE, lz = z - cz * CHUNK_SIZE;
	auto it = chunks.find(ChunkKey(cx, cz));
	if (it == chunks.end()) return;
	Chunk& chunk = it->second;
	BlockId current = chunk.Get(lx, y, lz);

	static const int ddx[4] = { 1,-1,0,0 };
	static const int ddz[4] = { 0,0,1,-1 };

	BlockId above = GetWorldBlock(chunks, cx, cz, lx, y + 1, lz);
	bool fedFromAbove = (above == fluidType);

	int bestNeighborLevel = 0;
	for (int i = 0; i < 4; i++) {
		BlockId n = GetWorldBlock(chunks, cx, cz, lx + ddx[i], y, lz + ddz[i]);
		if (n == fluidType) {
			int lvl = GetFluidLevel(chunks, x + ddx[i], y, z + ddz[i], fluidType);
			if (lvl > bestNeighborLevel) bestNeighborLevel = lvl;
		}
	}
	int maxLevel = MaxFluidLevel(fluidType);
	int computedLevel = fedFromAbove ? maxLevel : (bestNeighborLevel - 1);

	// воздух ИЛИ заменимый блок (цветы, трава, факелы и т.п.)
	if (current == g_AirId || IsReplaceableByFluid(current)) {
		if (computedLevel <= 0) return;
		chunk.Set(lx, y, lz, fluidType);
		chunk.fluidLevels[Chunk::LightIndex(lx, y, lz)] = (uint8_t)(computedLevel & 0x0F) | (fedFromAbove ? 0x10 : 0);
		DirtyChunksAroundBlock(chunks, x, y, z, 2, true);
		EnqueueFluidUpdate(x, y - 1, z, fluidType);
		for (int i = 0; i < 4; i++) EnqueueFluidUpdate(x + ddx[i], y, z + ddz[i], fluidType);
		return;
	}

	if (current != fluidType) return;

	auto flIt = chunk.fluidLevels.find(Chunk::LightIndex(lx, y, lz));
	bool hasExplicit = (flIt != chunk.fluidLevels.end());

	if (hasExplicit && computedLevel <= 0 && !fedFromAbove) {
		// жидкость потеряла питание — испаряется
		chunk.Set(lx, y, lz, g_AirId);
		chunk.fluidLevels.erase(flIt);
		DirtyChunksAroundBlock(chunks, x, y, z, 2, true);
		EnqueueFluidUpdate(x, y + 1, z, fluidType);
		EnqueueFluidUpdate(x, y - 1, z, fluidType);
		for (int i = 0; i < 4; i++) EnqueueFluidUpdate(x + ddx[i], y, z + ddz[i], fluidType);
		return;
	}

	if (hasExplicit) {
		uint8_t newPacked = (uint8_t)((fedFromAbove ? maxLevel : computedLevel) & 0x0F) | (fedFromAbove ? 0x10 : 0);
		if (flIt->second != newPacked) {
			flIt->second = newPacked;
			DirtyChunksAroundBlock(chunks, x, y, z, 2, true);
		}
	}

	int effectiveLevel = hasExplicit ? (fedFromAbove ? maxLevel : computedLevel) : maxLevel;

	BlockId below = GetWorldBlock(chunks, cx, cz, lx, y - 1, lz);
	if (below == g_AirId) {
		EnqueueFluidUpdate(x, y - 1, z, fluidType);
	}
	else if (effectiveLevel > 1) {
		for (int i = 0; i < 4; i++) {
			BlockId n = GetWorldBlock(chunks, cx, cz, lx + ddx[i], y, lz + ddz[i]);
			if (n == g_AirId) EnqueueFluidUpdate(x + ddx[i], y, z + ddz[i], fluidType);
		}
	}
}

void UpdateFluids(std::unordered_map<int64_t, Chunk>& chunks, float dt, float& waterAccum, float& lavaAccum) {
	constexpr float kWaterTick = 0.20f;
	constexpr float kLavaTick = 0.60f; // лава обновляется реже/медленнее воды, как просили
	constexpr int kMaxPerTick = 1500;

	waterAccum += dt;
	lavaAccum += dt;

	if (waterAccum >= kWaterTick) {
		waterAccum = 0.0f;
		std::unique_lock<std::shared_mutex> lock(g_chunksMutex);
		std::vector<FluidPos> batch;
		{
			std::lock_guard<std::mutex> qlock(g_fluidMutex);
			int n = (std::min)((int)g_waterQueue.size(), kMaxPerTick);
			batch.reserve(n);
			for (int i = 0; i < n; i++) { batch.push_back(g_waterQueue.front()); g_waterQueue.pop(); g_waterQueuedSet.erase(batch.back()); }
		}
		for (auto& p : batch) ProcessFluidCell(chunks, p.x, p.y, p.z, g_WaterId);
	}
	if (lavaAccum >= kLavaTick) {
		lavaAccum = 0.0f;
		std::unique_lock<std::shared_mutex> lock(g_chunksMutex);
		std::vector<FluidPos> batch;
		{
			std::lock_guard<std::mutex> qlock(g_fluidMutex);
			int n = (std::min)((int)g_lavaQueue.size(), kMaxPerTick);
			batch.reserve(n);
			for (int i = 0; i < n; i++) { batch.push_back(g_lavaQueue.front()); g_lavaQueue.pop(); g_lavaQueuedSet.erase(batch.back()); }
		}
		for (auto& p : batch) ProcessFluidCell(chunks, p.x, p.y, p.z, g_LavaId);
	}
}

void UpdateChunkStreaming(std::unordered_map<int64_t, Chunk>& chunks,
	std::unordered_map<int64_t, ChunkState>& states,
	int playerChunkX, int playerChunkZ,
	bool rescanNeeded,
	float freeTime)
{
	const double frameStart = GetTime();
	auto timeLeft = [&]() -> float {
		return freeTime - (float)(GetTime() - frameStart);
		};

	if (rescanNeeded) {
		std::vector<int64_t> toRemove;
		{
			std::shared_lock<std::shared_mutex> chunksLock(g_chunksMutex);
			for (auto& [key, chunk] : chunks) {
				if (abs(chunk.chunkX - playerChunkX) > g_renderDistance ||
					abs(chunk.chunkZ - playerChunkZ) > g_renderDistance)
					toRemove.push_back(key);
			}
		}

		if (!toRemove.empty() && timeLeft() > 0.0005f) {
			std::unique_lock<std::shared_mutex> chunksLock(g_chunksMutex);
			for (int64_t key : toRemove) chunks.erase(key);
			for (int64_t key : toRemove) states[key] = ChunkState::Unloaded;
		}

		if (timeLeft() > 0.001f) {
			// Snapshot which chunk keys already exist so we don't touch chunks
			// under queueMutex without the chunks lock.
			std::unordered_set<int64_t> existing;
			{
				std::shared_lock<std::shared_mutex> chunksLock(g_chunksMutex);
				existing.reserve(chunks.size());
				for (const auto& [key, _] : chunks)
					existing.insert(key);
			}

			std::lock_guard<std::mutex> lock(queueMutex);
			while (!generationQueue.empty())
				generationQueue.pop();

			// Priority: closer rings first, then angle for stable order around camera
			for (int dx = -g_renderDistance; dx <= g_renderDistance; dx++) {
				for (int dz = -g_renderDistance; dz <= g_renderDistance; dz++) {
					int nx = playerChunkX + dx, nz = playerChunkZ + dz;
					int64_t key = ChunkKey(nx, nz);
					if (existing.count(key) || g_inFlight.count(key)) continue;
					generationQueue.push({ nx, nz, ChunkPriority(dx, dz) });
					states[key] = ChunkState::Loading;
				}
			}
		}
		cv.notify_all();
	}

	// ---- Insert finished generated chunks (time-sliced) ----
	if (timeLeft() > 0.0008f) {
		std::vector<ChunkBuildResult> ready;
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			const int kMaxInserts = 16;
			while (!finishedChunks.empty() && (int)ready.size() < kMaxInserts) {
				ready.push_back(std::move(finishedChunks.front()));
				finishedChunks.pop();
			}
			for (auto& res : ready) g_inFlight.erase(res.key);
		}

		if (!ready.empty()) {
			std::unique_lock<std::shared_mutex> chunksLock(g_chunksMutex);
			for (auto& res : ready) {
				if (timeLeft() < 0.0003f) break;
				int cx = res.chunk.chunkX, cz = res.chunk.chunkZ;
				if (abs(cx - playerChunkX) > g_renderDistance || abs(cz - playerChunkZ) > g_renderDistance) {
					states[res.key] = ChunkState::Unloaded;
					continue;
				}

				chunks[res.key] = std::move(res.chunk);
				chunks[res.key].MarkGeometryDirty(); // new chunk always needs mesh
				states[res.key] = ChunkState::Loading;

				// Neighbours need light update only (geometry of neighbour didn't change)
				const int offsets[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };
				for (auto& [ox, oz] : offsets) {
					auto it = chunks.find(ChunkKey(cx + ox, cz + oz));
					if (it != chunks.end())
						it->second.MarkLightDirty();
				}
			}
		}
	}

	// ---- Queue mesh jobs for dirty chunks (priority around camera) ----
	if (timeLeft() > 0.0005f) {
		std::vector<MeshJob> jobs;
		{
			std::shared_lock<std::shared_mutex> chunksLock(g_chunksMutex);
			for (auto& [key, chunk] : chunks) {
				if (!chunk.NeedsRebuild()) continue;
				// Skip if already being meshed — will re-check after upload
				if (g_meshInFlight.count(key)) continue;
				jobs.push_back({ key, ChunkPriority(chunk.chunkX - playerChunkX, chunk.chunkZ - playerChunkZ) });
			}
		}
		// Sort by priority so closest chunks get meshed first
		std::sort(jobs.begin(), jobs.end(), [](const MeshJob& a, const MeshJob& b) {
			return a.priority < b.priority;
			});

		if (!jobs.empty()) {
			std::vector<int64_t> queuedKeys;
			{
				std::lock_guard<std::mutex> lock(queueMutex);
				for (auto& j : jobs) {
					if (g_meshInFlight.count(j.key)) continue;
					// Avoid flooding the queue
					if (meshQueue.size() + g_meshInFlight.size() > 96) break;
					meshQueue.push(j);
					queuedKeys.push_back(j.key);
				}
				cv.notify_all();
			}
			// Clear dirty flags only for successfully queued jobs.
			// If geometry/light changes again while in-flight, DirtyChunksAroundBlock
			// will re-set them and the chunk will be re-queued after upload.
			if (!queuedKeys.empty()) {
				std::unique_lock<std::shared_mutex> chunksLock(g_chunksMutex);
				for (int64_t key : queuedKeys) {
					auto it = chunks.find(key);
					if (it != chunks.end())
						it->second.MarkClean();
				}
			}
		}
	}

	// ---- Upload finished meshes (larger budget, still time-sliced) ----
	{
		// Adaptive budget: use up to 60% of remaining freeTime, but at least 1.5 ms, at most 4 ms
		const double kUploadBudgetSec = std::clamp((double)timeLeft() * 0.60, 0.0015, 0.004);
		const int kMaxUploads = 8;
		const double t0 = GetTime();
		int uploaded = 0;

		while (uploaded < kMaxUploads && (GetTime() - t0) < kUploadBudgetSec) {
			MeshBuildResult res;
			{
				std::lock_guard<std::mutex> lock(queueMutex);
				if (finishedMeshes.empty()) break;
				res = std::move(finishedMeshes.front());
				finishedMeshes.pop();
			}

			auto it = chunks.find(res.key);
			if (it != chunks.end()) {
				UploadChunkMesh(it->second, std::move(res.data));
				// If nothing re-dirtied the chunk while it was building, mark loaded.
				// If NeedsRebuild() is true again, it will be re-queued next frame.
				if (!it->second.NeedsRebuild())
					states[res.key] = ChunkState::Loaded;
			}

			uploaded++;
		}
	}
}

int CollectVisibleTorches(const std::vector<const Chunk*>& visible, Vector3 pos,
	Vector3* out, int maxCount)
{
	struct Item { float d2; Vector3 p; };
	std::vector<Item> items;
	for (const Chunk* c : visible) {
		for (const Vector3& p : c->lightSourcePositions) {
			items.push_back({ Vector3DistanceSqr(p, pos), p });
		}
	}
	if (items.empty()) return 0;
	std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.d2 < b.d2; });
	int n = (int)items.size();
	if (n > maxCount) n = maxCount;
	for (int i = 0; i < n; i++) out[i] = items[i].p;
	return n;
}

void _updateCreativeFly(Camera3D& camera, float& yaw, float& pitch,
	const std::unordered_map<int64_t, Chunk>& chunks,
	float& eyeHeight, Vector3& flyVel, float baseSpeed, float mouseSens)
{
	Vector2 mouseDelta = GetMouseDelta();
	yaw += mouseDelta.x * mouseSens;
	pitch -= mouseDelta.y * mouseSens;
	const float maxPitch = PI * 0.5f - 0.001f;
	pitch = Clamp(pitch, -maxPitch, maxPitch);

	Vector3 look = {
		cosf(pitch) * cosf(yaw),
		sinf(pitch),
		cosf(pitch) * sinf(yaw)
	};
	Vector3 flatForward = Vector3Normalize(Vector3{ look.x, 0.0f, look.z });
	Vector3 right = Vector3Normalize(Vector3CrossProduct(flatForward, { 0.0f, 1.0f, 0.0f }));

	float dt = GetFrameTime();
	if (dt > 0.05f) dt = 0.05f;

	eyeHeight += (1.8f - eyeHeight) * (std::min)(1.0f, dt * 14.0f);

	Vector3 wish = { 0.0f, 0.0f, 0.0f };
	if (IsKeyDown(KEY_W)) wish = Vector3Add(wish, flatForward);
	if (IsKeyDown(KEY_S)) wish = Vector3Subtract(wish, flatForward);
	if (IsKeyDown(KEY_D)) wish = Vector3Add(wish, right);
	if (IsKeyDown(KEY_A)) wish = Vector3Subtract(wish, right);
	if (IsKeyDown(KEY_SPACE)) wish.y += 1.0f;
	if (IsKeyDown(KEY_LEFT_SHIFT)) wish.y -= 1.0f;
	if (Vector3LengthSqr(wish) > 0.0f)
		wish = Vector3Normalize(wish);

	Vector3 target = Vector3Scale(wish, baseSpeed);
	float rate = (Vector3LengthSqr(wish) > 0.0f) ? 10.0f : 8.0f;
	float t = 1.0f - expf(-rate * dt);
	flyVel = Vector3Lerp(flyVel, target, t);

	auto isSolidAt = [&chunks](float x, float y, float z) {
		int bx = (int)floorf(x), by = (int)floorf(y), bz = (int)floorf(z);
		int cx = (int)floorf((float)bx / CHUNK_SIZE);
		int cz = (int)floorf((float)bz / CHUNK_SIZE);
		int lx = bx - cx * CHUNK_SIZE;
		int lz = bz - cz * CHUNK_SIZE;
		BlockId type = GetWorldBlock(chunks, cx, cz, lx, by, lz);
		return type != g_AirId && !IsFluidBlock(type) && type != g_known.torch && !IsCross(type);
		};
	auto canOccupy = [&isSolidAt](Vector3 pos, float height) {
		const float bodyRadius = 0.25f;
		const float feetY = pos.y - height;
		const float bodyHeight = height > 1.4f ? 1.8f : 1.25f;
		for (float x : { pos.x - bodyRadius, pos.x + bodyRadius }) {
			for (float z : { pos.z - bodyRadius, pos.z + bodyRadius }) {
				if (isSolidAt(x, feetY + 0.08f, z) ||
					isSolidAt(x, feetY + bodyHeight * 0.5f, z) ||
					isSolidAt(x, feetY + bodyHeight - 0.08f, z))
					return false;
			}
		}
		return true;
		};

	Vector3 movement = Vector3Scale(flyVel, dt);
	Vector3 next = camera.position;
	next.x += movement.x;
	if (canOccupy(next, eyeHeight)) camera.position.x = next.x;
	else flyVel.x = 0.0f;

	next = camera.position;
	next.z += movement.z;
	if (canOccupy(next, eyeHeight)) camera.position.z = next.z;
	else flyVel.z = 0.0f;

	next = camera.position;
	next.y += movement.y;
	if (canOccupy(next, eyeHeight)) camera.position.y = next.y;
	else flyVel.y = 0.0f;

	if (camera.position.y - eyeHeight < 0.0f) {
		camera.position.y = eyeHeight;
		flyVel.y = 0.0f;
	}
	camera.target = Vector3Add(camera.position, look);
}

void _updateSpectatorCamera(Camera3D& camera, float& yaw, float& pitch,
	float baseSpeed, float mouseSens)
{
	Vector2 mouseDelta = GetMouseDelta();
	yaw += mouseDelta.x * mouseSens;
	pitch -= mouseDelta.y * mouseSens;
	const float maxPitch = PI * 0.5f - 0.001f;
	pitch = Clamp(pitch, -maxPitch, maxPitch);

	Vector3 look = {
		cosf(pitch) * cosf(yaw),
		sinf(pitch),
		cosf(pitch) * sinf(yaw)
	};
	Vector3 right = Vector3CrossProduct(look, { 0.0f, 1.0f, 0.0f });
	if (Vector3LengthSqr(right) < 1e-8f)
		right = Vector3CrossProduct(look, { 0.0f, 0.0f, 1.0f });
	right = Vector3Normalize(right);

	float dt = GetFrameTime();
	if (dt > 0.05f) dt = 0.05f;

	Vector3 wish = { 0.0f, 0.0f, 0.0f };
	if (IsKeyDown(KEY_W)) wish = Vector3Add(wish, look);
	if (IsKeyDown(KEY_S)) wish = Vector3Subtract(wish, look);
	if (IsKeyDown(KEY_D)) wish = Vector3Add(wish, right);
	if (IsKeyDown(KEY_A)) wish = Vector3Subtract(wish, right);
	if (Vector3LengthSqr(wish) > 0.0f)
		wish = Vector3Normalize(wish);

	camera.position = Vector3Add(camera.position, Vector3Scale(wish, baseSpeed * dt));
	camera.target = Vector3Add(camera.position, look);
}

void _updateWalkCamera(Camera3D& camera, float& yaw, float& pitch,
	const std::unordered_map<int64_t, Chunk>& chunks, float& verticalVelocity, float& eyeHeight,
	Vector3& walkVel, float mouseSens)
{
	Vector2 mouseDelta = GetMouseDelta();
	yaw += mouseDelta.x * mouseSens;
	pitch -= mouseDelta.y * mouseSens;
	const float maxPitch = PI * 0.5f - 0.001f;
	pitch = Clamp(pitch, -maxPitch, maxPitch);

	Vector3 forward = { cosf(pitch) * cosf(yaw), sinf(pitch), cosf(pitch) * sinf(yaw) };
	Vector3 flatForward = Vector3Normalize(Vector3{ forward.x, 0.0f, forward.z });
	Vector3 right = Vector3Normalize(Vector3CrossProduct(flatForward, { 0.0f, 1.0f, 0.0f }));

	float dt = GetFrameTime();
	if (dt > 0.05f) dt = 0.05f;

	const bool crouching = IsKeyDown(KEY_LEFT_SHIFT);
	const bool sprinting = IsKeyDown(KEY_LEFT_CONTROL);
	float moveSpeed = crouching ? 1.3f : (sprinting ? 5.6f : 4.3f);
	const float targetEyeHeight = crouching ? 1.12f : 1.8f;
	const float oldEyeHeight = eyeHeight;
	eyeHeight += (targetEyeHeight - eyeHeight) * (std::min)(1.0f, dt * 14.0f);
	camera.position.y += eyeHeight - oldEyeHeight;

	auto fluidAt = [&chunks](float x, float y, float z) -> BlockId {
		int bx = (int)floorf(x), by = (int)floorf(y), bz = (int)floorf(z);
		int cx = (int)floorf((float)bx / CHUNK_SIZE);
		int cz = (int)floorf((float)bz / CHUNK_SIZE);
		int lx = bx - cx * CHUNK_SIZE;
		int lz = bz - cz * CHUNK_SIZE;
		return GetWorldBlock(chunks, cx, cz, lx, by, lz);
		};
	BlockId eyeFluid = fluidAt(camera.position.x, camera.position.y, camera.position.z);
	BlockId feetFluid = fluidAt(camera.position.x, camera.position.y - eyeHeight + 0.1f, camera.position.z);
	const bool headSubmerged = IsFluidBlock(eyeFluid);
	const bool inLava = (eyeFluid == g_LavaId || feetFluid == g_LavaId);
	const bool swimming = headSubmerged || IsFluidBlock(feetFluid);
	if (swimming) moveSpeed *= (inLava ? 0.35f : 0.5f);

	const float radius = 0.25f;
	const float groundProbe = 0.075f;

	auto isSolidAt = [&chunks](float x, float y, float z) {
		int bx = (int)floorf(x), by = (int)floorf(y), bz = (int)floorf(z);
		int cx = (int)floorf((float)bx / CHUNK_SIZE);
		int cz = (int)floorf((float)bz / CHUNK_SIZE);
		int lx = bx - cx * CHUNK_SIZE;
		int lz = bz - cz * CHUNK_SIZE;
		BlockId type = GetWorldBlock(chunks, cx, cz, lx, by, lz);
		return type != g_AirId && !IsFluidBlock(type) && type != g_known.torch && !IsCross(type);
		};
	auto canOccupy = [&isSolidAt](Vector3 pos, float height) {
		const float bodyRadius = 0.25f;
		const float feetY = pos.y - height;
		const float bodyHeight = height > 1.4f ? 1.8f : 1.25f;
		for (float x : { pos.x - bodyRadius, pos.x + bodyRadius }) {
			for (float z : { pos.z - bodyRadius, pos.z + bodyRadius }) {
				if (isSolidAt(x, feetY + 0.08f, z) ||
					isSolidAt(x, feetY + bodyHeight * 0.5f, z) ||
					isSolidAt(x, feetY + bodyHeight - 0.08f, z))
					return false;
			}
		}
		return true;
		};
	auto getSupportTop = [&chunks](float x, float z, float feet) -> float {
		const int bx = (int)floorf(x);
		const int bz = (int)floorf(z);
		const int belowY = (int)floorf(feet - 0.001f);
		int cx = (int)floorf((float)bx / CHUNK_SIZE);
		int cz = (int)floorf((float)bz / CHUNK_SIZE);
		int lx = bx - cx * CHUNK_SIZE;
		int lz = bz - cz * CHUNK_SIZE;
		for (int by = belowY; by >= belowY - 1; --by) {
			const float top = (float)by + 1.0f;
			BlockId type = GetWorldBlock(chunks, cx, cz, lx, by, lz);
			if (type != g_AirId && !IsFluidBlock(type) && type != g_known.torch && !IsCross(type) && top <= feet + 0.001f)
				return top;
		}
		return -100000.0f;
		};
	auto getSupportY = [&](Vector3 pos, float height) {
		const float feetY = pos.y - height;
		float supportY = -100000.0f;
		for (float sx : { -radius + 0.025f, radius - 0.025f }) {
			for (float sz : { -radius + 0.025f, radius - 0.025f }) {
				float top = getSupportTop(pos.x + sx, pos.z + sz, feetY + groundProbe);
				if (top > supportY && top <= feetY + groundProbe) supportY = top;
			}
		}
		return supportY;
		};

	const float feetY = camera.position.y - eyeHeight;
	float supportY = getSupportY(camera.position, eyeHeight);
	const bool grounded = verticalVelocity <= 0.0f && supportY > -99999.0f && feetY <= supportY + groundProbe;

	Vector3 wish = { 0.0f, 0.0f, 0.0f };
	if (IsKeyDown(KEY_W)) wish = Vector3Add(wish, flatForward);
	if (IsKeyDown(KEY_S)) wish = Vector3Subtract(wish, flatForward);
	if (IsKeyDown(KEY_D)) wish = Vector3Add(wish, right);
	if (IsKeyDown(KEY_A)) wish = Vector3Subtract(wish, right);
	if (Vector3LengthSqr(wish) > 0.0f)
		wish = Vector3Scale(Vector3Normalize(wish), moveSpeed);

	float rate = grounded ? 14.0f : 4.0f;
	float t = 1.0f - expf(-rate * dt);
	walkVel.x += (wish.x - walkVel.x) * t;
	walkVel.z += (wish.z - walkVel.z) * t;
	walkVel.y = 0.0f;

	Vector3 movement = { walkVel.x * dt, 0.0f, walkVel.z * dt };

	auto allowHorizontal = [&](Vector3 next) {
		if (!canOccupy(next, eyeHeight)) return false;
		if (crouching && grounded) {
			float nextFeet = next.y - eyeHeight;
			float nextSupport = getSupportY(next, eyeHeight);
			if (!(nextSupport > -99999.0f && nextFeet <= nextSupport + groundProbe))
				return false;
		}
		return true;
		};

	Vector3 next = camera.position;
	next.x += movement.x;
	if (allowHorizontal(next)) camera.position.x = next.x;
	else walkVel.x = 0.0f;

	next = camera.position;
	next.z += movement.z;
	if (allowHorizontal(next)) camera.position.z = next.z;
	else walkVel.z = 0.0f;

	if (swimming) {
		const float sinkRate = inLava ? 7.0f : 10.0f;    // лава гуще — медленнее реагирует
		const float riseAccel = inLava ? 7.0f : 10.0f;
		const float maxRiseSpeed = inLava ? 1.1f : 2.0f;
		const float maxSinkSpeed = inLava ? -1.0f : -1.5f;

		if (IsKeyDown(KEY_SPACE)) {
			verticalVelocity += riseAccel * dt;
			if (verticalVelocity > maxRiseSpeed) verticalVelocity = maxRiseSpeed;
		}
		else if (IsKeyDown(KEY_LEFT_SHIFT)) {
			verticalVelocity -= sinkRate * dt;
			if (verticalVelocity < maxSinkSpeed) verticalVelocity = maxSinkSpeed;
		}
		else {
			// без ввода — плавучесть: если голова под жидкостью, медленно всплываем;
			// если уже на поверхности, просто гасим скорость у кромки
			float buoyancy = headSubmerged ? (inLava ? 1.5f : 2.5f) : -4.0f;
			verticalVelocity += buoyancy * dt;
			verticalVelocity = Clamp(verticalVelocity, maxSinkSpeed, maxRiseSpeed);
		}

		Vector3 nextVertical = camera.position;
		nextVertical.y += verticalVelocity * dt;
		if (canOccupy(nextVertical, eyeHeight)) camera.position.y = nextVertical.y;
		else verticalVelocity = 0.0f;
	}
	else if (grounded) {
		camera.position.y = supportY + eyeHeight;
		verticalVelocity = 0.0f;
		if (IsKeyDown(KEY_SPACE)) verticalVelocity = 8.0f;
	}
	else {
		verticalVelocity -= 24.0f * dt;
		const float nextFeetY = feetY + verticalVelocity * dt;
		float landingY = -100000.0f;
		if (verticalVelocity <= 0.0f && nextFeetY <= feetY) {
			for (float sx : { -radius + 0.025f, radius - 0.025f }) {
				for (float sz : { -radius + 0.025f, radius - 0.025f }) {
					float top = getSupportTop(camera.position.x + sx, camera.position.z + sz, feetY + 0.001f);
					if (top > landingY && top <= feetY + 0.001f && nextFeetY <= top + 0.02f)
						landingY = top;
				}
			}
		}
		if (landingY > -99999.0f) {
			camera.position.y = landingY + eyeHeight;
			verticalVelocity = 0.0f;
			if (IsKeyDown(KEY_SPACE)) verticalVelocity = 8.0f;
		}
		else {
			Vector3 nextVertical = camera.position;
			nextVertical.y += verticalVelocity * dt;
			if (canOccupy(nextVertical, eyeHeight)) camera.position.y = nextVertical.y;
			else {
				if (verticalVelocity > 0.0f) verticalVelocity = 0.0f;
			}
		}
	}
	if (camera.position.y - eyeHeight < 0.0f) {
		camera.position.y = eyeHeight;
		verticalVelocity = 0.0f;
	}
	camera.target = Vector3Add(camera.position, forward);
}

Color GetChunkStateColor(ChunkState state) {
	switch (state) {
	case ChunkState::Loaded:    return GREEN;
	case ChunkState::Loading:   return YELLOW;
	case ChunkState::Unloaded:  return RED;
	default:                    return BLACK;
	}
}

std::string getCameraLookAtDirection(Camera3D& camera) {
	Vector3 forward = Vector3Subtract(camera.target, camera.position);
	forward.y = 0.0f;

	if (Vector3Length(forward) < 0.001f)
		return "Up/Down";

	float angleRad = atan2f(forward.x, forward.z);
	float angleDeg = angleRad * RAD2DEG;

	if (angleDeg < 0.0f)
		angleDeg += 360.0f;

	if (angleDeg >= 315.0f || angleDeg < 45.0f) return "South";
	if (angleDeg >= 45.0f && angleDeg < 135.0f) return "East";
	if (angleDeg >= 135.0f && angleDeg < 225.0f) return "North";
	if (angleDeg >= 225.0f && angleDeg < 315.0f) return "West";

	return "Unknown";
}

size_t GetProcessRSS_MB() {
#if defined(_WIN32)
	PROCESS_MEMORY_COUNTERS_EX pmc{};
	if (GetProcessMemoryInfo(GetCurrentProcess(),
		reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
		sizeof(pmc)))
	{
		return static_cast<size_t>(pmc.WorkingSetSize / (1024 * 1024));
	}
#endif
	return 0;
}

size_t EstimateTextureVRAM(const Texture2D& tex) {
	if (tex.id == 0) return 0;
	size_t bytes = (size_t)tex.width * tex.height * 4;
	if (tex.mipmaps > 1) {
		bytes = bytes * 4 / 3;
	}
	return bytes;
}

size_t EstimateMeshVRAM(const Mesh& mesh) {
	return (size_t)mesh.vertexCount * (3 * 4 + 2 * 4 + 3 * 4 + 4)
		+ (size_t)mesh.triangleCount * 3 * sizeof(unsigned short);
}

size_t EstimateModelVRAM(const Model& model) {
	size_t bytes = 0;
	for (int i = 0; i < model.meshCount; i++) {
		bytes += EstimateMeshVRAM(model.meshes[i]);
	}
	return bytes;
}

int main() {
	InitWindow(1920, 1040, "Sandbox 3D");
	SetTargetFPS(0);

	rlImGuiSetup(true);

	g_worldShader = LoadShaderFromMemory(kWorldVS, kWorldFS);
	g_dayFactorLoc = GetShaderLocation(g_worldShader, "dayFactor");

	Camera3D camera;
	camera.position = { 32.0f, 90.0f, 32.0f };
	camera.target = { 32.0f, 40.0f, 32.0f };
	camera.up = { 0, 1, 0 };
	camera.fovy = 60;
	camera.projection = CAMERA_PERSPECTIVE;

	float yaw = 0.0f;
	float pitch = -0.6f;
	float baseSpeed = 8.0f;
	float mouseSensitivity = 0.0022f;
	Vector3 flyVelocity = { 0.0f, 0.0f, 0.0f };
	Vector3 walkVelocity = { 0.0f, 0.0f, 0.0f };

	bool mouseLocked = true;
	bool showDebugInformation = false;
	bool isFast = false;
	int gameMode = 2;
	bool creativeFlying = true;
	float lastSpaceTapTime = -10.0f;
	float verticalVelocity = 0.0f;
	float eyeHeight = 1.8f;
	int selectedBlockIndex = 0;

	int worldSeed = GenerateRandomSeed();
	std::cout << "World seed: " << worldSeed << std::endl;

	g_worldSeed = worldSeed;
	InitTerrainNoise(worldSeed);
	InitCaveNoise(worldSeed);

	std::unordered_map<int64_t, Chunk> worldChunks;
	std::map<std::string, Texture2D> textureCache;

	g_blocks.LoadAll("resources/blockstates", textureCache);
	ConfigureBlockSupportTypes();
	ResolveKnownBlockIds();
	BuildPlaceableBlockList();

	g_waterTexture = GetOrLoadTexture(textureCache, "minecraft:block/water_still");
	SetTextureFilter(g_waterTexture, TEXTURE_FILTER_POINT);

	g_lavaTexture = GetOrLoadTexture(textureCache, "minecraft:block/lava_still");
	SetTextureFilter(g_lavaTexture, TEXTURE_FILTER_POINT);

	g_worldChunksPtr = &worldChunks;

	unsigned int hc = std::thread::hardware_concurrency();
	unsigned int threadCount = hc == 0 ? 4u : hc;
	if (threadCount > 1) threadCount -= 1;
	if (threadCount > 6) threadCount = 6;
	if (threadCount < 2) threadCount = 2;
	for (unsigned int i = 0; i < threadCount; i++) {
		workerThreads.emplace_back(WorkerThreadFunc);
	}

	int playerChunkX = (int)floor(camera.position.x / CHUNK_SIZE);
	int playerChunkZ = (int)floor(camera.position.z / CHUNK_SIZE);
	int lastPlayerChunkX = playerChunkX;
	int lastPlayerChunkZ = playerChunkZ;
	int lastStreamRenderDistance = g_renderDistance;

	std::unordered_map<int64_t, ChunkState> chunkStates;
	UpdateChunkStreaming(worldChunks, chunkStates, playerChunkX, playerChunkZ, true, 0.016f);

	float dayTime = 0.25f;
	float waterTickAccum = 0.0f, lavaTickAccum = 0.0f;

	while (!WindowShouldClose()) {
		float deltaTime = GetFrameTime();
		float freeTime = 1.0f / 60.0f - deltaTime;
		if (freeTime < 0.001f) freeTime = 0.001f; // always leave a tiny slice

		dayTime = fmodf(dayTime + deltaTime / 240.0f, 1.0f);
		float sunAngle = dayTime * 2.0f * PI;
		g_sunDirection = Vector3Normalize(Vector3{ cosf(sunAngle), -sinf(sunAngle), 0.25f });
		g_dayFactor = Clamp((sinf(sunAngle) + 0.12f) / 0.32f, 0.0f, 1.0f);

		if (mouseLocked) {
			if (gameMode == 1 && IsKeyPressed(KEY_SPACE)) {
				float now = (float)GetTime();
				if (now - lastSpaceTapTime <= 0.30f) {
					creativeFlying = !creativeFlying;
					verticalVelocity = 0.0f;
					flyVelocity = { 0.0f, 0.0f, 0.0f };
					walkVelocity = { 0.0f, 0.0f, 0.0f };
					lastSpaceTapTime = -10.0f;
				}
				else {
					lastSpaceTapTime = now;
				}
			}

			if (gameMode == 2) {
				const float specSpeed = isFast ? 35.0f : baseSpeed;
				_updateSpectatorCamera(camera, yaw, pitch, specSpeed, mouseSensitivity);
			}
			else if (gameMode == 1 && creativeFlying) {
				const float flySpeed = isFast ? 16.0f : baseSpeed;
				_updateCreativeFly(camera, yaw, pitch, worldChunks, eyeHeight, flyVelocity, flySpeed, mouseSensitivity);
			}
			else {
				_updateWalkCamera(camera, yaw, pitch, worldChunks, verticalVelocity, eyeHeight, walkVelocity, mouseSensitivity);
			}
		}

		playerChunkX = (int)floor(camera.position.x / CHUNK_SIZE);
		playerChunkZ = (int)floor(camera.position.z / CHUNK_SIZE);

		bool playerChunkChanged = (playerChunkX != lastPlayerChunkX || playerChunkZ != lastPlayerChunkZ);
		bool distanceChanged = (g_renderDistance != lastStreamRenderDistance);
		bool rescanNeeded = playerChunkChanged || distanceChanged;

		UpdateChunkStreaming(worldChunks, chunkStates, playerChunkX, playerChunkZ, rescanNeeded, freeTime);
		UpdateFluids(worldChunks, deltaTime, waterTickAccum, lavaTickAccum);

		Vector3 lookDirection = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
		BlockHit targetedBlock = RaycastBlock(worldChunks, camera.position, lookDirection, 8.0f);
		if (mouseLocked && IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && targetedBlock.hit
			&& targetedBlock.type != g_BedrockId)
		{
			int cx = (int)floorf((float)targetedBlock.x / CHUNK_SIZE);
			int cz = (int)floorf((float)targetedBlock.z / CHUNK_SIZE);
			int lx = targetedBlock.x - cx * CHUNK_SIZE;
			int lz = targetedBlock.z - cz * CHUNK_SIZE;

			{
				std::unique_lock<std::shared_mutex> lock(g_chunksMutex);
				auto chunkIt = worldChunks.find(ChunkKey(cx, cz));
				if (chunkIt != worldChunks.end()) {
					// Removing any block (especially a torch or an opaque block)
					// can change light propagation. Mark a full light-radius
					// of chunks dirty so their baked vertex lighting is rebuilt.
					chunkIt->second.Set(lx, targetedBlock.y, lz, g_AirId);
					DirtyChunksAroundBlock(worldChunks, targetedBlock.x, targetedBlock.y, targetedBlock.z);
					RemoveUnsupportedBlocks(worldChunks, targetedBlock.x, targetedBlock.y, targetedBlock.z);
					EnqueueFluidUpdate(targetedBlock.x, targetedBlock.y, targetedBlock.z, g_WaterId);
					EnqueueFluidUpdate(targetedBlock.x, targetedBlock.y, targetedBlock.z, g_LavaId);
					targetedBlock.hit = false;
				}
			}
		}

		if (mouseLocked && IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) && targetedBlock.hit
			&& targetedBlock.type != g_known.torch && !IsCross(targetedBlock.type)) {
			int px = targetedBlock.x + targetedBlock.nx;
			int py = targetedBlock.y + targetedBlock.ny;
			int pz = targetedBlock.z + targetedBlock.nz;
			int pcx = (int)floorf((float)px / CHUNK_SIZE);
			int pcz = (int)floorf((float)pz / CHUNK_SIZE);
			int plx = px - pcx * CHUNK_SIZE;
			int plz = pz - pcz * CHUNK_SIZE;
			if (py >= 0 && py < WORLD_HEIGHT && !g_placeableBlocks.empty()) {
				std::unique_lock<std::shared_mutex> lock(g_chunksMutex);
				auto placeChunk = worldChunks.find(ChunkKey(pcx, pcz));
				BlockId at = GetWorldBlock(worldChunks, pcx, pcz, plx, py, plz);
				if (placeChunk != worldChunks.end() && (at == g_AirId || IsCross(at))) {
					const float playerRadius = 0.25f;
					const float playerFeetY = camera.position.y - eyeHeight;
					const float playerHeight = 1.8f;
					const bool overlapsPlayer =
						camera.position.x + playerRadius > (float)px &&
						camera.position.x - playerRadius < (float)px + 1.0f &&
						playerFeetY + playerHeight > (float)py &&
						playerFeetY < (float)py + 1.0f &&
						camera.position.z + playerRadius > (float)pz &&
						camera.position.z - playerRadius < (float)pz + 1.0f;
					if (!overlapsPlayer) {
						BlockId toPlace = g_placeableBlocks[selectedBlockIndex];
						if (GetSupportType(toPlace) == SupportType::None ||
							BlockHasRequiredSupport(worldChunks, px, py, pz, toPlace)) {
							placeChunk->second.Set(plx, py, plz, toPlace);
							DirtyChunksAroundBlock(worldChunks, px, py, pz);
							RemoveUnsupportedBlocks(worldChunks, px, py, pz);
							if (toPlace == g_WaterId || toPlace == g_LavaId)
								EnqueueFluidUpdate(px, py, pz, toPlace);
						}
					}
				}
			}
		}

		lastPlayerChunkX = playerChunkX;
		lastPlayerChunkZ = playerChunkZ;
		lastStreamRenderDistance = g_renderDistance;

		if (IsKeyPressed(KEY_F3)) {
			showDebugInformation = !showDebugInformation;
		}

		if (IsKeyPressed(KEY_LEFT_ALT)) {
			mouseLocked = !mouseLocked;
			if (mouseLocked) DisableCursor();
			else EnableCursor();
		}

		if (mouseLocked)
			DisableCursor();

		isFast = IsKeyDown(KEY_LEFT_CONTROL);

		Matrix matView = MatrixLookAt(camera.position, camera.target, camera.up);
		float aspect = (float)GetScreenWidth() / (float)GetScreenHeight();
		Matrix matProj = MatrixPerspective(camera.fovy * DEG2RAD, aspect, 0.1f, 800.0f);
		Matrix matViewProj = MatrixMultiply(matView, matProj);

		Frustum frustum;
		ExtractFrustum(frustum, matViewProj);

		std::vector<const Chunk*> visible;
		std::unordered_set<int64_t> regionsToCheck;
		for (const auto& [key, chunk] : worldChunks) {
			int rx = (int)floor((float)chunk.chunkX / REGION_SIZE);
			int rz = (int)floor((float)chunk.chunkZ / REGION_SIZE);
			regionsToCheck.insert(RegionKey(rx, rz));
		}

		int drawnChunks = 0;
		int culledByRegion = 0;
		int culledByFrustum = 0;

		for (int64_t rkey : regionsToCheck) {
			int rx = (int)(rkey >> 32);
			int rz = (int)(rkey & 0xFFFFFFFF);

			Vector3 rMin, rMax;
			GetRegionAABB(rx, rz, rMin, rMax);

			if (!IsAABBInFrustum(frustum, rMin, rMax)) {
				culledByRegion += REGION_SIZE * REGION_SIZE;
				continue;
			}

			for (int dx = 0; dx < REGION_SIZE; dx++) {
				for (int dz = 0; dz < REGION_SIZE; dz++) {
					int cx = rx * REGION_SIZE + dx;
					int cz = rz * REGION_SIZE + dz;
					int64_t ckey = ChunkKey(cx, cz);

					auto it = worldChunks.find(ckey);
					if (it == worldChunks.end()) continue;

					const Chunk& chunk = it->second;

					Vector3 cMin, cMax;
					GetChunkAABB(cx, cz, cMin, cMax);

					if (!IsAABBInFrustum(frustum, cMin, cMax)) {
						culledByFrustum++;
						continue;
					}

					visible.push_back(&chunk);
					drawnChunks++;
				}
			}
		}

		SetShaderValue(g_worldShader, g_dayFactorLoc, &g_dayFactor, SHADER_UNIFORM_FLOAT);

		BeginDrawing();

		Color daySky = { 112, 174, 255, 255 };
		Color nightSky = { 8, 12, 35, 255 };
		Color sky = {
			(unsigned char)(nightSky.r + (daySky.r - nightSky.r) * g_dayFactor),
			(unsigned char)(nightSky.g + (daySky.g - nightSky.g) * g_dayFactor),
			(unsigned char)(nightSky.b + (daySky.b - nightSky.b) * g_dayFactor), 255 };
		ClearBackground(sky);

		BeginMode3D(camera);

		Vector3 celestialCenter = Vector3Add(camera.position, Vector3Scale(Vector3Normalize(Vector3{ -g_sunDirection.x, -g_sunDirection.y, -g_sunDirection.z }), 170.0f));
		DrawSphere(celestialCenter, 7.0f, Color{ 255, 225, 145, 255 });
		Vector3 moonCenter = Vector3Add(camera.position, Vector3Scale(Vector3Normalize(g_sunDirection), 165.0f));
		DrawSphere(moonCenter, 5.0f, Color{ 205, 218, 255, (unsigned char)(255.0f * (1.0f - g_dayFactor)) });

		for (const Chunk* c : visible)
			if (c->hasMesh) DrawModel(c->model, { 0, 0, 0 }, 1.0f, WHITE);

		rlDisableBackfaceCulling();
		for (const Chunk* c : visible)
			if (c->hasWaterMesh) DrawModel(c->waterModel, { 0, 0, 0 }, 1.0f, WHITE);
		rlEnableBackfaceCulling();

		if (targetedBlock.hit) {
			const VoxelShape& shape = GetBlockShape(targetedBlock.type);
			if (!shape.boxes.empty()) {
				Vector3 blockOrigin{ (float)targetedBlock.x, (float)targetedBlock.y, (float)targetedBlock.z };
				DrawBlockOutline(shape, blockOrigin, camera.position, worldChunks);
			}
		}
		EndMode3D();

		rlImGuiBegin();

		ImGui::Begin("Debug");
		ImGui::Text("FPS: %d", GetFPS());
		ImGui::Text("Frame time: %.3f ms", GetFrameTime() * 1000.0f);
		ImGui::Text("Mouse locked: %s", mouseLocked ? "yes" : "no");
		const char* gameModes[] = { "Survival", "Creative", "Spectator" };
		if (ImGui::Combo("Game Mode", &gameMode, gameModes, IM_ARRAYSIZE(gameModes))) {
			verticalVelocity = 0.0f;
			eyeHeight = 1.8f;
			flyVelocity = { 0.0f, 0.0f, 0.0f };
			walkVelocity = { 0.0f, 0.0f, 0.0f };
			creativeFlying = false;
			lastSpaceTapTime = -10.0f;
		}
		if (gameMode == 1)
			ImGui::Text("Creative fly: %s ", creativeFlying ? "ON" : "OFF");
		ImGui::SliderFloat("Mouse Sensitivity", &mouseSensitivity, 0.0005f, 0.006f, "%.4f");
		ImGui::SliderFloat("Fly Speed", &baseSpeed, 2.0f, 20.0f, "%.1f");
		ImGui::SliderInt("Render Distance", &g_renderDistance, 2, 128);
		{
			static char blockFilter[64] = "";
			const char* currentLabel = (selectedBlockIndex >= 0 && selectedBlockIndex < (int)g_placeableBlocks.size())
				? g_blocks.Get(g_placeableBlocks[selectedBlockIndex]).name.c_str() : "none";
			if (ImGui::BeginCombo("Place Block", currentLabel)) {
				ImGui::InputText("##filter", blockFilter, sizeof(blockFilter));
				for (int i = 0; i < (int)g_placeableBlocks.size(); i++) {
					const std::string& name = g_blocks.Get(g_placeableBlocks[i]).name;
					if (blockFilter[0] != '\0' && name.find(blockFilter) == std::string::npos) continue;
					bool isSelected = (selectedBlockIndex == i);
					if (ImGui::Selectable(name.c_str(), isSelected)) selectedBlockIndex = i;
					if (isSelected) ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		ImGui::End();

		ImGui::Begin("Chunk Map");
		{
			int mapRadius = (std::min)(g_renderDistance, 24) + 2;
			float cellSize = 6.0f;

			ImDrawList* drawList = ImGui::GetWindowDrawList();
			ImVec2 origin = ImGui::GetCursorScreenPos();

			for (int dz = -mapRadius; dz <= mapRadius; dz++) {
				for (int dx = -mapRadius; dx <= mapRadius; dx++) {
					int cx = playerChunkX + dx;
					int cz = playerChunkZ + dz;
					int64_t key = ChunkKey(cx, cz);

					ChunkState state = ChunkState::NotLoaded;
					auto it = chunkStates.find(key);
					if (it != chunkStates.end()) state = it->second;

					Color c = GetChunkStateColor(state);
					ImU32 imColor = IM_COL32(c.r, c.g, c.b, 255);

					float px = origin.x + (dx + mapRadius) * cellSize;
					float py = origin.y + (dz + mapRadius) * cellSize;

					drawList->AddRectFilled(ImVec2(px, py), ImVec2(px + cellSize - 1, py + cellSize - 1), imColor);
				}
			}

			ImGui::Dummy(ImVec2((mapRadius * 2 + 1) * cellSize, (mapRadius * 2 + 1) * cellSize));
		}
		ImGui::End();

		ImGui::Begin("Memory Debug");
		{
			static double lastRssTime = 0.0;
			static size_t rssMB = 0;
			double now = GetTime();
			if (now - lastRssTime > 0.5) {
				rssMB = GetProcessRSS_MB();
				lastRssTime = now;
			}
			ImGui::Text("Process RAM (RSS): %zu MB", rssMB);

			ImGui::Separator();

			size_t chunksStorage = 0;
			size_t chunksMeshVRAM = 0;
			int loadedChunks = 0;
			int singleChunks = 0;
			int layeredChunks = 0;
			int denseChunks = 0;

			for (const auto& [key, chunk] : worldChunks) {
				chunksStorage += chunk.GetStorageBytes();
				if (chunk.hasMesh)
					chunksMeshVRAM += EstimateModelVRAM(chunk.model);

				if (chunk.hasWaterMesh)
					chunksMeshVRAM += EstimateModelVRAM(chunk.waterModel);

				loadedChunks++;
				switch (chunk.mode) {
				case Chunk::Mode::Single:   singleChunks++; break;
				case Chunk::Mode::Layered:  layeredChunks++; break;
				case Chunk::Mode::Dense: denseChunks++; break;
				}
			}

			ImGui::Text("Chunks loaded: %d", loadedChunks);
			ImGui::Text("  Single:  %d", singleChunks);
			ImGui::Text("  Layered: %d", layeredChunks);
			ImGui::Text("  Dense:   %d", denseChunks);

			ImGui::Text("Block storage (RAM): %.2f KB", chunksStorage / 1024.0f);

			ImGui::Separator();

			size_t texturesVRAM = 0;
			for (const auto& [id, tex] : textureCache) {
				texturesVRAM += EstimateTextureVRAM(tex);
			}
			ImGui::Text("Textures VRAM: %.2f KB (%zu tex)", texturesVRAM / 1024.0f, textureCache.size());
			ImGui::Text("Chunk meshes VRAM: %.2f KB", chunksMeshVRAM / 1024.0f);

			size_t totalVRAM = texturesVRAM + chunksMeshVRAM;
			ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
				"Total estimated VRAM: %.2f MB", totalVRAM / (1024.0f * 1024.0f));

			ImGui::Separator();
			ImGui::TextDisabled("Note: VRAM is estimated (OpenGL doesn't expose exact usage easily)");
		}
		ImGui::End();

		rlImGuiEnd();

		int centerX = GetScreenWidth() / 2;
		int centerY = GetScreenHeight() / 2;
		BeginBlendMode(BLEND_CUSTOM);
		rlSetBlendFactors(RL_ONE_MINUS_DST_COLOR, RL_ZERO, RL_FUNC_ADD);
		DrawLine(centerX - 7, centerY, centerX - 2, centerY, WHITE);
		DrawLine(centerX + 2, centerY, centerX + 7, centerY, WHITE);
		DrawLine(centerX, centerY - 7, centerX, centerY - 2, WHITE);
		DrawLine(centerX, centerY + 2, centerX, centerY + 7, WHITE);
		DrawPixel(centerX, centerY, WHITE);
		EndBlendMode();

		if (showDebugInformation) {
			std::string dir = getCameraLookAtDirection(camera);
			DrawText(TextFormat("Direction: %s", dir.c_str()), 10, 10, 30, WHITE);

			if (std::isfinite(camera.position.x) &&
				std::isfinite(camera.position.y) &&
				std::isfinite(camera.position.z)) {
				DrawText(TextFormat("XYZ: %.3f, %.3f, %.3f",
					camera.position.x, camera.position.y, camera.position.z),
					10, 45, 24, WHITE);
			}
			else {
				DrawText("XYZ: N/A, N/A, N/A", 10, 45, 24, WHITE);
			}
		}

		EndDrawing();
	}

	workerRunning = false;
	cv.notify_all();
	for (auto& t : workerThreads) {
		if (t.joinable()) t.join();
	}

	g_worldChunksPtr = nullptr;

	worldChunks.clear();

	for (auto& [id, tex] : textureCache) {
		UnloadTexture(tex);
	}
	UnloadShader(g_worldShader);

	rlImGuiShutdown();
	CloseWindow();

	return 0;
}