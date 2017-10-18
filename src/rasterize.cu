/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>


namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct RenderSubdivision {
		int tileOffset;
		int tileCount; // Amount of tiles
		int tileSize; // Size of tiles at this level
	};

	struct RenderTile {
		int tileLength; // The size in pixels
		int tileLevel;	// Subdivision level
		int capacity; // The amount of primitives that can hold
		int primitiveOffset; // The index on the primitive buffer
		int currentIndex; // The current last primitive set
		glm::ivec2 from; // AABB min
		glm::ivec2 to; // AABB max
		int parentIndex;
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		// glm::vec3 col;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		// int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
		glm::vec3 min;
		glm::vec3 max;
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		// VertexAttributeTexcoord texcoord0;
		// TextureData* dev_diffuseTex;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static int TILE_SIZE = 4; // In pixels
static int TILE_PRIMITIVE_CAPACITY_BASE = 4096;

static int width = 0;
static int height = 0;
static int effectiveTileSubdivisions = 0;
static int totalTiles = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int * dev_fragmentMutex = NULL;

static RenderTile * dev_tile_headers = NULL; // Tile information
static int * dev_tile_primitives = NULL; // The tile primitive indices
static RenderSubdivision  * dev_subdivisions = NULL; // The offsets for each subdivision level

static RenderSubdivision * subdivisionData = NULL; // Client!

static int * dev_depth = NULL;	// you might need this buffer when doing depth test


/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = fragmentBuffer[index].color;

		// TODO: add your fragment shader code here
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    
	cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaMalloc(&dev_fragmentMutex, width * height * sizeof(int));
	cudaMemset(dev_fragmentMutex, 0, width * height * sizeof(int));

	int maxDimension = glm::max(w, h);
	int maxSubdivisions = 16;
	int tileHeaderMemory = 0;
	int tilePrimitiveMemory = 0;

	// Find the amount of subdivisions we will need
	for (int i = 0; i < maxSubdivisions; i++)
	{
		int tileSize = TILE_SIZE * glm::pow(2, i);
		int tileCount = glm::ceil(maxDimension / (float)tileSize);

		totalTiles += tileCount * tileCount;
		tilePrimitiveMemory += tileCount * tileCount * TILE_PRIMITIVE_CAPACITY_BASE * sizeof(int);
		
		if (tileSize >= maxDimension)
		{
			effectiveTileSubdivisions = i;
			break;
		}
	}

	effectiveTileSubdivisions = glm::max(effectiveTileSubdivisions, 1);
	
	// Build the render tile header data
	int tileOffset = 0;
	RenderTile * tileHeaderData = new RenderTile[totalTiles];
	subdivisionData = new RenderSubdivision[effectiveTileSubdivisions];

	int currentPrimitiveOffset = 0;

	for (int i = 0; i < effectiveTileSubdivisions; i++)
	{
		RenderTile tile;
		tile.tileLevel = i;
		tile.tileLength = TILE_SIZE * glm::pow(2, i);
		tile.currentIndex = 0; // No primitives yet! This index must be cleared on each frame

		int tileCount = glm::ceil(maxDimension / (float)tile.tileLength);
		tile.capacity = TILE_PRIMITIVE_CAPACITY_BASE; // For now, tile capacity is constant (TODO: make it dynamic)

		// Precompute some information for this subdivision level
		subdivisionData[i].tileOffset = tileOffset;
		subdivisionData[i].tileCount = tileCount;
		subdivisionData[i].tileSize = tile.tileLength;

		for (int y = 0; y < tileCount; ++y)
		{
			for (int x = 0; x < tileCount; ++x)
			{
				tile.from = glm::clamp(glm::vec2(x * tile.tileLength, y * tile.tileLength), glm::vec2(0.f), glm::vec2(width, height));
				tile.to = glm::clamp(glm::vec2((x+1) * tile.tileLength, (y+1) * tile.tileLength), glm::vec2(0.f), glm::vec2(width, height));
				tile.primitiveOffset = currentPrimitiveOffset + ((y * tileCount) + x) * tile.capacity;

				// Precompute the next parent tile index
				int parentX = x / 2;
				int parentY = y / 2;
				tile.parentIndex = tileOffset + (tileCount * tileCount) + parentY * (tileCount/2) + parentX;

				tileHeaderData[tileOffset + (y * tileCount) + x] = tile;
			}
		}

		printf("Tile offset for level %d : %d, primitive offset: %d, (buffer size: %d), capacity: %d \n", i, tileOffset, currentPrimitiveOffset,(tileCount * tileCount), tile.capacity);

		currentPrimitiveOffset += tileCount * tileCount * tile.capacity;
		tileOffset += tileCount * tileCount;
	}

	tileHeaderMemory = totalTiles * sizeof(RenderTile);
	
	printf("Size [%d,%d] | Tile subdivisions: %d \n", w, h, effectiveTileSubdivisions);
	printf("Tile header memory: %f MB | Tile primitive memory: %f MB \n", tileHeaderMemory / (1024.f * 1024.f), (tilePrimitiveMemory / (1024.f * 1024.f)));
	
	int subdivisionMemory = effectiveTileSubdivisions * sizeof(RenderSubdivision);

	cudaMalloc(&dev_subdivisions, subdivisionMemory);
	checkCUDAError("Alloc tile subdivisions");
	cudaMemcpy(dev_subdivisions, subdivisionData, subdivisionMemory, cudaMemcpyHostToDevice);
	checkCUDAError("Copy tile subdivisions");

	cudaMalloc(&dev_tile_headers, tileHeaderMemory);
	checkCUDAError("Alloc tile headers");
	cudaMemcpy(dev_tile_headers, tileHeaderData, tileHeaderMemory, cudaMemcpyHostToDevice);
	checkCUDAError("Copy tile headers");

	cudaMalloc(&dev_tile_primitives, tilePrimitiveMemory);
	cudaMemset(dev_tile_primitives, -1, tilePrimitiveMemory); // -1 means invalid index!
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	checkCUDAError("rasterizeInit"); 
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}
}

__global__ 
void _vertexTransformAndAssembly(int numVertices, PrimitiveDevBufPointers primitive,  glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, int width, int height) 
{
	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) 
	{
		glm::vec3 p = primitive.dev_position[vid];
		glm::vec3 n = primitive.dev_normal[vid];
		glm::vec3 eyeNormal = MV_normal * n;
		glm::vec3 eyePos = glm::vec3(MV * glm::vec4(p, 1.f));

		glm::vec4 ssPos = MVP * glm::vec4(p, 1.f);
		ssPos /= ssPos.w;
		ssPos.x = (ssPos.x * .5f + .5f) * width;
		ssPos.y = (ssPos.y * -.5f + .5f) * height;

		VertexOut out;
		out.pos = ssPos;
		out.eyePos = eyePos;
		out.eyeNor = eyeNormal;

		primitive.dev_verticesOut[vid] = out;

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
	}
}

__global__
void clearTileIndices(int totalTileCount, RenderTile * dev_tile_header)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totalTileCount)
		dev_tile_header[index].currentIndex = 0;
}

// Pineda
__forceinline__
__host__ __device__
float edgeFunction(glm::vec2 &a, glm::vec2  &b, glm::vec2 &c)
{
	return ((c[0] - a[0]) * (b[1] - a[1])) - ((c[1] - a[1]) * (b[0] - a[0]));
}

__global__
void rasterizeTiles(int numTiles, int numSubdivisions, int width, int height, RenderSubdivision * dev_subdivisions, RenderTile * dev_tile_header, int * dev_tile_primitives, Primitive* dev_primitives, Fragment *dev_fragmentBuffer, int * dev_fragmentMutex)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numTiles)
	{
		// Base tile is always smallest
		RenderTile & baseTile = dev_tile_header[index];

		for (int y = baseTile.from.y + 1; y <= baseTile.to.y - 1; y++)
		{
			for (int x = baseTile.from.x + 1; x <= baseTile.to.x - 1; x++)
			{
				glm::vec2 point = glm::vec2(x, y);
				Fragment resultFragment;

				RenderTile * tile = &baseTile;
				for (int i = 0; i < numSubdivisions; i++)
				{
					int totalPrimitives = glm::min(tile->currentIndex, tile->capacity);

					for (int p = 0; p < totalPrimitives; p++)
					{
						int primitiveIndex = dev_tile_primitives[tile->primitiveOffset + p];
						Primitive & prim = dev_primitives[primitiveIndex];

						if (x >= prim.min.x && x <= prim.max.x && y >= prim.min.y && y <= prim.max.y)
						{
							glm::vec2 v0 = glm::vec2(prim.v[0].pos);
							glm::vec2 v1 = glm::vec2(prim.v[1].pos);
							glm::vec2 v2 = glm::vec2(prim.v[2].pos);

							bool inside = true;
							inside &= edgeFunction(v0, v1, point) > 0.f;
							inside &= edgeFunction(v1, v2, point) > 0.f;
							inside &= edgeFunction(v2, v0, point) > 0.f;

							if (inside)
							{
								resultFragment.color = glm::vec3(1.f);// glm::vec3(glm::abs(prim.v[0].eyeNor.z));
							}
						}
					}

					// Jump to parent tile
					if (i < numSubdivisions)
						tile = &dev_tile_header[tile->parentIndex];
				}
				
				int fragIndex = y * width + x;
				dev_fragmentBuffer[fragIndex] = resultFragment;
			}
		}
	}
}

__global__
void updateTiles(int numPrimitives, int w, int h, int tileSubdivisions, int baseTileSize, Primitive* dev_primitives, 
	RenderTile * dev_tile_header, int * dev_tile_primitives, RenderSubdivision  * dev_subdivisions)
{
	int primitiveIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIndex < numPrimitives)
	{
		Primitive & p = dev_primitives[primitiveIndex];
		glm::vec4 v1 = p.v[0].pos;
		glm::vec4 v2 = p.v[1].pos;
		glm::vec4 v3 = p.v[2].pos;

		// Get the AABB (with Z too, for early reject)
		p.min = glm::min(glm::vec3(v1), glm::min(glm::vec3(v2), glm::vec3(v3)));
		p.max = glm::max(glm::vec3(v1), glm::max(glm::vec3(v2), glm::vec3(v3)));

		glm::vec2 screenMin = glm::vec2(p.min);
		glm::vec2 screenMax = glm::vec2(p.max);

		glm::vec2 screenSize = glm::abs(glm::vec2(p.max) - glm::vec2(p.min));

		for (int i = 0; i < tileSubdivisions; ++i)
		{
			RenderSubdivision & subdiv = dev_subdivisions[i];

			glm::vec2 tileSize = glm::vec2(subdiv.tileSize, subdiv.tileSize);
			glm::ivec2 sizeAtResolution = glm::ceil(screenSize / tileSize);

			int affectedTiles = glm::max(sizeAtResolution.x, sizeAtResolution.y);

			// If the size of this triangle is comparable to the tile size, stop at this level
			// Also stop if this is the last subdivisions (__very__ large triangles)
			if (affectedTiles <= 4 || i == tileSubdivisions - 1)
			{
				int tileCount = dev_subdivisions[i].tileCount;
				int tileOffset = dev_subdivisions[i].tileOffset;

				// Make sure we don't go out of bounds for this level
				glm::ivec2 from = glm::clamp(glm::floor(screenMin / tileSize), glm::vec2(0), glm::vec2(tileCount));
				glm::ivec2 to = glm::clamp(glm::ceil(screenMax / tileSize), glm::vec2(0), glm::vec2(tileCount));

				// Now write into the tile buffer
				for (int y = from.y; y <= to.y; ++y)
				{
					for (int x = from.x; x <= to.x; ++x)
					{
						RenderTile & tile = dev_tile_header[tileOffset + (tileCount * y) + x];

						// Get the list head and set this primitive
						int lastIndex = atomicAdd(&tile.currentIndex, 1);

						// We don't really care if the index goes above this point, we just care about not setting
						// memory outside this array
						if(lastIndex < tile.capacity)
							dev_tile_primitives[tile.primitiveOffset + lastIndex] = primitiveIndex;
					}
				}

				// We already found our level, don't do anything else
				return;
			}
		}
	}
}

static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) 
{
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) 
	{
		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
		// TODO: other primitive types (point, line)
	}
}



/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentMutex, 0, width * height * sizeof(int));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	// Clear tile indices
	dim3 blockSizeTiles(64);
	dim3 blockCountTiles((totalTiles - 1) / blockSizeTiles.x + 1);
	clearTileIndices << <blockCountTiles, blockSizeTiles >> >(totalTiles, dev_tile_headers);

	// Update tile data
	dim3 numThreadsPerBlockTiles(64);
	dim3 blockCountForPrimitives((totalNumPrimitives - 1) / numThreadsPerBlockTiles.x + 1);
	updateTiles << <blockCountForPrimitives, numThreadsPerBlockTiles >> > (totalNumPrimitives, width, height, effectiveTileSubdivisions,
		TILE_SIZE, dev_primitives, dev_tile_headers, dev_tile_primitives, dev_subdivisions);

	// Rasterize tiles
	int totalTiles = subdivisionData[0].tileCount * subdivisionData[0].tileCount;
	dim3 blockCountForRasterization((totalTiles - 1) / numThreadsPerBlockTiles.x + 1);
	rasterizeTiles << <blockCountForRasterization, numThreadsPerBlockTiles >> > (totalTiles, effectiveTileSubdivisions, width, height, dev_subdivisions, dev_tile_headers, dev_tile_primitives, dev_primitives, dev_fragmentBuffer, dev_fragmentMutex);

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
